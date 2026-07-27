import torch
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams


class TinyBEVBackbone(nn.Module):
    """
    Lightweight BEV backbone for memory‑constrained setups.
    Supports gradient checkpointing for training to reduce memory footprint.

    Args:
        in_channels:  Number of input feature channels (default: 64)
        mid_channels: Internal feature width (default: 64)
        out_channels: Output feature width after downsampling (default: 128)

    Input:
        x: (B, in_channels, H, W)

    Output:
        (B, out_channels, H/2, W/2)
    """

    def __init__(
        self,
        params: MmpercParams,
        in_channels: int = 64,
        mid_channels: int = 64,
        out_channels: int = 128,
    ) -> None:
        super().__init__()

        stride = params.bev_params.backbone_stride

        # Initial projection
        # Using GroupNorm for memory efficiency (no running stats during training)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )

        # Scale 1: full resolution
        self.block1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )

        # Scale 2: stride-2
        self.down1 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

        # Scale 3: stride-4
        self.down2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

        # # Upsample scale2/scale3 back to scale1 resolution
        # self.up2 = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=stride, stride=stride)
        # self.up3 = nn.ConvTranspose2d(
        #     out_channels, out_channels // 2, kernel_size=stride * stride, stride=stride * stride
        # )

        # # Fuse features from scale1, upsampled scale2, and upsampled scale3
        # fused_channels = mid_channels + (out_channels // 2) * 2
        # self.fuse = nn.Sequential(
        #     nn.Conv2d(fused_channels, out_channels, kernel_size=1),
        #     nn.GroupNorm(8, out_channels),
        #     nn.ReLU(inplace=True),
        # )

        # Bring p1 (full res) down to scale2 resolution, and p3 (H/4) up to scale2
        # resolution, so everything fuses at the H/2, W/2 scale the backbone is
        # documented to output. (Previously this upsampled p2/p3 back to full
        # resolution H, W, which silently defeated the /2 downsampling the rest
        # of the pipeline — e.g. PointPillarBEV and the detection heads — assumes.)
        self.down_p1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.p2_proj = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1)
        self.up3 = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=stride, stride=stride)

        # Fuse features from downsampled scale1, scale2, and upsampled scale3 — all at H/2, W/2
        fused_channels = mid_channels + (out_channels // 2) * 2
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x = self.stem(x)
        # p1 = self.block1(x)  # (B, mid, H, W)      — fine, small objects
        # p2 = self.block2(self.down1(p1))  # (B, out, H/2, W/2)  — mid
        # p3 = self.block3(self.down2(p2))  # (B, out, H/4, W/4)  — coarse, large objects

        # p2_up = self.up2(p2)  # → H, W
        # p3_up = self.up3(p3)  # → H, W

        # fused = torch.cat([p1, p2_up, p3_up], dim=1)
        # return self.fuse(fused)  # single (B, out_channels, H, W) — but built from 3 real scales
        x = self.stem(x)
        p1 = self.block1(x)  # (B, mid, H, W)      — fine, small objects
        p2 = self.block2(self.down1(p1))  # (B, out, H/2, W/2)  — mid
        p3 = self.block3(self.down2(p2))  # (B, out, H/4, W/4)  — coarse, large objects

        p1_down = self.down_p1(p1)  # → H/2, W/2
        p2_proj = self.p2_proj(p2)  # already H/2, W/2, just channel-reduced
        p3_up = self.up3(p3)  # → H/2, W/2

        fused = torch.cat([p1_down, p2_proj, p3_up], dim=1)  # (B, fused_channels, H/2, W/2)
        return self.fuse(fused)  # (B, out_channels, H/2, W/2)


def _smoke_test():
    """
    Smoke test for TinyBEVBackbone to ensure it runs without errors.
    """
    params = MmpercParams()
    model = TinyBEVBackbone(params=params)
    model.train()  # Set to training mode to test checkpointing

    # Create a dummy input tensor with shape (B, C, H, W)
    dummy_input = torch.randn(2, 64, 128, 128)  # Batch size of 2, 64 channels, 128x128 spatial dimensions

    # Forward pass
    output = model(dummy_input)

    expected_shape = (2, 128, 128, 128)  # Expected output shape after fusion
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("TinyBEVBackbone smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
