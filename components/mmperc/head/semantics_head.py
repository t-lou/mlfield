import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FullResSemHead(nn.Module):
    """
    Lightweight full-resolution semantic segmentation head.
    Uses Upsample + Conv instead of ConvTranspose for better memory efficiency.
    Fuses in camera skip features (s4, s2) from TinyCameraEncoder, U-Net style.

    Input:
        feat: (B, C, H/8, W/8)
        skip_feats: {
            "s4": (B, skip_channels, H/4, W/4),
            "s2": (B, skip_channels, H/2, W/2),
        }

    Output:
        logits: (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int, num_classes: int, skip_channels: int | None = None):
        super().__init__()

        skip_channels = skip_channels if skip_channels is not None else in_channels

        self.reduce = nn.Conv2d(in_channels, 64, kernel_size=1)

        # H/8 → H/4: Upsample + Conv instead of ConvTranspose for memory efficiency

        # H/8 → H/4: Conv, then resize-to-target instead of ConvTranspose (memory efficient)
        # NOTE: the resize step is done in forward() via F.interpolate(..., size=...)
        # rather than a fixed nn.Upsample(scale_factor=2). The encoder computes s2/s4/s8
        # independently via successive stride-2 convs, each with its own floor-rounding,
        # so their sizes are not guaranteed to be an exact x2 relationship unless the
        # original image H/W is a clean multiple of 8. Resizing to the skip's actual
        # shape makes this robust to arbitrary input resolutions.
        self.up1_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + skip_channels, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # H/4 → H/2
        self.up2_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        self.fuse2 = nn.Sequential(
            nn.Conv2d(32 + skip_channels, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # H/2 → H
        self.up3_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        self.pred = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, feat: Tensor, skip_feats: dict[str, Tensor], out_size: tuple[int, int] | None = None) -> Tensor:
        s4_hw = skip_feats["s4"].shape[-2:]
        s2_hw = skip_feats["s2"].shape[-2:]

        # Full output resolution: derive from s2 (x2) unless caller passes the true image size.
        if out_size is None:
            out_size = (s2_hw[0] * 2, s2_hw[1] * 2)

        x = self.reduce(feat)  # (B, 64, H/8, W/8)
        x = self.up1_conv(x)
        x = F.interpolate(x, size=s4_hw, mode="bilinear", align_corners=False)  # -> exact H/4
        x = self.fuse1(torch.cat([x, skip_feats["s4"]], dim=1))

        x = self.up2_conv(x)
        x = F.interpolate(x, size=s2_hw, mode="bilinear", align_corners=False)  # -> exact H/2
        x = self.fuse2(torch.cat([x, skip_feats["s2"]], dim=1))

        x = self.up3_conv(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)  # -> exact H
        return self.pred(x)


def _smoke_test():
    """
    Smoke test for FullResSemHead.
    """
    B, C_in, H, W = 2, 64, 32, 32
    num_classes = 10
    feat = torch.rand(B, C_in, H // 8, W // 8)
    skip_feats = {
        "s4": torch.rand(B, C_in, H // 4, W // 4),
        "s2": torch.rand(B, C_in, H // 2, W // 2),
    }
    head = FullResSemHead(in_channels=C_in, num_classes=num_classes, skip_channels=C_in)
    logits = head(feat, skip_feats)
    assert logits.shape == (B, num_classes, H, W), f"Expected shape (B, {num_classes}, H, W), got {logits.shape}"
    print("FullResSemHead smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
