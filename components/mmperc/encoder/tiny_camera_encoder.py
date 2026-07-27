import torch
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable conv block for better accuracy-per-memory.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
            ),
            nn.GroupNorm(8 if in_channels >= 8 else 1, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class TinyCameraEncoder(nn.Module):
    """
    Minimal camera encoder.

    Converts an RGB image:
        (B, 3, H, W)

    Into a sequence of camera tokens:
        (B, N_cam, C)
    and a feature map:
        (B, C, H', W')

    where:
        - N_cam = H' * W' after downsampling
        - C = out_channels (default 128)
    """

    def __init__(self, params: MmpercParams) -> None:
        super().__init__()

        self.out_channels = params.bev_params.bev_channels

        # Stage 1: 1/2 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 1/4 resolution
        self.stage2 = DepthwiseSeparableConv(32, 64, stride=2)

        # Stage 3: 1/8 resolution
        self.stage3 = DepthwiseSeparableConv(64, self.out_channels, stride=2)

        # Lightweight top-down multi-scale fusion.
        self.s4_lateral = nn.Conv2d(64, self.out_channels, kernel_size=1, bias=False)
        self.s4_to_s8 = DepthwiseSeparableConv(self.out_channels, self.out_channels, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, padding=1, groups=self.out_channels, bias=False
            ),
            nn.GroupNorm(8, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.out_channels),
            nn.ReLU(inplace=True),
        )

        # LayerNorm applied after flattening into tokens
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: RGB image tensor of shape (B, 3, H, W)

        Returns:
            tokens: (B, N_cam, out_channels)
                    where N_cam = H' * W' after downsampling
            feat:   (B, out_channels, H', W')
        """
        s2: Tensor = self.stem(x)  # (B, 32, H/2, W/2)
        s4: Tensor = self.stage2(s2)  # (B, 64, H/4, W/4)
        s8: Tensor = self.stage3(s4)  # (B, C, H/8, W/8)

        # Bring a cheap mid-resolution context path into 1/8 scale.
        s4_lat = self.s4_lateral(s4)
        s4_ds = self.s4_to_s8(s4_lat)
        feat: Tensor = self.fuse(s8 + s4_ds)
        B, C, H2, W2 = feat.shape

        # Flatten spatial dimensions → sequence of tokens
        # (B, C, H', W') → (B, H'*W', C)
        tokens: Tensor = feat.flatten(2).transpose(1, 2)

        # Normalize token embeddings
        tokens = self.norm(tokens)

        return tokens, feat


def _smoke_test():
    """
    Smoke test for TinyCameraEncoder.
    """
    B, C_in, H, W = 2, 3, 256, 256
    x = torch.rand(B, C_in, H, W)
    params = MmpercParams()
    encoder = TinyCameraEncoder(params=params)
    tokens, feat = encoder(x)
    assert tokens.shape[0] == B and tokens.shape[2] == params.bev_params.bev_channels
    assert feat.shape[0] == B and feat.shape[1] == params.bev_params.bev_channels
    print("TinyCameraEncoder smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
