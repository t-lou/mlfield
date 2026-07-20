import torch.nn as nn
from torch import Tensor

import common.params as params


class TinyBEVBackbone(nn.Module):
    """
    Lightweight BEV backbone for memory‑constrained setups.

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
        in_channels: int = 64,
        mid_channels: int = 64,
        out_channels: int = 128,
    ) -> None:
        super().__init__()

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Local feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Spatial downsampling + channel expansion
        self.down = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=params.BACKBONE_STRIDE, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final refinement
        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.down(x)
        x = self.block3(x)
        return x
