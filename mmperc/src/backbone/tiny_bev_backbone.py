import torch.nn as nn


class TinyBEVBackbone(nn.Module):
    """
    A lightweight BEV backbone suitable for small GPUs.
    Input:  (B, C_in, H, W)
    Output: (B, C_out, H/2, W/2)
    """

    def __init__(self, in_channels=64, mid_channels=64, out_channels=128):
        super().__init__()

        # Normalize input channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Two basic conv blocks
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

        # Downsample to reduce memory + increase receptive field
        self.down = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final refinement block
        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.down(x)
        x = self.block3(x)
        return x
