import common.params as params
import torch
import torch.nn as nn


# Note: This Head is dummy and probably not doable with the available data.
class DrivableAreaHead(nn.Module):
    """
    Lightweight drivable-area segmentation head.
    Input:  (B, C, H, W)
    Output: (B, 1, H, W)  sigmoid mask
    """

    def __init__(self, mid_channels=64):
        super().__init__()

        # First refinement block
        self.block1 = nn.Sequential(
            nn.Conv2d(params.BEV_CHANNELS, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Second refinement block
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Final 1×1 conv → 1 channel mask
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, bev):
        """
        bev: (B, C, H, W)
        """
        x = self.block1(bev)
        x = self.block2(x)
        logits = self.out_conv(x)
        mask = torch.sigmoid(logits)
        return mask
