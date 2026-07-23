from torch import nn


class FullResSemHead(nn.Module):
    """
    Lightweight full-resolution semantic segmentation head.
    Uses Upsample + Conv instead of ConvTranspose for better memory efficiency.

    Input:
        feat: (B, C, H/8, W/8)

    Output:
        logits: (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.reduce = nn.Conv2d(in_channels, 64, kernel_size=1)

        # H/8 → H/4: Upsample + Conv instead of ConvTranspose for memory efficiency
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # H/4 → H/2
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # H/2 → H
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.pred = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, feat):
        x = self.reduce(feat)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.pred(x)
