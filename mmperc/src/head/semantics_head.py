from torch import Tensor, nn


class FullResSemHead(nn.Module):
    """
    Lightweight full-resolution semantic segmentation head.

    Input:
        feat: (B, C, H/8, W/8)

    Output:
        logits: (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        # Reduce channels for efficiency
        self.reduce = nn.Conv2d(in_channels, 64, kernel_size=1)

        # Upsample 1: H/8 → H/4
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # Upsample 2: H/4 → H/2
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # Upsample 3: H/2 → H
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # Final prediction
        self.pred = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, feat: Tensor) -> Tensor:
        x = self.reduce(feat)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.pred(x)
        return logits
