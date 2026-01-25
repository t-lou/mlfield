import torch
from torch import Tensor, nn


class BBox2dHead(nn.Module):
    """
    Predicts:
      - heatmap: (B, 1, H, W)
      - box regression: (B, 6, H, W)
        [dx, dy, log(w), log(l), sin(yaw), cos(yaw)]
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # A tiny conv block improves stability over a bare 1×1 conv
        self.cls = self.make_bev_head(in_channels, 1)

        self.reg = self.make_bev_head(in_channels, 6)

    def forward(self, x: Tensor) -> dict:
        heatmap = torch.sigmoid(self.cls(x))
        reg = self.reg(x)
        return {"heatmap": heatmap, "reg": reg}

    @staticmethod
    def make_bev_head(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
        )
