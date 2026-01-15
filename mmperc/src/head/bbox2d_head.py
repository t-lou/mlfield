import torch
from torch import nn


class BBox2dHead(nn.Module):
    """
    Predicts:
      - heatmap: (B, 1, H, W)
      - box regression: (B, 6, H, W)
        [dx, dy, log(w), log(l), sin(yaw), cos(yaw)]
    """

    def __init__(self, in_channels):
        super().__init__()
        self.cls = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.reg = nn.Conv2d(in_channels, 6, kernel_size=1)

    def forward(self, x):
        heatmap = torch.sigmoid(self.cls(x))
        reg = self.reg(x)
        return {"heatmap": heatmap, "reg": reg}
