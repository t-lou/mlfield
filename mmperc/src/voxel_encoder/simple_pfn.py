import torch.nn as nn
import torch.nn.functional as F


class SimplePFN(nn.Module):
    """
    Input:  pillars (B, P, M, C_in)
    Output: pillar_features (B, P, C_out)
    """

    def __init__(self, in_channels=5, out_channels=64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, pillars, pillar_count=None):
        B, P, M, C = pillars.shape  # (B, P, M, C_in)

        x = self.linear(pillars)  # (B, P, M, C_out)
        x = x.view(B * P * M, -1)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(B, P, M, -1)  # (B, P, M, C_out)

        # max pool over points in each pillar
        x = x.max(dim=2).values  # (B, P, C_out)
        return x
