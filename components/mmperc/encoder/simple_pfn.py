import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimplePFN(nn.Module):
    """
    A minimal Pillar Feature Network (PFN) block.

    Args:
        in_channels:  Number of input point features per pillar
        out_channels: Output feature dimension per pillar (default: 64)

    Input:
        pillars: (B, P, M, C_in)
            B = batch size
            P = number of pillars
            M = max points per pillar
            C_in = input feature dimension

    Output:
        (B, P, C_out)
            One feature vector per pillar (max-pooled over points)
    """

    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        # GroupNorm for memory efficiency - normalize over groups instead of batch
        # Use 1D version with proper reshape
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(self, pillars: Tensor, pillar_count: Tensor | None = None) -> Tensor:
        B, P, M, C = pillars.shape

        # Linear projection applied per point
        x = self.linear(pillars)  # (B, P, M, C_out)

        # Max-pool over points within each pillar first
        x = x.max(dim=2).values  # (B, P, C_out)

        # Reshape for GroupNorm: (B*P, C_out)
        x = x.reshape(B * P, -1)
        x = self.norm(x)
        x = F.relu(x)
        x = x.reshape(B, P, -1)

        return x
