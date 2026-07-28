import torch
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
            One feature vector per pillar (max-pooled + mean-pooled over points)
    """

    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        # Combine max- and mean-pooled features, then project back down to
        # out_channels. Cheap (one more reduction over M, one more linear)
        # relative to the point-wise linear above.
        self.pool_project = nn.Linear(out_channels * 2, out_channels, bias=False)
        # GroupNorm for memory efficiency - normalize over groups instead of batch
        # Use 1D version with proper reshape
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(self, pillars: Tensor) -> Tensor:
        B, P, M, C = pillars.shape

        # Linear projection applied per point
        x = self.linear(pillars)  # (B, P, M, C_out)

        # Pool over points within each pillar: max captures the most salient
        # point, mean captures the overall pillar statistics — cheap to
        # compute together and more informative than max alone.
        mask = pillars.abs().sum(dim=-1, keepdim=True) > 0  # (B, P, M, 1)
        mask_expanded = mask.expand_as(x)

        x_max = x.masked_fill(~mask_expanded, float("-inf")).max(dim=2).values  # (B, P, C_out)
        x_max = torch.nan_to_num(x_max, neginf=0.0)  # handle fully-empty pillars

        point_count = mask.sum(dim=2).clamp(min=1)  # (B, P, 1)
        x_mean = x.masked_fill(~mask_expanded, 0.0).sum(dim=2) / point_count  # (B, P, C_out)

        x = self.pool_project(torch.cat([x_max, x_mean], dim=-1))  # (B, P, C_out)

        # Reshape for GroupNorm: (B*P, C_out)
        x = x.reshape(B * P, -1)
        x = self.norm(x)
        x = F.relu(x)
        x = x.reshape(B, P, -1)

        return x


def _smoke_test():
    """
    Smoke test for SimplePFN.
    """
    B, P, M, C_in = 2, 4, 8, 9
    pillars = torch.rand(B, P, M, C_in)
    pfn = SimplePFN(in_channels=C_in, out_channels=64)
    output = pfn(pillars)
    assert output.shape == (B, P, 64), f"Expected shape (B, P, 64), got {output.shape}"
    print("SimplePFN smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
