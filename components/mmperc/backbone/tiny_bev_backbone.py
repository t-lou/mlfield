import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from components.definitions.mmperc_params import MmpercParams


class TinyBEVBackbone(nn.Module):
    """
    Lightweight BEV backbone for memory‑constrained setups.
    Supports gradient checkpointing for training to reduce memory footprint.

    Args:
        in_channels:  Number of input feature channels (default: 64)
        mid_channels: Internal feature width (default: 64)
        out_channels: Output feature width after downsampling (default: 128)
        use_checkpoint: Whether to use gradient checkpointing for memory efficiency (default: False)

    Input:
        x: (B, in_channels, H, W)

    Output:
        (B, out_channels, H/2, W/2)
    """

    def __init__(
        self,
        params: MmpercParams,
        in_channels: int = 64,
        mid_channels: int = 64,
        out_channels: int = 128,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.use_checkpoint = use_checkpoint

        # Initial projection
        # Using GroupNorm for memory efficiency (no running stats during training)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )

        # Local feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )

        # Spatial downsampling + channel expansion
        self.down = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=params.bev_params.backbone_stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

        # Final refinement
        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to reduce memory during training
            x = checkpoint(self.stem, x, use_reentrant=False)
            x = checkpoint(self.block1, x, use_reentrant=False)
            x = checkpoint(self.block2, x, use_reentrant=False)
            x = checkpoint(self.down, x, use_reentrant=False)
            x = checkpoint(self.block3, x, use_reentrant=False)
        else:
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.down(x)
            x = self.block3(x)
        return x
