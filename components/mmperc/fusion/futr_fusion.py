import torch.nn as nn
from torch import Tensor

from components.definitions.mmperc_params import MmpercParams


class FuTrFusionBlock(nn.Module):
    """
    Memory-safe FuTr-style fusion:
    - Camera tokens query BEV tokens
    - Produces a small fused camera representation
    - Broadcasts back into BEV space
    """

    def __init__(self, params: MmpercParams, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        C = params.bev_params.bev_channels

        # Project BEV tokens (C) → (C)
        self.bev_proj = nn.Linear(C, C)

        # Project camera tokens (C) → (C)
        self.cam_proj = nn.Linear(C, C)

        # Cross-attention: camera queries → BEV keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN on camera tokens
        self.ffn = nn.Sequential(
            nn.Linear(C, C * 4),
            nn.ReLU(inplace=True),
            nn.Linear(C * 4, C),
        )

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)

        # Fused scale+shift projection: one matmul instead of two
        self.to_film = nn.Linear(C, C * 2)

    def forward(self, bev: Tensor, camera: Tensor) -> Tensor:
        """
        bev:    (B, C, H, W)
        camera: (B, N_cam, C)
        """
        B, C, H, W = bev.shape

        # Flatten BEV → tokens
        bev_tokens = bev.flatten(2).transpose(1, 2)  # (B, HW, C)
        bev_tokens = self.bev_proj(bev_tokens)

        # Project camera tokens
        cam_tokens = self.cam_proj(camera)  # (B, N_cam, C)

        # need_weights=False lets PyTorch dispatch to the fused
        # scaled_dot_product_attention kernel (flash / mem-efficient attn)
        # instead of materializing the full (B, heads, N_cam, HW) weight
        # matrix — this is the main memory/runtime cost of this block.
        attn_out, _ = self.cross_attn(
            query=cam_tokens,
            key=bev_tokens,
            value=bev_tokens,
            need_weights=False,
        )

        # Residual + norm
        cam_fused = self.norm1(cam_tokens + attn_out)

        # FFN + residual + norm
        cam_fused = self.norm2(cam_fused + self.ffn(cam_fused))

        # Aggregate camera tokens → a single global camera feature
        cam_global = cam_fused.mean(dim=1)  # (B, C)

        # Convert to scale/shift for BEV modulation
        scale, shift = self.to_film(cam_global).chunk(2, dim=-1)
        scale = scale.view(B, C, 1, 1)
        shift = shift.view(B, C, 1, 1)

        # FiLM-style modulation
        fused_bev = bev * (1 + scale) + shift

        return fused_bev
