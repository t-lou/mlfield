import torch.nn as nn
from torch import Tensor

import common.params as params


class FuTrFusionBlock(nn.Module):
    """
    Memory-safe FuTr-style fusion:
    - Camera tokens query BEV tokens
    - Produces a small fused camera representation
    - Broadcasts back into BEV space
    """

    def __init__(self, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        C = params.BEV_CHANNELS

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

        # Project fused camera tokens back into BEV modulation
        self.to_scale = nn.Linear(C, C)
        self.to_shift = nn.Linear(C, C)

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

        # Camera queries → BEV keys/values
        attn_out, _ = self.cross_attn(
            query=cam_tokens,  # (B, N_cam, C)
            key=bev_tokens,  # (B, HW, C)
            value=bev_tokens,
        )

        # Residual + norm
        cam_fused = self.norm1(cam_tokens + attn_out)

        # FFN + residual + norm
        cam_fused = self.norm2(cam_fused + self.ffn(cam_fused))

        # Aggregate camera tokens → a single global camera feature
        cam_global = cam_fused.mean(dim=1)  # (B, C)

        # Convert to scale/shift for BEV modulation
        scale = self.to_scale(cam_global).view(B, C, 1, 1)
        shift = self.to_shift(cam_global).view(B, C, 1, 1)

        # FiLM-style modulation
        fused_bev = bev * (1 + scale) + shift

        return fused_bev
