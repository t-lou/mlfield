import common.params as params
import torch.nn as nn
from torch import Tensor


class FuTrFusionBlock(nn.Module):
    """
    FuTr-style BEV-camera fusion block.

    Inputs:
        bev:        (B, C, H, W) BEV feature map from lidar backbone
        camera: (B, N_cam, C) camera tokens from camera encoder

    Output:
        fused_bev:  (B, C, H, W) fused BEV feature map
    """

    def __init__(
        self,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        bev_channels = params.BEV_CHANNELS

        # Linear projection for BEV tokens (keeps dimension = bev_channels)
        self.bev_to_tokens = nn.Linear(bev_channels, bev_channels)

        # Cross-attention: BEV queries attend to camera tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bev_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (B, L, C) instead of (L, B, C)
        )

        # Standard transformer feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(bev_channels, bev_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(bev_channels * 4, bev_channels),
        )

        # LayerNorm for residual connections
        self.norm1 = nn.LayerNorm(bev_channels)
        self.norm2 = nn.LayerNorm(bev_channels)

    def forward(self, bev: Tensor, camera: Tensor) -> Tensor:
        """
        bev:        (B, C, H, W)
        camera: (B, N_cam, C)

        Returns:
            fused_bev: (B, C, H, W)
        """
        B, C, H, W = bev.shape

        # ------------------------------------------------------------
        # 1. Convert BEV feature map → sequence of tokens
        #    (B, C, H, W) → (B, HW, C)
        # ------------------------------------------------------------
        bev_tokens: Tensor = bev.flatten(2).transpose(1, 2)

        # Optional linear projection (keeps dimension = C)
        bev_tokens = self.bev_to_tokens(bev_tokens)

        # ------------------------------------------------------------
        # 2. Cross-attention: BEV queries attend to camera tokens
        # ------------------------------------------------------------
        attn_out, _ = self.cross_attn(
            query=bev_tokens,  # (B, HW, C)
            key=camera,  # (B, N_cam, C)
            value=camera,  # (B, N_cam, C)
        )

        # Residual connection + norm
        fused: Tensor = bev_tokens + attn_out
        fused = self.norm1(fused)

        # ------------------------------------------------------------
        # 3. Feed-forward network + residual + norm
        # ------------------------------------------------------------
        ffn_out: Tensor = self.ffn(fused)
        fused = fused + ffn_out
        fused = self.norm2(fused)

        # ------------------------------------------------------------
        # 4. Convert tokens back to BEV map
        #    (B, HW, C) → (B, C, H, W)
        # ------------------------------------------------------------
        fused_bev: Tensor = fused.transpose(1, 2).reshape(B, C, H, W)
        return fused_bev
