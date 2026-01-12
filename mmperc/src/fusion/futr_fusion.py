import torch.nn as nn


class FuTrFusionBlock(nn.Module):
    """
    Tiny FuTr-style fusion block.
    - Takes BEV feature map from backbone: (B, C, H, W)
    - Optional camera tokens: (B, N_cam, C)
    - Performs cross-attention: BEV queries attend to camera tokens
    - Returns fused BEV feature map: (B, C, H, W)
    """

    def __init__(self, bev_channels=128, num_heads=4, dropout=0.1):
        super().__init__()

        self.bev_channels = bev_channels

        # BEV → tokens
        self.bev_to_tokens = nn.Linear(bev_channels, bev_channels)

        # Cross-attention: BEV queries, camera keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bev_channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network (standard transformer block)
        self.ffn = nn.Sequential(
            nn.Linear(bev_channels, bev_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(bev_channels * 4, bev_channels),
        )

        self.norm1 = nn.LayerNorm(bev_channels)
        self.norm2 = nn.LayerNorm(bev_channels)

    def forward(self, bev, cam_tokens=None):
        """
        bev: (B, C, H, W)
        cam_tokens: (B, N_cam, C) or None
        """

        B, C, H, W = bev.shape

        # Flatten BEV → (B, HW, C)
        bev_tokens = bev.flatten(2).transpose(1, 2)  # (B, HW, C)
        bev_tokens = self.bev_to_tokens(bev_tokens)

        # If no camera tokens, skip fusion
        if cam_tokens is None:
            fused = bev_tokens
        else:
            # Cross-attention: BEV queries attend to camera tokens
            attn_out, _ = self.cross_attn(
                query=bev_tokens,  # (B, HW, C)
                key=cam_tokens,  # (B, N_cam, C)
                value=cam_tokens,
            )
            fused = bev_tokens + attn_out  # residual

        # FFN block
        fused = self.norm1(fused)
        ffn_out = self.ffn(fused)
        fused = fused + ffn_out
        fused = self.norm2(fused)

        # Reshape back to BEV map
        fused_bev = fused.transpose(1, 2).reshape(B, C, H, W)
        return fused_bev
