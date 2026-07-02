"""Flexible ViT Encoder for DINO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import PatchEmbed
from .position_embedding import build_2d_sincos_position_embedding


class VitEncoder(nn.Module):
    def __init__(self, patch_size=16, embed_dim=384, depth=12, num_heads=6):
        super().__init__()

        # Patch embedding (your updated flexible version)
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (base resolution = 224)
        base_grid = 224 // patch_size
        self.pos_embed = build_2d_sincos_position_embedding(
            grid_size=base_grid, embed_dim=embed_dim, add_cls_token=True
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(self, H, W):
        N = self.pos_embed.shape[1] - 1  # exclude CLS
        old_size = int(N**0.5)

        cls_pos = self.pos_embed[:, 0:1]
        spatial_pos = self.pos_embed[:, 1:]

        spatial_pos = spatial_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)

        spatial_pos = F.interpolate(spatial_pos, size=(H, W), mode="bicubic", align_corners=False)

        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(1, H * W, -1)

        return torch.cat([cls_pos, spatial_pos], dim=1)

    def forward(self, imgs):
        B = imgs.shape[0]

        # Patch embedding
        x = self.patch_embed(imgs)  # (B, HW, C)

        # Compute new grid size
        H = imgs.shape[2] // self.patch_embed.proj.kernel_size[0]
        W = imgs.shape[3] // self.patch_embed.proj.kernel_size[0]

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        # Add positional embeddings
        pos = self.interpolate_pos_encoding(H, W)
        x = x + pos

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]  # CLS token output
