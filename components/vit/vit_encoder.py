"""Flexible ViT Encoder for DINO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import PatchEmbed
from .position_embedding import build_2d_sincos_position_embedding
from .vit_block import VitBlock


class VitEncoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.1,
        qkv_bias=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self._patch_size = patch_size

        # Patch embedding (your updated flexible version)
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        base_res = 224
        base_grid = base_res // patch_size
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_position_embedding(grid_size=base_grid, embed_dim=embed_dim, add_cls_token=True),
        )

        # Transformer blocks
        dpr = [drop_path_rate * i / (depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=dpr[i],
                    qkv_bias=qkv_bias,
                )
                for i in range(depth)
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

        # Add CLS token, note that it should be dropped in v2
        cls = self.cls_token.repeat(B, 1, 1)  # faster that expand
        x = torch.cat((cls, x), dim=1)

        # Compute new grid size
        H_patch = imgs.shape[2] // self._patch_size
        W_patch = imgs.shape[3] // self._patch_size

        # Add positional embeddings
        pos = self.interpolate_pos_encoding(H_patch, W_patch)
        x = x + pos

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]  # CLS token output


def _smoke_test():
    model = VitEncoder()
    imgs = torch.randn(2, 3, 224, 224)
    _ = model(imgs)


if __name__ == "__main__":
    _smoke_test()
