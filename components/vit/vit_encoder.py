"""Flexible ViT Encoder for DINO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.vit.patch_embed import PatchEmbed
from components.vit.position_embedding import build_2d_sincos_position_embedding
from components.vit.vit_block import VitBlock


class VitEncoder(nn.Module):
    def __init__(
        self,
        base_res=224,
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
        base_grid = base_res // patch_size
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_position_embedding(grid_size=base_grid, embed_dim=embed_dim, add_cls_token=True),
        )

        # Transformer blocks
        if depth <= 0:
            raise ValueError("depth must be >= 1")
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
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

    def _tokenize(self, imgs, add_cls_token=True):
        B = imgs.shape[0]

        # Patch embedding
        x = self.patch_embed(imgs)  # (B, HW, C)

        # Compute patch grid for positional interpolation
        H_patch = imgs.shape[2] // self._patch_size
        W_patch = imgs.shape[3] // self._patch_size

        # Add positional embeddings
        pos = self.interpolate_pos_encoding(H_patch, W_patch)

        if add_cls_token:
            cls_ = self.cls_token.repeat(B, 1, 1)  # faster than expand
            x = torch.cat((cls_, x), dim=1)
            x = x + pos
        else:
            # If not adding CLS token, just add positional embeddings to patch tokens
            x = x + pos[:, 1:, :]

        return x

    def forward_full(self, imgs, patch_keep_mask=None, add_cls_token=True, return_padding_mask=False):
        """
        Encode image tokens and optionally keep only selected patch tokens.

        Args:
            imgs: Input images, shape (B, 3, H, W)
            patch_keep_mask: Optional boolean mask of shape (B, num_patches),
                where True means keep this patch token.
            add_cls_token: If True, prepend CLS token before transformer blocks.
            return_padding_mask: If True, return the padding mask for selected tokens.

        Returns:
            Token features after transformer + norm, optionally paired with padding mask.
        """
        x = self._tokenize(imgs, add_cls_token=add_cls_token)

        padding_mask = None
        if patch_keep_mask is not None:
            if patch_keep_mask.ndim != 2:
                raise ValueError("patch_keep_mask must have shape (B, num_patches)")

            if add_cls_token:
                cls_keep = torch.ones(
                    (patch_keep_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=patch_keep_mask.device,
                )
                token_keep_mask = torch.cat([cls_keep, patch_keep_mask], dim=1)
            else:
                token_keep_mask = patch_keep_mask

            keep_counts = token_keep_mask.sum(dim=1)
            max_keep = int(keep_counts.max().item())
            keep_order = torch.argsort(token_keep_mask.to(torch.int64), dim=1, descending=True)
            keep_idx = keep_order[:, :max_keep]
            x = torch.gather(x, dim=1, index=keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]))

            selected = torch.gather(token_keep_mask, dim=1, index=keep_idx)
            padding_mask = ~selected
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, padding_mask=padding_mask)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.norm(x)
        if return_padding_mask:
            return x, padding_mask
        return x

    def forward(self, imgs, patch_keep_mask=None, add_cls_token=True):
        """
        Forward pass for the ViT encoder.

        Args:
            imgs: Input images, shape (B, 3, H, W)
            patch_keep_mask: Optional boolean mask of shape (B, num_patches),
                where True means keep this patch token.
            add_cls_token: If True, prepend CLS token before transformer blocks.

        Returns:
            CLS token output.
        """
        return self.forward_full(imgs, patch_keep_mask, add_cls_token)[:, 0]  # CLS token output, shape: (B, embed_dim)


def _smoke_test():
    model = VitEncoder()
    imgs = torch.randn(2, 3, 224, 224)
    _ = model(imgs)


if __name__ == "__main__":
    _smoke_test()
