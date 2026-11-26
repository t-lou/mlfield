"""Flexible ViT Encoder supporting multiple scales and configurations."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.vit.patch_embed import PatchEmbed
from components.vit.position_embedding import build_2d_sincos_position_embedding
from components.vit.vit_block import VitBlock, get_norm_layer
from components.vit.vit_config import ViTConfig, ViTVariant


class VitEncoder(nn.Module):
    """Vision Transformer Encoder with flexible configuration and multi-scale support.

    Supports:
    - Multiple scales: Tiny, Small, Medium, Base, Large, Extra Large
    - Configurable patch sizes and input resolutions
    - Extensible normalization and activation layers
    - Optional CLS token and positional embeddings

    Example:
        >>> # Small variant (default)
        >>> encoder = VitEncoder()
        >>>
        >>> # Large variant with custom patch size
        >>> config = ViTConfig.from_variant(ViTVariant.L, patch_size=8)
        >>> encoder = VitEncoder(config=config)
    """

    def __init__(
        self,
        config: ViTConfig | None = None,
        *,
        base_res: int | None = None,
        patch_size: int | None = None,
        embed_dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float | None = None,
        attn_drop: float | None = None,
        proj_drop: float | None = None,
        drop_path_rate: float | None = None,
        qkv_bias: bool | None = None,
        add_cls_token: bool | None = None,
    ):
        """Initialize ViT encoder.

        Args:
            config: ViTConfig instance for configuration
            The explicit keyword parameters are supported for backward
            compatibility. Prefer passing a ViTConfig for new code.
        """
        super().__init__()

        # Keep accepting the historical keyword API: without a config the keywords are
        # applied on top of the Small variant. They are stored in the config, so
        # ``self.config`` always describes the model that was actually built.
        legacy_kwargs = {
            name: value
            for name, value in {
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "mlp_ratio": mlp_ratio,
                "base_res": base_res,
                "patch_size": patch_size,
                "attn_drop": attn_drop,
                "proj_drop": proj_drop,
                "drop_path_rate": drop_path_rate,
                "qkv_bias": qkv_bias,
                "add_cls_token": add_cls_token,
            }.items()
            if value is not None
        }
        if config is None:
            config = ViTConfig.from_variant(ViTVariant.S, **legacy_kwargs)
        elif legacy_kwargs:
            raise TypeError("Cannot combine config with legacy keyword parameters")

        self.config = config
        self.embed_dim = config.embed_dim
        self._depth = config.depth
        self._num_heads = config.num_heads
        self._mlp_ratio = config.mlp_ratio
        self._patch_size = config.patch_size
        self._add_cls_token = config.add_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
        )

        # CLS token (optional)
        if self._add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.cls_token = None

        # Positional embedding
        base_grid = config.base_res // self._patch_size
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_position_embedding(
                grid_size=base_grid,
                embed_dim=self.embed_dim,
                add_cls_token=self._add_cls_token,
            ),
        )

        # Transformer blocks with stochastic depth
        dpr = [config.drop_path_rate * i / max(self._depth - 1, 1) for i in range(self._depth)]
        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=self.embed_dim,
                    num_heads=self._num_heads,
                    mlp_ratio=self._mlp_ratio,
                    attn_drop=config.attn_drop,
                    proj_drop=config.proj_drop,
                    drop_path=dpr[i],
                    qkv_bias=config.qkv_bias,
                    norm_layer=config.norm_layer,
                    act_layer=config.act_layer,
                )
                for i in range(self._depth)
            ]
        )

        # Final normalization
        self.norm = get_norm_layer(config.norm_layer)(self.embed_dim)

        self.add_cls_token = self._add_cls_token

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using standard ViT initialization.

        Linear layers use truncated normal (std=0.02) with zero bias, as in DINO / DINOv2 /
        I-JEPA; norm layers keep their default (weight 1, bias 0).
        """
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def interpolate_pos_encoding(self, H, W):
        """Interpolate positional embeddings to arbitrary spatial dimensions.

        Allows the model to handle input resolutions different from base_res.
        Uses bicubic interpolation for smooth spatial interpolation. At the base
        resolution the stored embedding is returned as is.

        Args:
            H, W: Target height and width in patches

        Returns:
            Interpolated positional embeddings
        """
        N = self.pos_embed.shape[1] - (1 if self.add_cls_token else 0)
        old_size = math.isqrt(N)

        if old_size == H and old_size == W:
            return self.pos_embed

        if self.add_cls_token:
            cls_pos = self.pos_embed[:, 0:1]
            spatial_pos = self.pos_embed[:, 1:]
        else:
            cls_pos = None
            spatial_pos = self.pos_embed

        # Interpolate in fp32 (bicubic is not available for every half-precision backend)
        spatial_pos = spatial_pos.float().reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        spatial_pos = F.interpolate(spatial_pos, size=(H, W), mode="bicubic", align_corners=False)
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(1, H * W, -1).to(self.pos_embed.dtype)

        if cls_pos is not None:
            return torch.cat([cls_pos, spatial_pos], dim=1)
        return spatial_pos

    def _tokenize(self, imgs):
        """Convert image to patch tokens with positional embeddings.

        Args:
            imgs: Input images (B, 3, H, W)

        Returns:
            Tokenized embeddings with positional information
        """
        B = imgs.shape[0]

        # Patch embedding
        x = self.patch_embed(imgs)  # (B, HW, C)

        # Compute patch grid for positional interpolation
        H_patch = imgs.shape[2] // self._patch_size
        W_patch = imgs.shape[3] // self._patch_size

        # Add positional embeddings with interpolation support
        pos = self.interpolate_pos_encoding(H_patch, W_patch)

        if self.add_cls_token:
            cls_ = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_, x), dim=1)

        return x + pos

    def _forward_tokens(self, imgs, patch_keep_mask=None):
        """Tokenize, optionally drop tokens, and run the transformer blocks (before the final norm).

        Returns:
            (tokens, padding_mask, has_padding). ``padding_mask`` is None without a
            ``patch_keep_mask``; ``has_padding`` is False if every sample kept the same
            number of tokens, in which case the blocks run without an attention mask
            (which allows the fast flash-attention kernels).
        """
        x = self._tokenize(imgs)

        padding_mask = None
        has_padding = False
        if patch_keep_mask is not None:
            if patch_keep_mask.ndim != 2:
                raise ValueError("patch_keep_mask must have shape (B, num_patches)")
            num_cls = 1 if self.add_cls_token else 0
            if patch_keep_mask.shape[1] != x.shape[1] - num_cls:
                raise ValueError(
                    f"patch_keep_mask has {patch_keep_mask.shape[1]} patches per sample, "
                    f"but the input produces {x.shape[1] - num_cls}"
                )
            patch_keep_mask = patch_keep_mask.to(torch.bool)

            if self.add_cls_token:
                cls_keep = torch.ones(
                    (patch_keep_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=patch_keep_mask.device,
                )
                token_keep_mask = torch.cat([cls_keep, patch_keep_mask], dim=1)
            else:
                token_keep_mask = patch_keep_mask

            keep_counts = token_keep_mask.sum(dim=1)
            # one host sync for both statistics
            min_keep, max_keep = torch.stack([keep_counts.min(), keep_counts.max()]).tolist()

            if min_keep == token_keep_mask.shape[1]:
                # nothing dropped: no gather needed
                padding_mask = torch.zeros_like(token_keep_mask)
            else:
                # Stable sort: kept tokens stay in their original order, so the CLS token is
                # always at index 0 and downstream code can rely on the token order.
                keep_order = torch.sort(token_keep_mask.to(torch.uint8), dim=1, descending=True, stable=True)[1]
                keep_idx = keep_order[:, :max_keep]
                x = torch.gather(x, dim=1, index=keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]))

                selected = torch.gather(token_keep_mask, dim=1, index=keep_idx)
                padding_mask = ~selected
                has_padding = min_keep != max_keep
                if has_padding:
                    x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Transformer blocks
        block_mask = padding_mask if has_padding else None
        for blk in self.blocks:
            x = blk(x, padding_mask=block_mask)

        return x, padding_mask, has_padding

    def forward_full(self, imgs, patch_keep_mask=None, return_padding_mask=False):
        """Full transformer pass with optional token masking.

        Args:
            imgs: Input images (B, 3, H, W)
            patch_keep_mask: Optional mask for selective token processing
                (B, num_patches), True/1 = keep. Kept tokens are returned in their original
                order (CLS first), padded to the longest sample; padded outputs are zero.
            return_padding_mask: Return the padding mask

        Returns:
            Features and optionally padding mask
        """
        x, padding_mask, has_padding = self._forward_tokens(imgs, patch_keep_mask)

        x = self.norm(x)
        if has_padding:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if return_padding_mask:
            return x, padding_mask
        return x

    def forward(self, imgs, patch_keep_mask=None):
        """Forward pass returning CLS token output.

        Args:
            imgs: Input images (B, 3, H, W)
            patch_keep_mask: Optional mask for selective tokens

        Returns:
            CLS token embeddings (B, embed_dim)
        """
        if not self.add_cls_token:
            raise ValueError("forward() requires CLS token. Use forward_full() or set add_cls_token=True")
        x, _, _ = self._forward_tokens(imgs, patch_keep_mask)
        # the norm is per-token, so normalizing only the CLS token gives identical results
        return self.norm(x[:, 0])

    def forward_detr(self, imgs):
        """Forward pass for DETR-style detection (returns spatial feature map).

        Args:
            imgs: Input images (B, 3, H, W)

        Returns:
            Spatial feature map (B, C, H, W)
        """
        if self.add_cls_token:
            raise ValueError("forward_detr() requires no CLS token. Create encoder with add_cls_token=False")

        x = self.forward_full(imgs)  # (B, HW, C)
        B, N, C = x.shape
        H_patch = imgs.shape[2] // self._patch_size
        W_patch = imgs.shape[3] // self._patch_size
        x = x.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x

    @torch.no_grad()
    def get_last_selfattention(self, imgs):
        """Attention probabilities of the last block, as used for DINO-style attention maps.

        Call in eval mode.

        Args:
            imgs: Input images (B, 3, H, W)

        Returns:
            Attention probabilities (B, num_heads, N, N); with a CLS token, row 0 is CLS-to-all.
        """
        x = self._tokenize(imgs)
        for blk in self.blocks[:-1]:
            x = blk(x)
        return self.blocks[-1].get_attention(x)

    @classmethod
    def from_variant(cls, variant: ViTVariant, add_cls_token: bool = True, **kwargs) -> "VitEncoder":
        """Create encoder from a predefined variant.

        Args:
            variant: ViTVariant enum value
            add_cls_token: Whether to add CLS token
            **kwargs: Optional parameter overrides

        Returns:
            VitEncoder instance
        """
        config = ViTConfig.from_variant(variant, add_cls_token=add_cls_token, **kwargs)
        return cls(config=config)


def _smoke_test():
    """Smoke test for VitEncoder with multiple variants."""

    print("Testing VitEncoder variants...")

    # Test each variant
    for variant in [ViTVariant.T, ViTVariant.S, ViTVariant.M]:
        print(f"\n  Testing {variant.name} variant...")
        encoder = VitEncoder.from_variant(variant)
        imgs = torch.randn(2, 3, 224, 224)

        # Test CLS output
        cls_output = encoder(imgs)
        assert cls_output.shape == (2, encoder.embed_dim), f"CLS output shape mismatch for {variant.name}"

        # Test full output
        full_output = encoder.forward_full(imgs)
        assert full_output.shape[0] == 2, f"Batch size mismatch for {variant.name}"

        print(f"    ✓ {variant.name}: embed_dim={encoder.embed_dim}, depth={encoder.config.depth}")

    # Token dropping with different keep counts per sample (JEPA / MAE style)
    encoder = VitEncoder.from_variant(ViTVariant.T).eval()
    imgs = torch.randn(2, 3, 224, 224)
    keep = torch.ones(2, 196, dtype=torch.bool)
    keep[1, 100:] = False
    full, padding_mask = encoder.forward_full(imgs, keep, return_padding_mask=True)
    assert full.shape == (2, 197, encoder.embed_dim) and padding_mask.sum() == 96
    assert torch.allclose(encoder(imgs, keep), full[:, 0], atol=1e-5), "CLS must stay at index 0"

    # Spatial feature map without CLS token (DETR style) at a non-base resolution
    detr_encoder = VitEncoder.from_variant(ViTVariant.T, add_cls_token=False)
    assert detr_encoder.forward_detr(torch.randn(1, 3, 128, 192)).shape == (1, detr_encoder.embed_dim, 8, 12)

    print("\n✓ All VitEncoder tests passed!")


if __name__ == "__main__":
    _smoke_test()
