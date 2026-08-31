"""Flexible ViT Encoder supporting multiple scales and configurations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.vit.patch_embed import PatchEmbed
from components.vit.position_embedding import build_2d_sincos_position_embedding
from components.vit.vit_block import VitBlock
from components.vit.vit_config import ViTConfig, ViTVariant


class VitEncoder(nn.Module):
    """Vision Transformer Encoder with flexible configuration and multi-scale support.

    Supports:
    - Multiple scales: Tiny, Small, Medium, Large, Extra Large
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

    def __init__(self, config: ViTConfig = None, **kwargs):
        """Initialize ViT encoder.

        Args:
            config: ViTConfig instance for configuration
            **kwargs: Legacy parameter support for backward compatibility
                (base_res, patch_size, embed_dim, depth, etc.)
        """
        super().__init__()

        # Handle legacy parameters for backward compatibility
        if config is None:
            # Try to build config from legacy kwargs
            if kwargs:
                variant = ViTVariant.S  # default
                # Extract known variant-level params if provided
                config = ViTConfig.from_variant(variant, **kwargs)
            else:
                config = ViTConfig()

        self.config = config
        self.embed_dim = config.embed_dim
        self._patch_size = config.patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim,
        )

        # CLS token (optional)
        if config.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        else:
            self.cls_token = None

        # Positional embedding
        base_grid = config.base_res // config.patch_size
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_position_embedding(
                grid_size=base_grid,
                embed_dim=config.embed_dim,
                add_cls_token=config.add_cls_token,
            ),
        )

        # Transformer blocks with stochastic depth
        dpr = [config.drop_path_rate * i / max(config.depth - 1, 1) for i in range(config.depth)]
        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_drop=config.attn_drop,
                    proj_drop=config.proj_drop,
                    drop_path=dpr[i],
                    qkv_bias=config.qkv_bias,
                )
                for i in range(config.depth)
            ]
        )

        # Final normalization
        self.norm = nn.LayerNorm(config.embed_dim)

        self.add_cls_token = config.add_cls_token

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using standard ViT initialization."""
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def interpolate_pos_encoding(self, H, W):
        """Interpolate positional embeddings to arbitrary spatial dimensions.

        Allows the model to handle input resolutions different from base_res.
        Uses bicubic interpolation for smooth spatial interpolation.

        Args:
            H, W: Target height and width in patches

        Returns:
            Interpolated positional embeddings
        """
        N = self.pos_embed.shape[1] - (1 if self.add_cls_token else 0)
        old_size = int(N**0.5)

        if self.add_cls_token:
            cls_pos = self.pos_embed[:, 0:1]
            spatial_pos = self.pos_embed[:, 1:]
        else:
            cls_pos = None
            spatial_pos = self.pos_embed

        spatial_pos = spatial_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        spatial_pos = F.interpolate(spatial_pos, size=(H, W), mode="bicubic", align_corners=False)
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(1, H * W, -1)

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
            cls_ = self.cls_token.repeat(B, 1, 1)
            x = torch.cat((cls_, x), dim=1)
            x = x + pos
        else:
            x = x + pos

        return x

    def forward_full(self, imgs, patch_keep_mask=None, return_padding_mask=False):
        """Full transformer pass with optional token masking.

        Args:
            imgs: Input images (B, 3, H, W)
            patch_keep_mask: Optional mask for selective token processing
            return_padding_mask: Return the padding mask

        Returns:
            Features and optionally padding mask
        """
        x = self._tokenize(imgs)

        padding_mask = None
        if patch_keep_mask is not None:
            if patch_keep_mask.ndim != 2:
                raise ValueError("patch_keep_mask must have shape (B, num_patches)")

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
        return self.forward_full(imgs, patch_keep_mask)[:, 0]

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

    print("\n✓ All VitEncoder tests passed!")


if __name__ == "__main__":
    _smoke_test()
