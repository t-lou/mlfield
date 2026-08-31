"""Flexible ViT configuration for multiple scales and variants.

Provides a unified configuration system for Vision Transformers that scales
across different model sizes while maintaining consistent interfaces.
"""

from dataclasses import dataclass
from enum import Enum


class ViTVariant(Enum):
    """Vision Transformer scale variants with architecture parameters.

    Standard scaling approach:
    - embed_dim: Embedding dimension (hidden size)
    - depth: Number of transformer blocks
    - num_heads: Number of attention heads
    - mlp_ratio: MLP hidden dimension ratio

    Design principles:
    - Head dimension fixed at 64 for efficiency
    - MLP ratio increases with model size for capacity
    - Depth increases for better feature learning
    """

    # Tiny variant: for testing and fast inference
    T = (192, 6, 3, 3.0)

    # Small variant: efficient for mobile/edge
    S = (384, 12, 6, 4.0)

    # Medium variant: good balance of speed and accuracy
    M = (512, 12, 8, 4.0)

    # Large variant: high accuracy
    L = (768, 24, 12, 4.0)

    # Extra Large variant: maximum accuracy
    XL = (1024, 24, 16, 4.0)

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio):
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Infer head dimension from total dim and num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = embed_dim // num_heads


@dataclass(frozen=True)
class ViTConfig:
    """Unified ViT configuration supporting multiple scales and use cases.

    Controls:
    - Patch size and input resolution
    - Transformer architecture (variant)
    - Dropout and regularization
    - Normalization and activation choices

    Example:
        >>> config_small = ViTConfig(variant=ViTVariant.S)
        >>> config_large = ViTConfig(variant=ViTVariant.L, patch_size=8)
    """

    # Architecture variant
    variant: ViTVariant = ViTVariant.S

    # Patch and input configuration
    patch_size: int = 16
    base_res: int = 224  # Base resolution for positional embeddings

    # Token configuration
    add_cls_token: bool = True

    # Regularization
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path_rate: float = 0.1

    # Attention configuration
    qkv_bias: bool = False

    # Normalization
    norm_layer: str = "LayerNorm"  # Could extend to GroupNorm, RMSNorm etc

    # Activation
    act_layer: str = "GELU"  # Could extend to SiLU, Mish, etc

    @property
    def embed_dim(self) -> int:
        return self.variant.embed_dim

    @property
    def depth(self) -> int:
        return self.variant.depth

    @property
    def num_heads(self) -> int:
        return self.variant.num_heads

    @property
    def head_dim(self) -> int:
        return self.variant.head_dim

    @property
    def mlp_ratio(self) -> float:
        return self.variant.mlp_ratio

    @property
    def mlp_dim(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def num_patches(self) -> int:
        """Number of patches for base resolution."""
        patches_per_side = self.base_res // self.patch_size
        return patches_per_side**2

    @property
    def seq_len(self) -> int:
        """Sequence length including CLS token if present."""
        return self.num_patches + (1 if self.add_cls_token else 0)

    @property
    def num_params_estimate(self) -> int:
        """Rough estimate of model parameters (for comparison)."""
        # Patch embedding: 3 * patch_size^2 * embed_dim
        patch_emb = 3 * (self.patch_size**2) * self.embed_dim

        # Transformer blocks: per block ~12 * embed_dim^2 (attention + MLP)
        transformer = self.depth * 12 * (self.embed_dim**2)

        # CLS token and positional embeddings (relatively small)
        embeddings = self.embed_dim * self.seq_len

        # Approximate total
        return int(patch_emb + transformer + embeddings)

    @classmethod
    def from_variant(cls, variant: ViTVariant, **kwargs) -> "ViTConfig":
        """Create config from a predefined variant with optional overrides.

        Args:
            variant: ViTVariant enum value
            **kwargs: Optional parameter overrides (e.g., patch_size=8)

        Returns:
            ViTConfig instance
        """
        return cls(variant=variant, **kwargs)

    def __repr__(self) -> str:
        """Detailed config representation."""
        return (
            f"ViTConfig(\n"
            f"  variant={self.variant.name},\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  depth={self.depth},\n"
            f"  num_heads={self.num_heads},\n"
            f"  mlp_dim={self.mlp_dim},\n"
            f"  patch_size={self.patch_size},\n"
            f"  num_patches={self.num_patches},\n"
            f"  seq_len={self.seq_len},\n"
            f"  params_estimate={self.num_params_estimate:,}\n"
            f")"
        )
