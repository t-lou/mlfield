"""Flexible ViT configuration for multiple scales and variants.

Provides a unified configuration system for Vision Transformers that scales
across different model sizes while maintaining consistent interfaces.
"""

from dataclasses import dataclass
from enum import Enum

# Names accepted for ``ViTConfig.norm_layer`` / ``ViTConfig.act_layer``.
# The matching factories live in ``components.vit.vit_block``.
SUPPORTED_NORM_LAYERS = ("LayerNorm", "RMSNorm")
SUPPORTED_ACT_LAYERS = ("GELU", "SiLU", "ReLU")


class ViTVariant(Enum):
    """Vision Transformer scale variants with architecture parameters.

    Standard scaling approach (matches the published ViT / DeiT sizes):
    - embed_dim: Embedding dimension (hidden size)
    - depth: Number of transformer blocks
    - num_heads: Number of attention heads
    - mlp_ratio: MLP hidden dimension ratio

    Variants:
    - T  = ViT-Tiny   (DeiT-Ti):  192 / 12 / 3,  ~5.7M params
    - S  = ViT-Small  (DeiT-S):   384 / 12 / 6,  ~22M params
    - M  = ViT-Medium (timm):     512 / 12 / 8,  ~39M params
    - B  = ViT-Base:              768 / 12 / 12, ~86M params
    - L  = ViT-Large:            1024 / 24 / 16, ~304M params
    - XL = ViT-Huge:             1280 / 32 / 16, ~632M params

    Design principles:
    - Head dimension is 64 (XL: 80, as in the published ViT-Huge)
    - MLP ratio is 4 for all sizes
    - Width and depth increase together for better feature learning
    """

    # Tiny variant: for testing and fast inference
    T = (192, 12, 3, 4.0)

    # Small variant: efficient for mobile/edge
    S = (384, 12, 6, 4.0)

    # Medium variant: good balance of speed and accuracy
    M = (512, 12, 8, 4.0)

    # Base variant: standard ViT-B
    B = (768, 12, 12, 4.0)

    # Large variant: high accuracy
    L = (1024, 24, 16, 4.0)

    # Extra Large variant: maximum accuracy (ViT-Huge)
    XL = (1280, 32, 16, 4.0)

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
    - Transformer architecture (variant, optionally overridden per field)
    - Dropout and regularization
    - Normalization and activation choices

    Example:
        >>> config_small = ViTConfig(variant=ViTVariant.S)
        >>> config_large = ViTConfig(variant=ViTVariant.L, patch_size=8)
        >>> config_custom = ViTConfig.from_variant(ViTVariant.S, depth=8)
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
    # NOTE: the reference DINO / DINOv2 / MAE / timm ViTs use qkv_bias=True.
    # The default stays False for backward compatibility; set it per recipe.
    qkv_bias: bool = False

    # Normalization
    norm_layer: str = "LayerNorm"  # One of SUPPORTED_NORM_LAYERS

    # Activation
    act_layer: str = "GELU"  # One of SUPPORTED_ACT_LAYERS

    # Optional per-field overrides of the variant's architecture (None = use variant).
    # Prefer ``ViTConfig.from_variant(variant, depth=..., embed_dim=...)`` to set them.
    embed_dim_override: int | None = None
    depth_override: int | None = None
    num_heads_override: int | None = None
    mlp_ratio_override: float | None = None

    # Input channels (3 = RGB). Last field, so existing positional construction keeps working.
    in_chans: int = 3

    def __post_init__(self) -> None:
        if self.patch_size <= 0 or self.base_res <= 0 or self.in_chans <= 0:
            raise ValueError("patch_size, base_res and in_chans must be positive")
        if not 0.0 <= self.attn_drop <= 1.0 or not 0.0 <= self.proj_drop <= 1.0:
            raise ValueError("attn_drop and proj_drop must be in [0, 1]")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if self.norm_layer not in SUPPORTED_NORM_LAYERS:
            raise ValueError(f"norm_layer must be one of {SUPPORTED_NORM_LAYERS}, got {self.norm_layer!r}")
        if self.act_layer not in SUPPORTED_ACT_LAYERS:
            raise ValueError(f"act_layer must be one of {SUPPORTED_ACT_LAYERS}, got {self.act_layer!r}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")

    @property
    def embed_dim(self) -> int:
        if self.embed_dim_override is not None:
            return self.embed_dim_override
        return self.variant.embed_dim

    @property
    def depth(self) -> int:
        if self.depth_override is not None:
            return self.depth_override
        return self.variant.depth

    @property
    def num_heads(self) -> int:
        if self.num_heads_override is not None:
            return self.num_heads_override
        return self.variant.num_heads

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_ratio(self) -> float:
        if self.mlp_ratio_override is not None:
            return self.mlp_ratio_override
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
        """Number of trainable parameters of ``VitEncoder``.

        The sin-cos positional embedding is a fixed buffer and not counted.
        """
        dim, hidden = self.embed_dim, self.mlp_dim
        norm = dim * (2 if self.norm_layer == "LayerNorm" else 1)

        # Patch embedding: conv weight + bias
        patch_emb = self.in_chans * (self.patch_size**2) * dim + dim

        # Per block: fused qkv (+bias), proj, MLP (fc1 + fc2, with biases), two norms
        qkv = 3 * dim * dim + (3 * dim if self.qkv_bias else 0)
        proj = dim * dim + dim
        mlp = dim * hidden + hidden + hidden * dim + dim
        block = qkv + proj + mlp + 2 * norm

        # CLS token and final norm
        cls_token = dim if self.add_cls_token else 0

        return int(patch_emb + self.depth * block + cls_token + norm)

    @classmethod
    def from_variant(cls, variant: ViTVariant, **kwargs) -> "ViTConfig":
        """Create config from a predefined variant with optional overrides.

        Args:
            variant: ViTVariant enum value
            **kwargs: Optional parameter overrides (e.g., patch_size=8). The architecture
                values embed_dim, depth, num_heads and mlp_ratio may be overridden as well.

        Returns:
            ViTConfig instance
        """
        for name in ("embed_dim", "depth", "num_heads", "mlp_ratio"):
            if name in kwargs:
                kwargs[f"{name}_override"] = kwargs.pop(name)
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
