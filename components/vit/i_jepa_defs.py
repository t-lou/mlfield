from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IJEPAConfig:
    """
    Config for I-JEPA with a ViT-S/16 backbone.

    Backbone (ViT-S) follows the standard "small" ViT recipe used across
    DeiT/MAE/I-JEPA-style papers: embed_dim=384, depth=12, heads=6, mlp_ratio=4.

    Masking / predictor hyperparameters follow Assran et al., 2023
    ("Self-Supervised Learning from Images with a Joint-Embedding Predictive
    Architecture"), Table 6 / Appendix settings for the multi-block masking
    strategy and the lightweight predictor.

    To switch to ViT-S/14, just change patch_size to 14 (and, if you want to
    keep the standard 224 pretraining resolution, image_size stays 224 -
    grid_size becomes 16x16 = 256 patches instead of 14x14 = 196).
    """

    # ---- Image / patchification ----
    image_size: int = 256
    patch_size: int = 16  # -> ViT-S/16. Set to 14 for ViT-S/14.

    # ---- ViT-S backbone (context_encoder / target_encoder) ----
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path_rate: float = 0.0  # paper uses light/no stochastic depth for ViT-S scale

    # ---- Predictor (narrow, shallow transformer) ----
    # I-JEPA paper: predictor width is much narrower than the encoder and
    # roughly half the depth. For ViT-S encoder (384-wide, 12 layers), a
    # 192-wide / 6-layer predictor mirrors the paper's ratio for larger
    # backbones (e.g. ViT-L/H use predictor_dim=384, ~half of encoder width).
    predictor_dim: int = 192
    predictor_depth: int = 6
    predictor_heads: int = 6
    predictor_mlp_ratio: float = 4.0

    # ---- Multi-block masking strategy (paper defaults) ----
    # Context block: single large block, ~85-100% scale, square aspect ratio.
    context_scale_min: float = 0.85
    context_scale_max: float = 1.0
    # Target blocks: M=4 blocks per image, each a smaller, possibly
    # non-square rectangle sampled independently.
    num_target_blocks: int = 4
    target_scale_min: float = 0.15
    target_scale_max: float = 0.2
    aspect_ratio_min: float = 0.75
    aspect_ratio_max: float = 1.5


def vit_s16_config(**overrides) -> IJEPAConfig:
    """ViT-S/16 I-JEPA config (default)."""
    return IJEPAConfig(patch_size=16, **overrides)


def vit_s14_config(**overrides) -> IJEPAConfig:
    """ViT-S/14 I-JEPA config: same backbone width/depth, finer patches."""
    return IJEPAConfig(patch_size=14, **overrides)
