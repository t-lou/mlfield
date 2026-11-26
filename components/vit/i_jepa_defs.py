from __future__ import annotations

from dataclasses import asdict, dataclass, field

from components.definitions.train_config import TrainConfig
from components.vit.vit_config import ViTConfig, ViTVariant

# Predictor defaults per backbone variant: (predictor_dim, predictor_depth, predictor_heads).
# I-JEPA uses a narrow predictor (384-wide, 12 layers for ViT-L/H); smaller backbones get a
# proportionally smaller one. "S" keeps the values this config always had.
PREDICTOR_DEFAULTS: dict[str, tuple[int, int, int]] = {
    "T": (128, 6, 2),
    "S": (192, 6, 6),
    "M": (256, 8, 4),
    "B": (384, 12, 6),
    "L": (384, 12, 6),
    "XL": (384, 12, 6),
}


# Fields that default to None and are then taken from the variant (see IJEPAConfig).
_VARIANT_FIELDS = (
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "predictor_dim",
    "predictor_depth",
    "predictor_heads",
)


@dataclass
class IJEPAConfig:
    """
    Config for I-JEPA with a configurable ViT backbone (default: ViT-S/16).

    Backbone: pick a ``variant`` (T, S, M, B, L, XL; see ``ViTVariant``). The backbone fields
    ``embed_dim``, ``depth``, ``num_heads``, ``mlp_ratio`` and the predictor fields
    ``predictor_dim``, ``predictor_depth``, ``predictor_heads`` default to ``None`` and are then
    taken from the variant. Set one explicitly to override it (existing configs that spell out
    all values keep working unchanged). After construction every field holds its concrete value.
    ``yaml_dict()`` (used by ``dump_yaml``) writes ``null`` for the fields that came from the
    variant, so a generated YAML can be switched to another variant by editing ``variant`` only.
    Programmatically use ``ijepa_config(variant, ...)`` rather than ``dataclasses.replace``.

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

    # ---- Backbone (context_encoder / target_encoder) ----
    variant: str = "S"  # one of ViTVariant: T, S, M, B, L, XL
    embed_dim: int | None = None  # None -> from variant
    depth: int | None = None  # None -> from variant
    num_heads: int | None = None  # None -> from variant
    mlp_ratio: float | None = None  # None -> from variant
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path_rate: float = 0.0  # paper uses light/no stochastic depth for ViT-S scale

    # ---- Predictor (narrow, shallow transformer) ----
    # I-JEPA paper: predictor width is much narrower than the encoder (384 for ViT-L/H).
    # None -> PREDICTOR_DEFAULTS[variant]
    predictor_dim: int | None = None
    predictor_depth: int | None = None
    predictor_heads: int | None = None
    predictor_mlp_ratio: float = 4.0

    # ---- Training data / batch settings ----
    data_dirs: tuple[str, ...] = ("./data/kaggle/",)
    train_config: TrainConfig = field(default_factory=TrainConfig)

    # ---- Multi-block masking strategy (paper defaults) ----
    # Context block: single large block, ~85-100% scale, square aspect ratio.
    context_scale_min: float = 0.85
    context_scale_max: float = 1.0
    context_aspect_ratio_min: float = 1.0
    context_aspect_ratio_max: float = 1.0
    # Target blocks: M=4 blocks per image, each a smaller, possibly
    # non-square rectangle sampled independently.
    num_target_blocks: int = 4
    target_scale_min: float = 0.15
    target_scale_max: float = 0.2
    aspect_ratio_min: float = 0.75
    aspect_ratio_max: float = 1.5

    def __post_init__(self) -> None:
        variant = str(self.variant).strip().upper()
        if variant not in ViTVariant.__members__:
            raise ValueError(f"Unknown variant {self.variant!r}, expected one of {list(ViTVariant.__members__)}")
        self.variant = variant
        v = ViTVariant[variant]
        self._explicit = frozenset(name for name in _VARIANT_FIELDS if getattr(self, name) is not None)

        # Resolve the backbone and predictor from the variant where not set explicitly.
        pred_dim, pred_depth, pred_heads = PREDICTOR_DEFAULTS[variant]
        defaults = {
            "embed_dim": v.embed_dim,
            "depth": v.depth,
            "num_heads": v.num_heads,
            "mlp_ratio": v.mlp_ratio,
            "predictor_dim": pred_dim,
            "predictor_depth": pred_depth,
            "predictor_heads": pred_heads,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, value)

        if self.image_size < self.patch_size:
            raise ValueError("image_size must be >= patch_size")
        if self.predictor_dim % self.predictor_heads != 0:
            raise ValueError(
                f"predictor_dim ({self.predictor_dim}) must be divisible by predictor_heads ({self.predictor_heads})"
            )
        self.vit_config()  # validates the backbone (head divisibility, drop rates, ...)

    def yaml_dict(self) -> dict:
        """Field dict for YAML dumps: values that came from the variant are written as None."""
        d = asdict(self)
        for name in _VARIANT_FIELDS:
            if name not in self._explicit:
                d[name] = None
        return d

    def vit_config(self) -> ViTConfig:
        """ViTConfig of the context / target encoder (no CLS token, I-JEPA uses patch tokens only)."""
        return ViTConfig.from_variant(
            ViTVariant[self.variant],
            patch_size=self.patch_size,
            base_res=self.image_size,
            add_cls_token=False,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path_rate=self.drop_path_rate,
            qkv_bias=self.qkv_bias,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )


def ijepa_config(variant: str = "S", patch_size: int = 16, **overrides) -> IJEPAConfig:
    """I-JEPA config for any ViT variant, e.g. ``ijepa_config("B", 16)`` or ``ijepa_config("XL", 14)``."""
    return IJEPAConfig(variant=variant, patch_size=patch_size, **overrides)


def vit_s16_config(**overrides) -> IJEPAConfig:
    """ViT-S/16 I-JEPA config (default)."""
    return IJEPAConfig(patch_size=16, **overrides)


def vit_s14_config(**overrides) -> IJEPAConfig:
    """ViT-S/14 I-JEPA config: same backbone width/depth, finer patches."""
    return IJEPAConfig(patch_size=14, **overrides)
