from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Final

from components.definitions.train_config import TrainConfig
from components.vit.vit_config import ViTConfig, ViTVariant

# Decoder defaults per encoder variant: (decoder_dim, decoder_depth, decoder_heads).
# The MAE paper uses one lightweight decoder (512 wide, 8 blocks, 16 heads) for ViT-B/L/H; "B", "L"
# and "XL" keep exactly that, and "B" is what this config always defaulted to. The smaller encoders
# get a proportionally smaller decoder (these three are our own choice, the paper has no such sizes).
DECODER_DEFAULTS: dict[str, tuple[int, int, int]] = {
    "T": (128, 4, 4),
    "S": (256, 4, 8),
    "M": (384, 6, 8),
    "B": (512, 8, 16),
    "L": (512, 8, 16),
    "XL": (512, 8, 16),
}

LR_SCHEDULES = ("constant", "cosine")

# Fields that default to None and are then taken from the variant (see MAEConfig).
_VARIANT_FIELDS = (
    "encoder_dim",
    "encoder_depth",
    "encoder_heads",
    "decoder_dim",
    "decoder_depth",
    "decoder_heads",
)


@dataclass
class MAEConfig:
    """
    Configuration for different MAE model variants.

    Model size: pick a ``variant`` (T, S, M, B, L, XL; see ``ViTVariant``, default B = the old default).
    The fields ``encoder_dim``, ``encoder_depth``, ``encoder_heads`` and ``decoder_dim``,
    ``decoder_depth``, ``decoder_heads`` default to ``None`` and are then taken from the variant
    (``DECODER_DEFAULTS`` for the decoder). Set one explicitly to override it; existing configs that
    spell out all values keep working unchanged. After construction every field holds its concrete
    value. ``yaml_dict()`` (used by ``dump_yaml``) writes ``null`` for the fields that came from the
    variant, so a generated YAML can be switched to another variant by editing ``variant`` only.
    Programmatically use ``mae_config(variant, ...)`` rather than ``dataclasses.replace``.

    Attributes:
        in_chans: Number of input channels (3 for RGB images)
        image_size: Input image resolution (pixels)
        patch_size: Size of image patches (pixels)
        encoder_dim: Embedding dimension for transformer encoder (None -> variant)
        encoder_depth: Number of transformer blocks in encoder (None -> variant)
        encoder_heads: Number of attention heads in encoder (None -> variant)
        decoder_dim: Embedding dimension for transformer decoder (None -> DECODER_DEFAULTS[variant])
        decoder_depth: Number of transformer blocks in decoder (None -> DECODER_DEFAULTS[variant])
        decoder_heads: Number of attention heads in decoder (None -> DECODER_DEFAULTS[variant])
        mask_ratio: Fraction of patches to mask during training (0-1)
        learning_rate: Peak learning rate for optimization (the absolute value, not a base lr that
            is scaled with the batch size)
        datasets: dataset paths and types (classification/detection)
        train_config: Training configuration parameters (e.g., epochs, optimizer)
        variant: Encoder size, one of T, S, M, B, L, XL
        norm_pix_loss: Reconstruct per-patch normalized pixels (MAE paper, better representations).
            Note: this is on by default, the previous version always used raw pixels.
        mlp_ratio: Encoder MLP ratio (None -> variant, 4.0)
        decoder_mlp_ratio: Decoder MLP ratio
        drop_path_rate: Stochastic depth of the encoder (MAE pre-training uses none)
        weight_decay: AdamW weight decay for matrices (biases, norms and tokens get none)
        lr_schedule: "constant" (previous behaviour) or "cosine"
        warmup_epochs: Linear warmup epochs, only with the cosine schedule (0 = none; may be fractional)
        min_lr: Final learning rate of the cosine schedule
        grad_clip: Max gradient norm (0 = no clipping)
    """

    in_chans: int = 3
    image_size: int = 224
    patch_size: int = 16
    encoder_dim: int | None = None
    encoder_depth: int | None = None
    encoder_heads: int | None = None
    decoder_dim: int | None = None
    decoder_depth: int | None = None
    decoder_heads: int | None = None
    mask_ratio: float = 0.75
    learning_rate: float = 1.5e-4
    datasets: tuple[dict[str, str], ...] = field(
        default_factory=lambda: (
            {"path": "./data/images1", "type": "detection"},
            {"path": "./data/images2", "type": "classification"},
        )
    )
    train_config: TrainConfig = field(default_factory=TrainConfig)

    # ---- added fields (appended, so positional construction of the old fields keeps working) ----
    variant: str = "B"  # one of ViTVariant: T, S, M, B, L, XL
    norm_pix_loss: bool = True
    mlp_ratio: float | None = None  # None -> from variant
    decoder_mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    weight_decay: float = 0.05
    lr_schedule: str = "constant"
    warmup_epochs: float = 0.0
    min_lr: float = 0.0
    grad_clip: float = 0.0

    def __post_init__(self):
        assert all(t in ["classification", "detection"] for t in (d["type"] for d in self.datasets)), (
            "Dataset types must be either 'classification' or 'detection'."
        )

        variant = str(self.variant).strip().upper()
        if variant not in ViTVariant.__members__:
            raise ValueError(f"Unknown variant {self.variant!r}, expected one of {list(ViTVariant.__members__)}")
        self.variant = variant
        v = ViTVariant[variant]
        names = (*_VARIANT_FIELDS, "mlp_ratio")
        self._explicit = frozenset(name for name in names if getattr(self, name) is not None)

        dec_dim, dec_depth, dec_heads = DECODER_DEFAULTS[variant]
        defaults = {
            "encoder_dim": v.embed_dim,
            "encoder_depth": v.depth,
            "encoder_heads": v.num_heads,
            "mlp_ratio": v.mlp_ratio,
            "decoder_dim": dec_dim,
            "decoder_depth": dec_depth,
            "decoder_heads": dec_heads,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, value)

        if self.patch_size <= 0 or self.image_size < self.patch_size:
            raise ValueError("image_size must be >= patch_size > 0")
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")
        if self.decoder_dim % self.decoder_heads != 0:
            raise ValueError(
                f"decoder_dim ({self.decoder_dim}) must be divisible by decoder_heads ({self.decoder_heads})"
            )
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")
        if self.warmup_epochs < 0 or self.min_lr < 0 or self.grad_clip < 0 or self.weight_decay < 0:
            raise ValueError("warmup_epochs, min_lr, grad_clip and weight_decay must be >= 0")
        self.vit_config()  # validates the encoder (head divisibility, drop rates, ...)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2

    def yaml_dict(self) -> dict:
        """Field dict for YAML dumps: values that came from the variant are written as None."""
        d = asdict(self)
        for name in (*_VARIANT_FIELDS, "mlp_ratio"):
            if name not in self._explicit:
                d[name] = None
        return d

    def vit_config(self) -> ViTConfig:
        """ViTConfig of the encoder: CLS token, qkv bias (as in the MAE reference), fixed sin-cos pos-embed."""
        return ViTConfig.from_variant(
            ViTVariant[self.variant],
            patch_size=self.patch_size,
            base_res=self.image_size,
            in_chans=self.in_chans,
            add_cls_token=True,
            qkv_bias=True,
            drop_path_rate=self.drop_path_rate,
            embed_dim=self.encoder_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            mlp_ratio=self.mlp_ratio,
        )


def mae_config(variant: str = "B", patch_size: int = 16, **overrides) -> MAEConfig:
    """MAE config for any ViT variant, e.g. ``mae_config("L", 16)`` or ``mae_config("S", 8, image_size=64)``."""
    return MAEConfig(variant=variant, patch_size=patch_size, **overrides)


# Smaller version for prototyping, as reference
MAE_MINI_CONFIG: Final[MAEConfig] = MAEConfig(
    in_chans=3,
    image_size=32,
    patch_size=4,
    variant="S",
    encoder_depth=8,  # S has 384 wide / 6 heads; the mini model is 8 blocks deep
    decoder_dim=192,
    decoder_depth=4,
    decoder_heads=6,
    mask_ratio=0.75,
    learning_rate=1e-3,
    datasets=tuple(),
    train_config=TrainConfig(),
)
