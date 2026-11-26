from dataclasses import asdict, dataclass, field

from components.definitions.train_config import TrainConfig
from components.vit.vit_config import ViTConfig, ViTVariant

#################################################
# ViT variants (see ViTVariant), selected via DINOConfig.variant
#################################################
# T  (ViT-Ti):  embed_dim 192,  depth 12, num_heads 3
# S  (ViT-S):   embed_dim 384,  depth 12, num_heads 6   (ViT-S/16, default)
# M  (ViT-M):   embed_dim 512,  depth 12, num_heads 8
# B  (ViT-B):   embed_dim 768,  depth 12, num_heads 12  (ViT-B/16)
# L  (ViT-L):   embed_dim 1024, depth 24, num_heads 16  (ViT-L/16)
# XL (ViT-H):   embed_dim 1280, depth 32, num_heads 16  (ViT-H/14: set patch_size = 14)
#################################################

# Fields that default to None and are then derived (variant / teacher_base_res).
_DERIVED_FIELDS = ("embed_dim", "depth", "num_heads", "mlp_ratio", "model_base_res")


@dataclass
class DINOConfig:
    """DINO config with a configurable ViT backbone (default: ViT-S/16).

    Pick a ``variant`` (T, S, M, B, L, XL). ``embed_dim``, ``depth``, ``num_heads`` and ``mlp_ratio``
    default to ``None`` and are then taken from the variant; set one explicitly to override it
    (configs that spell out all values keep working). After construction every field holds its
    concrete value. ``yaml_dict()`` (used by ``dump_yaml``) writes ``null`` for derived fields, so a
    generated YAML can be switched to another variant by editing ``variant`` only.
    Programmatically use ``dino_config(variant, ...)`` rather than ``dataclasses.replace``.
    """

    # Session configs
    num_teachers: int = 2
    num_students: int = 8
    teacher_base_res: int = 224
    student_base_res: int = 96
    model_base_res: int | None = None
    out_dim: int = 65536
    hidden_dim: int = 2048

    # DINO head
    bottleneck_dim: int = 256
    head_nlayers: int = 3
    norm_last_layer: bool = True  # False can be better for ViT-S but may destabilize training

    # ViT configs, default is ViT-s-16
    variant: str = "S"  # one of ViTVariant: T, S, M, B, L, XL
    patch_size: int = 16
    embed_dim: int | None = None  # None -> from variant
    depth: int | None = None  # None -> from variant
    num_heads: int | None = None  # None -> from variant
    mlp_ratio: float | None = None  # None -> from variant
    drop_path_rate: float = 0.1
    qkv_bias: bool = True

    # Augmentation: DINO recipe (gaussian blur, solarization, color jitter with p=0.8).
    # False keeps the plain pipeline (crop, flip, jitter, grayscale).
    dino_augmentations: bool = True

    # Training configs
    data_dirs: tuple[str, ...] = ("./data/kaggle/",)
    train_config: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        variant = str(self.variant).strip().upper()
        if variant not in ViTVariant.__members__:
            raise ValueError(f"Unknown variant {self.variant!r}, expected one of {list(ViTVariant.__members__)}")
        self.variant = variant
        self._explicit = frozenset(name for name in _DERIVED_FIELDS if getattr(self, name) is not None)

        v = ViTVariant[variant]
        defaults = {
            "embed_dim": v.embed_dim,
            "depth": v.depth,
            "num_heads": v.num_heads,
            "mlp_ratio": v.mlp_ratio,
            "model_base_res": self.teacher_base_res,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, value)

        if self.head_nlayers < 1:
            raise ValueError("head_nlayers must be >= 1")
        self.vit_config()  # validates the backbone (head divisibility, drop rates, ...)

    def yaml_dict(self) -> dict:
        """Field dict for YAML dumps: derived values are written as None."""
        d = asdict(self)
        for name in _DERIVED_FIELDS:
            if name not in self._explicit:
                d[name] = None
        return d

    def vit_config(self) -> ViTConfig:
        """ViTConfig of the backbone (with CLS token, which DINO needs)."""
        return ViTConfig.from_variant(
            ViTVariant[self.variant],
            patch_size=self.patch_size,
            base_res=self.model_base_res,
            add_cls_token=True,
            drop_path_rate=self.drop_path_rate,
            qkv_bias=self.qkv_bias,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )


def dino_config(variant: str = "S", patch_size: int = 16, **overrides) -> DINOConfig:
    """DINO config for any ViT variant, e.g. ``dino_config("B", 16)`` or ``dino_config("XL", 14)``."""
    return DINOConfig(variant=variant, patch_size=patch_size, **overrides)
