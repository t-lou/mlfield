from dataclasses import dataclass, field

from components.definitions.train_config import TrainConfig

#################################################
# Typical ViT config variants
#################################################
# ViT-s-16
# patch_size = 16
# embed_dim = 384
# depth = 12
# num_heads = 6
#
# ViT-b-16
# patch_size = 16
# embed_dim = 768
# depth = 12
# num_heads = 12
#
# ViT-l-16
# patch_size = 16
# embed_dim = 1024
# depth = 24
# num_heads = 16
#
# ViT-h-14
# patch_size = 14
# embed_dim = 1280
# depth = 32
# num_heads = 16
#################################################


@dataclass
class DINOConfig:
    # Session configs
    num_teachers: int = 2
    num_students: int = 8
    teacher_base_res: int = 224
    student_base_res: int = 96
    model_base_res: int | None = None
    out_dim: int = 65536
    hidden_dim: int = 2048

    # ViT configs, default is ViT-s-16
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6

    # Training configs
    data_dirs: tuple[str] = ("./data/kaggle/",)
    train_config: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.model_base_res is None:
            self.model_base_res = self.teacher_base_res
