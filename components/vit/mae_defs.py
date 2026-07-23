from dataclasses import dataclass, field
from typing import Final

from components.definitions.train_config import TrainConfig


@dataclass
class MAEConfig:
    """
    Configuration for different MAE model variants.

    Attributes:
        in_chans: Number of input channels (3 for RGB images)
        image_size: Input image resolution (pixels)
        patch_size: Size of image patches (pixels)
        batch_size: Training batch size
        encoder_dim: Embedding dimension for transformer encoder
        encoder_depth: Number of transformer blocks in encoder
        encoder_heads: Number of attention heads in encoder
        decoder_dim: Embedding dimension for transformer decoder
        decoder_depth: Number of transformer blocks in decoder
        decoder_heads: Number of attention heads in decoder
        mask_ratio: Fraction of patches to mask during training (0-1)
        learning_rate: Initial learning rate for optimization
        datasets: dataset paths and types (classification/detection)
        train_config: Training configuration parameters (e.g., epochs, optimizer)
    """

    in_chans: int = 3
    image_size: int = 224
    patch_size: int = 16
    encoder_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    decoder_dim: int = 512
    decoder_depth: int = 8
    decoder_heads: int = 16
    mask_ratio: float = 0.75
    learning_rate: float = 1.5e-4
    datasets: tuple[dict[str, str], ...] = field(
        default_factory=lambda: (
            {"path": "./data/images1", "type": "detection"},
            {"path": "./data/images2", "type": "classification"},
        )
    )
    train_config: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        assert all(t in ["classification", "detection"] for t in (d["type"] for d in self.datasets)), (
            "Dataset types must be either 'classification' or 'detection'."
        )


# Smaller version for prototyping, as reference
MAE_MINI_CONFIG: Final[MAEConfig] = MAEConfig(
    in_chans=3,
    image_size=32,
    patch_size=4,
    encoder_dim=384,
    encoder_depth=8,
    encoder_heads=6,
    decoder_dim=192,
    decoder_depth=4,
    decoder_heads=6,
    mask_ratio=0.75,
    learning_rate=1e-3,
    datasets=tuple(),
    train_config=TrainConfig(),
)
