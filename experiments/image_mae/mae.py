"""
Masked Autoencoder (MAE) Pre-training Implementation

This module implements a self-supervised Masked Autoencoder for vision tasks,
based on the MAE paper (He et al., 2021). MAE learns rich visual representations
by randomly masking image patches and reconstructing them.

Key Components:
- PatchEmbed: Converts images to patch embeddings
- TransformerBlock: Vision Transformer blocks for encoder/decoder
- MAE: Main autoencoder model with asymmetric encoder-decoder
- Dataset loaders for CIFAR-10, ImageNet, and COCO

Improvement Opportunities:
1. Add support for distributed training (DistributedDataParallel)
2. Implement gradient checkpointing for memory efficiency
3. Add mixed precision training for all components
4. Support for different backbone depths (tiny, base, large variants)
5. Add tensorboard logging for training visualization
6. Implement learning rate warmup scheduling
7. Add model ensemble support for inference
"""

import argparse
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from components.utils.logger import configure_logger, logger

DEFAULT_KAGGLE_DATASETS: Dict[str, str] = {
    # Default full ImageNet dataset for MAE pre-training.
    "imagenet": "dimensi0n/imagenet-256",
    # Smaller ImageNet-style subset kept available as an explicit alias.
    "imagenet_mini": "ifigotin/imagenetmini-1000",
    # COCO 2017 dataset for MAE pre-training (large, ~20GB compressed).
    "coco": "awsaf49/coco-2017-dataset",
}


@dataclass(frozen=True)
class MAEVariantConfig:
    """
    Configuration for different MAE model variants.

    Attributes:
        dataset_size: Total number of samples in the dataset
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
    """

    dataset_size: int
    image_size: int
    patch_size: int
    batch_size: int
    encoder_dim: int
    encoder_depth: int
    encoder_heads: int
    decoder_dim: int
    decoder_depth: int
    decoder_heads: int
    mask_ratio: float
    learning_rate: float


VARIANT_CONFIG = {
    # Fits typical 4GB GPUs for debugging (with AMP).
    "cifar10": MAEVariantConfig(
        dataset_size=50_000,
        image_size=32,
        patch_size=4,
        batch_size=32 * 1,
        encoder_dim=384,
        encoder_depth=8,
        encoder_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        decoder_heads=6,
        mask_ratio=0.75,
        learning_rate=1e-3,
    ),
    # MAE-Base style scale for larger GPUs (32-40GB+ suggested).
    "imagenet": MAEVariantConfig(
        dataset_size=1_281_167,
        image_size=256,  # ↑ use full 256 for better spatial detail
        patch_size=16,
        batch_size=16 * 1,
        # ---------------- Encoder ----------------
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        # ---------------- Decoder ----------------
        decoder_dim=768,  # ↑ stronger decoder for sharper teacher features
        decoder_depth=8,  # keep 8 (good balance)
        decoder_heads=12,  # ↓ from 16 → 12 (matches encoder, more stable)
        # ---------------- Masking ----------------
        mask_ratio=0.6,  # ↓ from 0.75 → 0.6 (better for dense tasks)
        # ---------------- Optimization ----------------
        learning_rate=1.5e-4,  # unchanged; still optimal for MAE-B
        # missing fields you should add:
        # use_sin_pos_embed=True,   # recommended: 2D sin-cos positional embeddings
        # drop_path_rate=0.1,       # stable for ViT-B encoder
        # decoder_drop_path_rate=0.0,
        # loss_type="mse",          # better for YOLO distillation
    ),
    # Same MAE scale as ImageNet but intended for smaller ImageNet-style subsets.
    "imagenet_mini": MAEVariantConfig(
        dataset_size=100_000,
        image_size=224,
        patch_size=16,
        batch_size=32 * 1,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
        learning_rate=1.5e-4,
    ),
    "coco": MAEVariantConfig(
        dataset_size=1_281_167,
        image_size=256,
        patch_size=16,
        batch_size=32 * 1,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
        learning_rate=1.5e-4,
    ),
}


class ImageOnlyDataset(Dataset):
    """
    Wrapper dataset that extracts only images from label-aware datasets.

    Used to convert classification datasets (which return (image, label) tuples)
    into image-only datasets for unsupervised pre-training.
    """

    def __init__(self, dataset: Dataset) -> None:
        """
        Args:
            dataset: A dataset that returns tuples containing image as first element
        """
        self.dataset = dataset

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get image at given index (discarding labels and other metadata).

        Args:
            idx: Index of sample to retrieve

        Returns:
            Image tensor
        """
        img, *_ = self.dataset[idx]
        return img


def make_cifar10_dataloader(root: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    """
    Create CIFAR-10 dataloader for MAE pre-training.

    CIFAR-10 is a 32x32 image dataset with 50K training samples. Good for
    quick experiments and testing on small GPUs (~4GB).

    Args:
        root: Root directory to store/load CIFAR-10 dataset
        batch_size: Number of samples per batch
        num_workers: Number of parallel data loading workers

    Returns:
        DataLoader yielding batches of (32, 32, 3) images

    Note:
        - Images are normalized with CIFAR-10 standard statistics
        - Dataset is downloaded automatically if not present
        - Shuffling enabled for training randomness
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        ImageOnlyDataset(dataset),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_imagenet_dataloader(
    root: str,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 8,
    train: bool = True,
) -> DataLoader:
    """
    Create ImageNet dataloader for MAE pre-training or fine-tuning.

    ImageNet-1K has 1.2M training samples at variable resolutions.
    Images are resized to `image_size` for ViT compatibility.

    Args:
        root: Root directory containing train/ and val/ subdirectories
        batch_size: Number of samples per batch
        image_size: Size to resize and crop images to
        num_workers: Number of parallel data loading workers
        train: If True, load training split; else load validation split

    Returns:
        DataLoader yielding batches of (image_size, image_size, 3) images

    Raises:
        FileNotFoundError: If expected directory structure is not found

    Note:
        - ImageNet must be manually downloaded and organized
        - Expected layout: <root>/train/<class_name>/*.JPEG
        - Shuffling enabled only for training split
        - Images normalized with ImageNet statistics

    Improvement: Consider adding:
        - Random augmentation (RandomResizedCrop, RandomHorizontalFlip) for training
        - Progressive resizing for faster convergence
        - Memory mapping for very large datasets
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    split = "train" if train else "val"
    dataset = datasets.ImageFolder(
        root=os.path.join(root, split),
        transform=transform,
    )

    return DataLoader(
        ImageOnlyDataset(dataset),
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_coco_dataloader(
    root: str,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 8,
    train: bool = True,
) -> DataLoader:
    """
    Create COCO 2017 dataloader for MAE pre-training.

    COCO has 118K training images with diverse objects, scenes, and textures.
    Images are resized and randomly cropped to `image_size` for consistency.

    Args:
        root: Root directory containing train2017, val2017, and annotations/
        batch_size: Number of samples per batch
        image_size: Size to resize and crop images to
        num_workers: Number of parallel data loading workers
        train: If True, load training split; else load validation split

    Returns:
        DataLoader yielding batches of (image_size, image_size, 3) images

    Note:
        - Uses RandomResizedCrop for scale and aspect ratio augmentation
        - Expected layout: <root>/train2017/, <root>/val2017/, <root>/annotations/
        - Annotations are loaded but discarded (image-only dataset)
        - Good dataset for learning diverse visual features

    Improvement: Consider adding:
        - Custom collate_fn to handle variable-size images
        - Stratified sampling by object distribution
        - On-the-fly image processing for memory efficiency
    """
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    split = "train" if train else "val"
    image_root = os.path.join(root, f"{split}2017")
    ann_file = os.path.join(root, "annotations", f"instances_{split}2017.json")

    dataset = datasets.CocoDetection(
        root=image_root,
        annFile=ann_file,
        transform=transform,
    )

    return DataLoader(
        ImageOnlyDataset(dataset),
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def _find_dataset_root_by_markers(base_dir: Path, required_dirs: Tuple[str, ...]) -> Optional[Path]:
    """
    Recursively find a directory containing all required marker subdirectories.

    Searches from shallowest to deepest paths to prefer top-level directories.
    Useful for finding dataset roots when they may be nested in archive extractions.

    Args:
        base_dir: Starting directory for search
        required_dirs: Tuple of directory names that must all be present

    Returns:
        Path to directory containing all markers, or None if not found

    Example:
        >>> root = _find_dataset_root_by_markers(Path('./data'), ('train', 'val'))
        # Returns path containing both train/ and val/ subdirectories

    Improvement: Consider adding:
        - Symbolic link resolution
        - Caching of found paths for repeated searches
        - Timeout for very deep directory structures
    """
    candidates = [base_dir]
    candidates.extend(p for p in base_dir.rglob("*") if p.is_dir())

    for path in candidates:
        if all((path / marker).exists() for marker in required_dirs):
            return path
    return None


def _download_kaggle_dataset(dataset_slug: str, dest_dir: Path) -> None:
    """
    Download and extract a Kaggle dataset.

    Requires kaggle package and authentication credentials at ~/.kaggle/kaggle.json
    Automatically handles nested zip files in extracted archives.

    Args:
        dataset_slug: Kaggle dataset identifier (format: 'owner/dataset')
        dest_dir: Destination directory for downloaded files

    Raises:
        RuntimeError: If kaggle package is missing or authentication fails

    Example:
        >>> _download_kaggle_dataset('awsaf49/coco-2017-dataset', Path('./data/coco'))

    Improvement: Consider adding:
        - Resumable downloads for large files
        - Download progress bar with tqdm
        - Checksum verification after download
        - Retry logic for network failures
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Kaggle support requires the 'kaggle' package. Install with: pip install kaggle") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local auth setup
        raise RuntimeError(
            "Kaggle authentication failed. Place credentials at ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)

    # Some Kaggle datasets contain zip files inside the first extracted directory.
    for zip_path in dest_dir.rglob("*.zip"):
        extract_dir = zip_path.parent
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def resolve_variant_data_root(
    variant: str,
    data_root: str,
    use_kaggle: bool,
    kaggle_imagenet_dataset: str,
    kaggle_imagenet_mini_dataset: str,
    kaggle_coco_dataset: str,
) -> str:
    """
    Resolve dataset root path for a given MAE variant with auto-download support.

    Handles four different dataset types with fallback logic:
    - cifar10: Auto-downloads if not present
    - imagenet: Looks for existing structure, optionally downloads from Kaggle
    - imagenet_mini: Looks for existing structure, optionally downloads from Kaggle
    - coco: Looks for existing structure, optionally downloads from Kaggle

    Args:
        variant: Dataset variant ('cifar10', 'imagenet', 'imagenet_mini', or 'coco')
        data_root: Base directory to search for or download datasets
        use_kaggle: Enable Kaggle auto-download if dataset not found
        kaggle_imagenet_dataset: Kaggle slug for ImageNet (256x256) alternative
        kaggle_imagenet_mini_dataset: Kaggle slug for ImageNet-mini alternative
        kaggle_coco_dataset: Kaggle slug for COCO alternative

    Returns:
        Resolved path to dataset root

    Raises:
        FileNotFoundError: If dataset not found and use_kaggle is False
        ValueError: If variant is not recognized

    Note:
        - CIFAR-10 downloads automatically via torchvision
        - ImageNet and ImageNet-Mini requires manual download or Kaggle credentials
        - ImageNet is ImageNet-256 variant (1.2M images, 256x256), needs preprocessing
        - COCO download is large (~20GB compressed)

    Improvement: Consider adding:
        - Parallel downloads for large files
        - Dataset integrity verification
        - Support for custom dataset paths
    """
    root = Path(data_root)
    if variant == "cifar10":
        return str(root)

    if variant in {"imagenet", "imagenet_mini"}:
        prepared_root = _find_dataset_root_by_markers(root, ("train", "val"))
        if prepared_root is not None:
            return str(prepared_root)

        if not use_kaggle:
            raise FileNotFoundError(
                f"Could not find ImageNet layout under {root}. Expected directories: train/ and val/."
            )

        cache_dir = root / "kaggle" / variant
        prepared_root = _find_dataset_root_by_markers(cache_dir, ("train", "val"))
        if prepared_root is None:
            # imagenet_mini is already split into train/val; imagenet-256 is not, so we place its download under train/.
            if variant == "imagenet_mini":
                _download_kaggle_dataset(kaggle_imagenet_mini_dataset, cache_dir)
            else:
                dir_train = cache_dir / "train"
                _download_kaggle_dataset(kaggle_imagenet_dataset, dir_train)
                dir_val = cache_dir / "val"
                dir_val.mkdir(parents=True, exist_ok=True)
            prepared_root = _find_dataset_root_by_markers(cache_dir, ("train", "val"))

        if prepared_root is None:
            raise FileNotFoundError("Downloaded ImageNet dataset does not expose train/ and val/ directories.")
        return str(prepared_root)

    # coco
    prepared_root = _find_dataset_root_by_markers(root, ("train2017", "val2017", "annotations"))
    if prepared_root is not None:
        return str(prepared_root)

    if not use_kaggle:
        raise FileNotFoundError(
            f"Could not find COCO layout under {root}. Expected train2017/, val2017/, annotations/."
        )

    cache_dir = root / "kaggle" / "coco"
    prepared_root = _find_dataset_root_by_markers(cache_dir, ("train2017", "val2017", "annotations"))
    if prepared_root is None:
        _download_kaggle_dataset(kaggle_coco_dataset, cache_dir)
        prepared_root = _find_dataset_root_by_markers(cache_dir, ("train2017", "val2017", "annotations"))

    if prepared_root is None:
        raise FileNotFoundError("Downloaded COCO dataset does not expose train2017/, val2017/, annotations/.")
    return str(prepared_root)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) with GeLU-like activation.

    Standard MLP used in transformer blocks: Linear -> ReLU6 -> Dropout -> Linear -> Dropout
    Expands features by mlp_ratio and then projects back to original dimension.

    Architecture:
        Input -> FC(dim -> hidden_dim) -> ReLU6 -> Dropout -> FC(hidden_dim -> dim) -> Dropout

    Improvement: Consider adding:
        - GELU activation instead of ReLU6 for better performance
        - Optional layer normalization
        - Depthwise separable convolutions for efficiency
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        Args:
            dim: Input and output feature dimension
            mlp_ratio: Expansion ratio for hidden layer (hidden_dim = dim * mlp_ratio)
            dropout: Dropout probability
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, dim)

        Returns:
            Output tensor of same shape as input
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Vision Transformer (ViT) encoder/decoder block with self-attention and MLP.

    Standard transformer architecture with pre-normalization:
    Implements: x -> LayerNorm -> MultiheadAttention -> x + attn_out
               x -> LayerNorm -> MLP -> x + mlp_out

    This design improves training stability and convergence compared to post-normalization.

    Improvement: Consider adding:
        - Flash attention for faster computation
        - Sparse attention patterns for longer sequences
        - Rotary position embeddings for better length extrapolation
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            mlp_ratio: Expansion ratio for MLP hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of same shape
        """
        # Pre-normalization for stability
        attn_in = self.norm1(x)
        # Self-attention
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        # Residual connection, in order to preserve information and improve gradient flow
        x = x + attn_out
        # Pre-normalization for MLP, again for stability
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings using a convolutional projection.

    Divides input image into non-overlapping patches and linearly embeds them.
    This is the standard approach in Vision Transformers (ViT).

    Implementation: Patches are extracted using Conv2d with stride=patch_size,
    then flattened and projected to embed_dim.

    Example:
        For 224x224 image with patch_size=16 and embed_dim=768:
        - Output shape: (batch, 196, 768) where 196 = (224/16)^2

    Improvement: Consider adding:
        - Learnable patch projection instead of just convolution
        - Overlapping patches for smoother transitions
        - Adaptive patch sizing based on image content
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        """
        Args:
            img_size: Input image size (assumes square images)
            patch_size: Size of each patch
            in_chans: Number of input channels (3 for RGB)
            embed_dim: Output embedding dimension

        Raises:
            ValueError: If img_size is not divisible by patch_size
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches and project to embedding space.

        Args:
            x: Input images of shape (batch, in_chans, img_size, img_size)

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        # Use convolution to extract patches and project to embedding dimension
        x = self.proj(x)  # (batch, embed_dim, grid_size, grid_size)
        # Flatten spatial dimensions and transpose to (batch, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


def build_2d_sincos_position_embedding(grid_size: int, embed_dim: int, add_cls_token: bool = True) -> torch.Tensor:
    """
    Create 2D sinusoidal position embeddings for Vision Transformer.

    Sinusoidal embeddings have several advantages:
    - No learnable parameters (consistent across datasets)
    - Can extrapolate to longer sequences
    - Each dimension encodes different frequencies

    The embedding combines separate sinusoidal patterns for height and width:
    pos_embed = [sin(w_h*h), cos(w_h*h), sin(w_w*w), cos(w_w*w)]
    where w are frequency weights following transformer conventions.

    Args:
        grid_size: Grid dimension (grid_size x grid_size patches)
        embed_dim: Embedding dimension (must be divisible by 4)
        add_cls_token: If True, prepend zeros for CLS token

    Returns:
        Position embeddings of shape (1, num_patches+cls, embed_dim)

    Raises:
        ValueError: If embed_dim is not divisible by 4

    Example:
        >>> pos_emb = build_2d_sincos_position_embedding(14, 768)
        >>> pos_emb.shape
        torch.Size([1, 197, 768])  # 196 patches + 1 CLS token

    Improvement: Consider adding:
        - Learnable position biases for fine-tuning
        - RoPE (Rotary Position Embeddings) for better extrapolation
        - Interpolation strategy for different resolutions
    """
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos position embeddings")

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    pos_h = grid[0].reshape(-1)
    pos_w = grid[1].reshape(-1)

    # Compute frequency weights for sinusoidal embeddings, following the transformer convention
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000**omega)  # Frequency scaling for sinusoidal embeddings

    out_h = torch.outer(pos_h, omega)
    out_w = torch.outer(pos_w, omega)
    pos_embed = torch.cat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)

    if add_cls_token:
        # Prepend a zero vector for the CLS token position embedding
        cls_pos = torch.zeros(1, embed_dim, dtype=pos_embed.dtype)
        pos_embed = torch.cat([cls_pos, pos_embed], dim=0)

    return pos_embed.unsqueeze(0)


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised visual representation learning.

    MAE learns representations by:
    1. Randomly masking 75% of image patches
    2. Encoding only the visible patches with a transformer
    3. Decoding all patches (including masked) to reconstruct the image
    4. Computing reconstruction loss only on masked patches

    This asymmetric design (small encoder, large decoder) learns strong representations
    while being computationally efficient. Pre-training on unlabeled data enables better
    fine-tuning on downstream tasks like object detection (YOLO).

    Architecture:
    - Encoder: Vision Transformer (12 blocks, 768-dim for ImageNet)
    - Decoder: Vision Transformer (8 blocks, 512-dim for ImageNet)
    - Patch embedding: 16x16 patches at 224x224 resolution (196 patches)

    Key hyperparameters:
    - mask_ratio: Fraction of patches to mask (typically 0.75 = 75%)
    - encoder_dim: Encoder embedding dimension
    - decoder_dim: Decoder embedding dimension (usually smaller)
    - Position embeddings: Sinusoidal (non-learnable, freezable)

    Improvement opportunities:
        - Support variable input resolutions
        - Gradient checkpointing for memory efficiency
        - Distributed training with DDP/FSDP
        - Support for different masking strategies (block masking, temporal masking)
        - Exponential Moving Average (EMA) updates for stability
        - Contrastive learning combined with reconstruction
    """

    def __init__(self, variant: str, in_chans: int = 3) -> None:
        """
        Initialize MAE model with variant-specific configuration.

        Args:
            variant: Model variant ('cifar10', 'imagenet', or 'coco')
            in_chans: Number of input channels (3 for RGB images)

        Raises:
            KeyError: If variant not in VARIANT_CONFIG

        Example:
            >>> mae = MAE('imagenet')
            >>> mae.load_checkpoint()  # Load pre-trained weights
        """
        super().__init__()

        self.cfg = VARIANT_CONFIG[variant]
        self.in_chans = in_chans

        # Patch embedding layer: converts images to patch embeddings
        self.patch_embed = PatchEmbed(
            img_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
            in_chans=in_chans,
            embed_dim=self.cfg.encoder_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = self.cfg.patch_size * self.cfg.patch_size * in_chans

        # Initialize learnable tokens and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.encoder_dim))
        self.pos_embed_enc = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.cfg.encoder_dim), requires_grad=False
        )

        # Build encoder blocks: a stack of transformer blocks for encoding visible patches
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=self.cfg.encoder_dim, num_heads=self.cfg.encoder_heads)
                for _ in range(self.cfg.encoder_depth)
            ]
        )
        # Layer normalization after encoder blocks for stability
        self.encoder_norm = nn.LayerNorm(self.cfg.encoder_dim)

        # Build decoder components: embedding, mask token, position embeddings, and transformer blocks
        self.decoder_embed = nn.Linear(self.cfg.encoder_dim, self.cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.cfg.decoder_dim))
        self.pos_embed_dec = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.cfg.decoder_dim), requires_grad=False
        )

        # Build decoder blocks: a stack of transformer blocks for reconstructing masked patches
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=self.cfg.decoder_dim, num_heads=self.cfg.decoder_heads)
                for _ in range(self.cfg.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.cfg.decoder_dim)
        self.decoder_pred = nn.Linear(self.cfg.decoder_dim, self.patch_dim)

        self._init_weights()

        self.path_final_ckpt = Path(__file__).resolve().parent / "mae_checkpoints" / variant / "final.pth"
        if not self.path_final_ckpt.parent.exists():
            self.path_final_ckpt.parent.mkdir(parents=True, exist_ok=True)

    def _init_weights(self) -> None:
        """
        Initialize model parameters using proper initialization schemes.

        Strategy:
        - CLS and MASK tokens: Normal distribution (std=0.02)
        - Position embeddings: Pre-computed sinusoidal (non-learnable)
        - Linear layers: Xavier uniform initialization
        - LayerNorm: Constant initialization (bias=0, weight=1)

        Proper initialization is critical for transformer training stability.
        """
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        with torch.no_grad():
            self.pos_embed_enc.copy_(
                build_2d_sincos_position_embedding(self.patch_embed.grid_size, self.cfg.encoder_dim)
            )
            self.pos_embed_dec.copy_(
                build_2d_sincos_position_embedding(self.patch_embed.grid_size, self.cfg.decoder_dim)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches for comparison with predictions.

        Reshapes images from (B, C, H, W) to (B, num_patches, patch_dim)
        where patch_dim = patch_size^2 * C = 48 for 16x16 RGB patches.

        Args:
            imgs: Images of shape (batch, channels, height, width)

        Returns:
            Patches of shape (batch, num_patches, patch_dim)

        Example:
            >>> imgs = torch.randn(2, 3, 224, 224)
            >>> patches = mae.patchify(imgs)  # (2, 196, 768)
        """
        p = self.cfg.patch_size
        n = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, n, p, n, p))
        x = torch.einsum("nchpwq->nhwpqc", x)  # Rearrange to (batch, n, n, p, p, channels)
        return x.reshape(shape=(imgs.shape[0], n * n, p * p * self.in_chans))

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images for visualization.

        Reverses the patchify operation: (B, num_patches, patch_dim) -> (B, C, H, W)

        Args:
            x: Patches of shape (batch, num_patches, patch_dim)

        Returns:
            Images of shape (batch, channels, height, width)

        Note:
            num_patches must be a perfect square: sqrt(num_patches) = height/patch_size
        """
        p = self.cfg.patch_size
        n = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], n, n, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)  # Rearrange to (batch, channels, n, p, n, p)
        return x.reshape(shape=(x.shape[0], self.in_chans, n * p, n * p))

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches and keep only visible patches.

        The core of MAE: randomly select patches to mask, encode only visible patches.
        This forces the model to learn meaningful representations by reconstruction.

        Algorithm:
        1. Generate random noise for each patch
        2. Sort by noise to determine mask/keep split
        3. Keep patches with lowest noise (by design, random)
        4. Create restoration indices to unpermute patches

        Args:
            x: Patch embeddings of shape (batch, num_patches, dim)
            mask_ratio: Fraction of patches to mask (e.g., 0.75 = mask 75%)

        Returns:
            Tuple of:
            - x_masked: Visible patches only (batch, num_keep, dim)
            - mask: Binary mask (batch, num_patches) where 1=masked, 0=visible
            - ids_restore: Indices to restore original patch order

        Example:
            >>> x = torch.randn(2, 196, 768)
            >>> x_masked, mask, restore_ids = mae.random_masking(x, mask_ratio=0.75)
            >>> x_masked.shape  # (2, 49, 768) - only 25% of patches
        """
        n, l, d = x.shape  # noqa: E741
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder_full(self, imgs: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """
        Encode all patches through the transformer encoder without masking.

        This helper is used by the YOLO distillation path to obtain deterministic
        teacher features from the MAE encoder. It mirrors the visible-patch encoder
        logic but skips masking unless a positive mask ratio is requested.
        """
        x, _, _ = self.forward_encoder(imgs, mask_ratio)
        return x

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode visible patches through transformer encoder.

        Process:
        1. Convert images to patch embeddings
        2. Add positional embeddings
        3. Randomly mask patches
        4. Prepend CLS token
        5. Process through transformer encoder blocks
        6. Apply layer normalization

        Args:
            imgs: Input images (batch, 3, H, W)
            mask_ratio: Fraction of patches to mask

        Returns:
            Tuple of:
            - x: Encoded features (batch, num_keep+1, encoder_dim)
            - mask: Binary mask showing which patches were masked
            - ids_restore: Indices to restore original patch order
        """
        # 1. Patch embedding
        x = self.patch_embed(imgs)
        # 2. Positional embedding
        x = x + self.pos_embed_enc[:, 1:, :]

        # 3. Optional masking
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            # For full encoder: no masking
            mask = None
            ids_restore = None

        # 4. CLS token
        cls_token = self.cls_token + self.pos_embed_enc[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 5. Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)

        # 6. Final norm
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Decode encoder features and reconstruct all patches.

        Process:
        1. Project encoder embeddings to decoder dimension
        2. Create mask tokens for masked positions
        3. Concatenate visible and mask tokens in original order
        4. Add positional embeddings
        5. Process through transformer decoder blocks
        6. Predict patch values

        Args:
            x: Encoder output (batch, num_keep+1, encoder_dim)
            ids_restore: Indices to restore original patch order

        Returns:
            Predicted patches (batch, num_patches, patch_dim)

        Note:
            The decoder is asymmetrically designed to be smaller than encoder,
            which reduces computational cost while maintaining quality.
        """
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.pos_embed_dec

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return self.decoder_pred(x[:, 1:, :])

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MAE reconstruction loss.

        Key insight: Only compute loss on masked patches. This forces meaningful
        reconstruction of difficult regions while encoder focuses on visible patches.

        Loss = MSE(predicted_patches, target_patches) masked by mask, averaged over masked patches.

        Args:
            imgs: Original images (batch, 3, H, W)
            pred: Predicted patches (batch, num_patches, patch_dim)
            mask: Binary mask (batch, num_patches) where 1=masked

        Returns:
            Tuple of:
            - loss: Scalar loss value
            - target: Original patch values for visualization

        Note:
            Mean Squared Error (MSE) is used here. Alternatives:
            - L1 loss (more robust to outliers)
            - Smooth L1 loss (hybrid of L1 and L2)
            - Perceptual loss (using pre-trained discriminator)
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, target

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: encode, decode, compute loss.

        Args:
            imgs: Input images (batch, 3, H, W)

        Returns:
            Tuple of:
            - loss: Reconstruction loss on masked patches
            - pred: Predicted patches (batch, num_patches, patch_dim)
            - target: Target patches (batch, num_patches, patch_dim)
            - mask: Binary mask (batch, num_patches)
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, self.cfg.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss, target = self.forward_loss(imgs, pred, mask)
        return loss, pred, target, mask

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: Path to save checkpoint. If None, uses default path from config.

        Note:
            Model is moved to CPU before saving to avoid device-specific issues.
        """
        path_ckpt = self.path_final_ckpt if path is None else path

        state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}

        torch.save(state_dict_cpu, path_ckpt)

    def load_checkpoint(self, path: Optional[Path] = None, device: Optional[str] = None) -> None:
        """
        Load model checkpoint from disk.

        Args:
            path: Path to checkpoint file. If None, uses default path from config.
            device: Device to load checkpoint to ('cpu', 'cuda', 'cuda:0', etc.).
                   If None, uses current device.

        Note:
            Prints warning if checkpoint not found but doesn't raise exception.
        """
        path_ckpt = self.path_final_ckpt if path is None else path

        if not path_ckpt.exists():
            logger.info(f"{path_ckpt} not found, cannot load")
            return

        state_dict = torch.load(path_ckpt, map_location=device)
        self.load_state_dict(state_dict)


def mae_visualize(model: MAE, imgs: torch.Tensor, save_path: Path) -> None:
    """
    Visualize MAE reconstruction quality by showing original, masked, and reconstructed images.

    Creates a 3xN subplot figure where:
    - Row 0: Original images
    - Row 1: Images with masked patches zeroed out
    - Row 2: Reconstructed images from encoder-decoder

    Visually demonstrates what MAE learned to reconstruct masked regions.
    Good diagnostic tool for evaluating model training progress.

    Args:
        model: Trained MAE model (will be set to eval mode)
        imgs: Batch of input images (batch, 3, H, W)
        save_path: Path to save visualization image

    Note:
        - Shows maximum 6 images to keep figure readable
        - Clipped to [0, 1] range for display
        - Uses model.eval() and torch.no_grad() for inference

    Improvement: Consider adding:
        - Difference maps (original - reconstruction)
        - Uncertainty/confidence estimates
        - Progressive reconstruction frames (decoder layers)
        - Histogram of reconstruction errors
    """
    model.eval()
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(imgs, model.cfg.mask_ratio)
        pred = model.forward_decoder(latent, ids_restore)
        rec_imgs = model.unpatchify(pred)

        B, C, H, W = imgs.shape
        patch = model.cfg.patch_size

        mask = mask.unsqueeze(-1).repeat(1, 1, patch * patch * C)
        mask = model.unpatchify(mask)
        masked_imgs = imgs * (1 - mask)

    num_show = min(6, imgs.shape[0])
    fig, axes = plt.subplots(3, num_show, figsize=(3 * num_show, 9))

    for i in range(num_show):
        axes[0, i].imshow(imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(masked_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[1, i].set_title("Masked")
        axes[1, i].axis("off")

        axes[2, i].imshow(rec_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[2, i].set_title("Reconstructed")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def build_model_and_loader(
    variant: str,
    data_root: str = "./data",
    use_kaggle: bool = True,
    kaggle_imagenet_dataset: str = DEFAULT_KAGGLE_DATASETS["imagenet"],
    kaggle_imagenet_mini_dataset: str = DEFAULT_KAGGLE_DATASETS["imagenet_mini"],
    kaggle_coco_dataset: str = DEFAULT_KAGGLE_DATASETS["coco"],
) -> Tuple[MAEVariantConfig, MAE, DataLoader]:
    """
    Convenience function to build MAE model and corresponding dataloader.

    Handles:
    - Model instantiation and checkpoint loading
    - Dataset path resolution with auto-download
    - Dataloader creation

    Args:
        variant: Model variant ('cifar10', 'imagenet', 'coco')
        data_root: Root directory for datasets
        use_kaggle: Enable Kaggle auto-download
        kaggle_imagenet_dataset: Kaggle slug for ImageNet
        kaggle_coco_dataset: Kaggle slug for COCO

    Returns:
        Tuple of (config, model, dataloader)

    Example:
        >>> cfg, model, loader = build_model_and_loader('imagenet', './data')
        >>> for imgs in loader:
        ...     loss, pred, target, mask = model(imgs)
    """
    model = MAE(variant)
    model.load_checkpoint()

    cfg = VARIANT_CONFIG[variant]
    resolved_root = resolve_variant_data_root(
        variant=variant,
        data_root=data_root,
        use_kaggle=use_kaggle,
        kaggle_imagenet_dataset=kaggle_imagenet_dataset,
        kaggle_imagenet_mini_dataset=kaggle_imagenet_mini_dataset,
        kaggle_coco_dataset=kaggle_coco_dataset,
    )

    if variant == "cifar10":
        loader = make_cifar10_dataloader(root=resolved_root, batch_size=cfg.batch_size)
    elif variant in {"imagenet", "imagenet_mini"}:
        loader = make_imagenet_dataloader(
            root=resolved_root,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
        )
    else:
        loader = make_coco_dataloader(
            root=resolved_root,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
        )

    return cfg, model, loader


def train(
    variant: str = "cifar10",
    steps: int = -1,
    data_root: str = "./data",
    epoch: int = 0,
    use_kaggle: bool = True,
    kaggle_imagenet_dataset: str = DEFAULT_KAGGLE_DATASETS["imagenet"],
    kaggle_imagenet_mini_dataset: str = DEFAULT_KAGGLE_DATASETS["imagenet_mini"],
    kaggle_coco_dataset: str = DEFAULT_KAGGLE_DATASETS["coco"],
) -> None:
    """
    Train MAE model for one epoch with optional checkpointing and visualization.

    Training procedure:
    1. Load model, optimizer, and dataloader
    2. Iterate through batches
    3. Forward pass (encoding and decoding)
    4. Backward pass with optional mixed precision (AMP)
    5. Checkpoint every 10K steps
    6. Visualize reconstructions at epoch end

    Args:
        variant: Model variant ('cifar10', 'imagenet', 'coco')
        steps: Max steps per epoch (-1 = full epoch)
        data_root: Root directory for datasets
        epoch: Epoch number (used for logging/visualization)
        use_kaggle: Enable Kaggle auto-download
        kaggle_imagenet_dataset: Kaggle slug for ImageNet
        kaggle_coco_dataset: Kaggle slug for COCO

    Notes:
        - Uses AdamW optimizer with standard MAE hyperparameters
        - Mixed precision (AMP) enabled on CUDA for memory efficiency
        - Saves checkpoint every 10K steps and at epoch end
        - Generates reconstruction visualization at epoch end

    Improvement opportunities:
        - Add learning rate scheduling (cosine annealing, warmup)
        - Implement gradient accumulation for larger batch sizes
        - Add per-step validation on held-out set
        - Support distributed training (DDP)
        - Add tensorboard/wandb logging
        - Implement early stopping
        - Add exponential moving average (EMA) for better stability
    """
    cfg, model, loader = build_model_and_loader(
        variant,
        data_root=data_root,
        use_kaggle=use_kaggle,
        kaggle_imagenet_dataset=kaggle_imagenet_dataset,
        kaggle_imagenet_mini_dataset=kaggle_imagenet_mini_dataset,
        kaggle_coco_dataset=kaggle_coco_dataset,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)

    use_amp = device.type == "cuda"

    if use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    model.train()
    for step, imgs in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, pred, target, mask = model(imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, pred, target, mask = model(imgs)
            loss.backward()
            optimizer.step()

        if steps >= 0 and step >= steps:
            break
        elif (step + 1) % 10_000 == 0:
            logger.info(
                f"variant={variant} epoch={epoch} step={step} loss={float(loss):.6f} "
                f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)}"
            )

            model.save_checkpoint()

    logger.info(
        f"variant={variant} epoch={epoch} step={step} loss={float(loss):.6f} "
        f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)}"
    )
    model.save_checkpoint()

    path_vis = Path(f"mae_visualizations/{variant}")
    path_vis.mkdir(parents=True, exist_ok=True)
    mae_visualize(model, imgs, save_path=(path_vis / f"step_{epoch:06}_{step:06}.png"))


def main() -> None:
    """
    Command-line interface for MAE training.

    Parses arguments and launches training loop for specified epochs.
    Supports three dataset variants with automatic Kaggle download.

    Example usage:
        # CIFAR-10 training (quick)
        python main.py --variant cifar10 --epochs 10

        # ImageNet with Kaggle download
        python main.py --variant imagenet --epochs 100 --data-root ./data

        # COCO without Kaggle download (must have dataset ready)
        python main.py --variant coco --no-kaggle --data-root /path/to/coco
    """
    parser = argparse.ArgumentParser(description="MAE debug trainer with CIFAR-10 and ImageNet presets")
    parser.add_argument(
        "--variant", type=str, default="cifar10", choices=["cifar10", "imagenet", "imagenet_mini", "coco"]
    )
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for CIFAR-10 or ImageNet. CIFAR-10 downloads here. ImageNet must be prepared locally.",
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="Disable Kaggle auto-download for ImageNet/COCO. Requires datasets to already exist in --data-root.",
    )
    parser.add_argument(
        "--kaggle-imagenet-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASETS["imagenet"],
        help="Kaggle dataset slug for ImageNet-256 data (owner/dataset).",
    )
    parser.add_argument(
        "--kaggle-imagenet-mini-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASETS["imagenet_mini"],
        help="Kaggle dataset slug for ImageNet-mini data (owner/dataset).",
    )
    parser.add_argument(
        "--kaggle-coco-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASETS["coco"],
        help="Kaggle dataset slug for COCO-style data (owner/dataset).",
    )
    args = parser.parse_args()

    if args.variant not in VARIANT_CONFIG:
        supported = ", ".join(sorted(VARIANT_CONFIG))
        raise ValueError(f"Unknown variant '{args.variant}'. Supported: {supported}")

    for epoch_rel in range(args.epochs):
        epoch = args.start_epoch + epoch_rel
        train(
            variant=args.variant,
            steps=args.steps,
            data_root=args.data_root,
            epoch=epoch,
            use_kaggle=not args.no_kaggle,
            kaggle_imagenet_dataset=args.kaggle_imagenet_dataset,
            kaggle_imagenet_mini_dataset=args.kaggle_imagenet_mini_dataset,
            kaggle_coco_dataset=args.kaggle_coco_dataset,
        )


if __name__ == "__main__":
    configure_logger("mae")
    main()
