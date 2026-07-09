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
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from components.dataset.image_only_dataset import ImageOnlyDataset
from components.utils.device import get_device, resolve_num_workers
from components.utils.logger import configure_logger, logger
from components.vit.mae import MAE, VARIANT_CONFIG, MAEVariantConfig

DEFAULT_KAGGLE_DATASETS: Dict[str, str] = {
    # Default full ImageNet dataset for MAE pre-training.
    "imagenet": "dimensi0n/imagenet-256",
    # Smaller ImageNet-style subset kept available as an explicit alias.
    "imagenet_mini": "ifigotin/imagenetmini-1000",
    # COCO 2017 dataset for MAE pre-training (large, ~20GB compressed).
    "coco": "awsaf49/coco-2017-dataset",
}


def make_dataloader_args(batch_size: int, num_workers: Optional[int] = None):
    resolved_num_workers = resolve_num_workers(num_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
    }
    if resolved_num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": resolved_num_workers,
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
        )
    else:
        loader_kwargs["num_workers"] = 0

    return loader_kwargs


def make_cifar10_dataloader(root: str, batch_size: int, num_workers: Optional[int] = None) -> DataLoader:
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

    loader_kwargs = make_dataloader_args(batch_size=batch_size, num_workers=num_workers)

    return DataLoader(dataset, **loader_kwargs)


def make_dataloader_from_classification_dataset(
    data_dirs: list[str],
    batch_size: int,
    image_size: int = 224,
    num_workers: Optional[int] = None,
    train: bool = True,
) -> DataLoader:
    """
    Create dataloader which are designed for classification.

    Images for classification tend to be similar in size and scale, and needs less cropping.

    Args:
        data_dirs: List of directories containing datasets.
        batch_size: Number of samples per batch
        image_size: Size to resize and crop images to
        num_workers: Number of parallel data loading workers
        train: If True, load training split; else load validation split

    Returns:
        DataLoader yielding batches of (image_size, image_size, 3) images

    Raises:
        FileNotFoundError: If expected directory structure is not found

    Note:
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

    dataset = ImageOnlyDataset(data_dirs, transform=transform)

    loader_kwargs = make_dataloader_args(batch_size=batch_size, num_workers=num_workers)

    return DataLoader(ImageOnlyDataset(dataset), **loader_kwargs)


def make_dataloader_from_detection_dataset(
    data_dirs: list[str],
    batch_size: int,
    image_size: int = 224,
    num_workers: Optional[int] = None,
    train: bool = True,
) -> DataLoader:
    """
    Create dataloader which are designed for detection.

    Detection dataset contains more objects and requires more cropping.

    Args:
        data_dirs: List of directories containing datasets.
        batch_size: Number of samples per batch
        image_size: Size to resize and crop images to
        num_workers: Number of parallel data loading workers
        train: If True, load training split; else load validation split

    Returns:
        DataLoader yielding batches of (image_size, image_size, 3) images

    Note:
        - Uses RandomResizedCrop for scale and aspect ratio augmentation

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

    dataset = ImageOnlyDataset(data_dirs, transform=transform)

    loader_kwargs = make_dataloader_args(batch_size=batch_size, num_workers=num_workers)

    return DataLoader(ImageOnlyDataset(dataset), **loader_kwargs)


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
    num_workers: Optional[int] = None,
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

    if variant == "cifar10":
        loader = make_cifar10_dataloader(root=data_root, batch_size=cfg.batch_size, num_workers=num_workers)
    elif variant in {"imagenet", "imagenet_mini"}:
        loader = make_dataloader_from_classification_dataset(
            data_dirs=[data_root],
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            num_workers=num_workers,
        )
    else:
        loader = make_dataloader_from_detection_dataset(
            data_dirs=[data_root],
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            num_workers=num_workers,
        )

    return cfg, model, loader


def train(
    variant: str = "cifar10",
    steps: int = -1,
    data_root: str = "./data",
    epoch: int = 0,
    num_workers: Optional[int] = None,
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
    )
    device = get_device()
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
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for datasets.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Defaults to automatic CPU-count based selection.",
    )
    args = parser.parse_args()

    if args.variant not in VARIANT_CONFIG:
        supported = ", ".join(sorted(VARIANT_CONFIG))
        raise ValueError(f"Unknown variant '{args.variant}'. Supported: {supported}")

    for epoch_rel in range(args.num_epochs):
        epoch = args.start_epoch + epoch_rel
        train(
            variant=args.variant,
            steps=args.steps,
            data_root=args.data_root,
            epoch=epoch,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    configure_logger("mae")
    main()
