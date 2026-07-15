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
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from components.dataset.image_only_dataset import ImageOnlyDataset
from components.utils.config import load_yaml
from components.utils.device import get_device, resolve_num_workers
from components.utils.logger import configure_logger, logger
from components.vit.mae import MAE
from components.vit.mae_defs import MAE_MINI_CONFIG, MAEConfig

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

    return DataLoader(dataset, **loader_kwargs)


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

    return DataLoader(dataset, **loader_kwargs)


def get_checkpoint_path(epoch: int) -> Path:
    checkpoint_dir = Path("mae_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"checkpoint_{epoch:05d}.pth"


def train(
    model: MAE,
    config: MAEConfig,
    steps: int = -1,
    data_root: str = "./data",
    start_epoch: int = 0,
    num_epochs: int = 1,
    num_workers: Optional[int] = None,
    mini: bool = False,
) -> None:
    """
    Train MAE model for a sequence of epochs with checkpointing.

    Args:
        model: MAE model already initialized and optionally loaded from checkpoint.
        config: Model config (model size, dataset type etc)
        steps: Max steps per epoch (-1 = full epoch)
        data_root: Root directory for datasets
        start_epoch: Starting epoch number for checkpointing and logs
        num_epochs: Number of epochs to run
        num_workers: Number of DataLoader workers
        mini: If True, use CIFAR-10 data loader and smaller model config
    """
    device = get_device()
    model = model.to(device)

    if mini:
        loader = make_cifar10_dataloader(root=data_root, batch_size=config.batch_size, num_workers=num_workers)
    else:
        loader = make_dataloader_from_classification_dataset(
            data_dirs=[data_root],
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_workers=num_workers,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    model.train()
    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset
        logger.info(f"Starting epoch {epoch}")

        for step, batch in enumerate(loader, start=1):
            if isinstance(batch, (tuple, list)):
                imgs = batch[0]
            else:
                imgs = batch

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

        logger.info(
            f"epoch={epoch} step={step} loss={float(loss):.6f} "
            f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)}"
        )
        checkpoint_path = get_checkpoint_path(epoch)
        model.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint for epoch {epoch}: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MAE debug trainer with CIFAR-10 and ImageNet presets")
    parser.add_argument("--path-config", type=str, default="./experiments/image_mae/mae_config.yaml")
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
    parser.add_argument("--mini", action="store_true", help="Whether to use ciphar10 dataset and smaller model.")
    args = parser.parse_args()

    if not args.mini:
        path_config = Path(args.path_config)
        config = load_yaml(path_config, MAEConfig)
    else:
        config = MAE_MINI_CONFIG

    model = MAE(config)
    checkpoint_path = get_checkpoint_path(args.start_epoch)
    if checkpoint_path.exists():
        model.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.info("No checkpoint found. Starting from scratch.")

    train(
        model=model,
        config=config,
        steps=args.steps,
        data_root=args.data_root,
        start_epoch=args.start_epoch,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        mini=args.mini,
    )


if __name__ == "__main__":
    configure_logger("mae")
    main()
