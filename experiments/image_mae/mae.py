"""
Masked Autoencoder (MAE) Pre-training Implementation

This module implements a self-supervised Masked Autoencoder for vision tasks,
based on the MAE paper (He et al., 2021). MAE learns rich visual representations
by randomly masking image patches and reconstructing them.

Key Components:
- PatchEmbed: Converts images to patch embeddings
- VitEncoder / VitBlock: Vision Transformer blocks for encoder/decoder
- MAE: Main autoencoder model with asymmetric encoder-decoder (sizes T/S/M/B/L/XL)
- Dataset loaders for CIFAR-10, ImageNet, and COCO

Training features: AdamW with weight decay only on matrices, optional cosine schedule with warmup,
optional gradient clipping, fp16 (with GradScaler) or bf16 autocast, resumable checkpoints
(model weights in ``checkpoint_XXXXX.pth``, optimizer / scaler state next to it).

Improvement Opportunities:
1. Add support for distributed training (DistributedDataParallel)
2. Implement gradient checkpointing for memory efficiency
3. Add tensorboard logging for training visualization
4. Add model ensemble support for inference
"""

import argparse
import math
import re
from contextlib import nullcontext
from pathlib import Path

import torch
from components.dataset.image_only_dataset import ImageOnlyDataset
from components.definitions.train_config import TrainConfig
from components.utils.config import load_yaml
from components.utils.fps_logger import FpsLogger
from components.utils.logger import configure_logger, logger
from components.utils.onnx_export import export_onnx
from components.vit.mae import MAE
from components.vit.mae_defs import MAE_MINI_CONFIG, MAEConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class _EncoderExport(torch.nn.Module):
    def __init__(self, model: MAE) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_encoder_full(images)


def make_dataloader_args(train_config: TrainConfig, pin_memory: bool | None = None):
    """DataLoader keyword arguments; ``pin_memory`` defaults to "only if CUDA is available"."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": train_config.batch_size,
        "shuffle": train_config.shuffle,
        "drop_last": True,
        "pin_memory": pin_memory,
    }
    if train_config.num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": train_config.num_workers,
                "prefetch_factor": train_config.prefetch_factor,
                "persistent_workers": train_config.persistent_workers,
            }
        )
    else:
        loader_kwargs["num_workers"] = 0

    return loader_kwargs


def make_cifar10_dataloader(root: str, train_config: TrainConfig) -> DataLoader:
    """
    Create CIFAR-10 dataloader for MAE pre-training.

    CIFAR-10 is a 32x32 image dataset with 50K training samples. Good for
    quick experiments and testing on small GPUs (~4GB).

    Args:
        root: Root directory to store/load CIFAR-10 dataset
        train_config: Training configuration (batch size, num workers, etc.)

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

    loader_kwargs = make_dataloader_args(train_config=train_config)

    return DataLoader(dataset, **loader_kwargs)


def _make_image_transform(image_size: int, dataset_type: str):
    if dataset_type == "classification":
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    if dataset_type == "detection":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def make_dataloader_from_dataset_specs(
    dataset_specs: list[dict[str, str]],
    train_config: TrainConfig,
    image_size: int = 224,
) -> DataLoader:
    """Create a concatenated dataloader from multiple dataset specs with their own transforms."""
    parts = []
    for spec in dataset_specs:
        dataset_path = spec["path"]
        dataset_type = spec["type"]
        if not dataset_path or not Path(dataset_path).exists():
            logger.warning(f"Dataset path {dataset_path!r} does not exist, skipping it")
            continue
        transform = _make_image_transform(image_size=image_size, dataset_type=dataset_type)
        parts.append(ImageOnlyDataset([dataset_path], transform=transform))

    if not parts:
        raise ValueError("No usable dataset specs: none were provided or none of their paths exist")

    combined_dataset = torch.utils.data.ConcatDataset(parts)
    loader_kwargs = make_dataloader_args(train_config=train_config)
    return DataLoader(combined_dataset, **loader_kwargs)


def make_dataloader_from_classification_dataset(
    data_dirs: list[str],
    train_config: TrainConfig,
    image_size: int = 224,
) -> DataLoader:
    """
    Create dataloader which are designed for classification.

    Images for classification tend to be similar in size and scale, and needs less cropping.

    Args:
        data_dirs: List of directories containing datasets
        train_config: Training configuration (batch size, num workers, etc.)
        image_size: Size to resize and crop images to

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

    loader_kwargs = make_dataloader_args(train_config=train_config)

    return DataLoader(dataset, **loader_kwargs)


def make_dataloader_from_detection_dataset(
    data_dirs: list[str],
    train_config: TrainConfig,
    image_size: int = 224,
) -> DataLoader:
    """
    Create dataloader which are designed for detection.

    Detection dataset contains more objects and requires more cropping.

    Args:
        data_dirs: List of directories containing datasets.
        train_config: Training configuration (batch size, num workers, etc.)
        image_size: Size to resize and crop images to

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

    loader_kwargs = make_dataloader_args(train_config=train_config)

    return DataLoader(dataset, **loader_kwargs)


def get_checkpoint_path(epoch: int, dir_ckpts: str | Path | None = None) -> Path:
    checkpoint_dir = Path(dir_ckpts or "mae_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"checkpoint_{epoch:05d}.pth"


def get_optimizer_state_path(checkpoint_path: str | Path) -> Path:
    """Where the optimizer / scaler state belongs to a model checkpoint (``checkpoint_00007.optim.pth``)."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(checkpoint_path.stem + ".optim.pth")


def find_latest_epoch(dir_ckpts: str | Path | None = None) -> int:
    """Highest epoch with a ``checkpoint_XXXXX.pth`` in the directory, or -1 if there is none."""
    checkpoint_dir = Path(dir_ckpts or "mae_checkpoints")
    epochs = [
        int(m.group(1))
        for p in checkpoint_dir.glob("checkpoint_*.pth")
        if (m := re.fullmatch(r"checkpoint_(\d+)\.pth", p.name))
    ]
    return max(epochs, default=-1)


def build_optimizer(model: MAE, config: MAEConfig) -> torch.optim.Optimizer:
    """AdamW (betas 0.9 / 0.95); no weight decay for biases, norms and the CLS / mask tokens."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "token" in name else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95))


def lr_at_step(step: int, total_steps: int, warmup_steps: int, config: MAEConfig, warmup: bool = True) -> float:
    """Learning rate for a (0-based) optimizer step; constant unless ``config.lr_schedule == "cosine"``."""
    base = config.learning_rate
    if config.lr_schedule == "constant":
        return base
    if warmup and step < warmup_steps:
        return base * (step + 1) / warmup_steps
    start = warmup_steps if warmup else 0
    progress = min(1.0, (step - start) / max(1, total_steps - start))
    return config.min_lr + (base - config.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_amp(device: torch.device, use_amp: bool, amp_dtype: str):
    """Return (autocast context, GradScaler or None). The scaler is only needed for float16."""
    if device.type != "cuda" or not use_amp:
        return nullcontext(), None
    if amp_dtype == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), None
    return torch.autocast(device_type="cuda", dtype=torch.float16), torch.amp.GradScaler("cuda", enabled=True)


def _save_training_state(
    checkpoint_path: Path, model: MAE, optimizer: torch.optim.Optimizer, scaler, epoch: int
) -> None:
    """Model weights (plain state dict, as before) plus optimizer / scaler state next to it."""
    model.save_checkpoint(checkpoint_path)
    state = {"optimizer": optimizer.state_dict(), "epoch": epoch}
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    path = get_optimizer_state_path(checkpoint_path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _load_optimizer_state(checkpoint_path: Path, optimizer: torch.optim.Optimizer, scaler, device) -> bool:
    path = get_optimizer_state_path(checkpoint_path)
    if not path.exists():
        logger.warning(f"No optimizer state {path}; the optimizer starts from scratch")
        return False
    state = torch.load(path, map_location=device, weights_only=True)
    try:
        optimizer.load_state_dict(state["optimizer"])
    except ValueError:
        logger.warning(f"Optimizer state in {path} does not match the current optimizer; starting it from scratch")
        return False
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return True


def train(
    model: MAE,
    config: MAEConfig,
    steps: int = -1,
    data_root: str = "./data",
    start_epoch: int = 0,
    mini: bool = False,
    use_amp: bool = True,
    amp_dtype: str = "float16",
) -> None:
    """
    Train MAE model for a sequence of epochs with checkpointing.

    Args:
        model: MAE model already initialized and optionally loaded from checkpoint.
        config: Model config (model size, dataset type etc)
        steps: Max steps per epoch (-1 = full epoch)
        data_root: Root directory for datasets
        start_epoch: Starting epoch number for checkpointing and logs. If it is > 0 the optimizer
            state saved with the checkpoint of epoch ``start_epoch - 1`` is restored as well.
        mini: If True, use CIFAR-10 data loader and smaller model config
        use_amp: Mixed precision on CUDA
        amp_dtype: "float16" (with GradScaler) or "bfloat16" (no scaler needed)

    ``config.train_config.num_epoch`` epochs are run. With ``lr_schedule="cosine"`` the schedule
    spans these epochs (linear warmup only when ``start_epoch == 0``).
    """
    device = config.train_config.get_device()
    model = model.to(device)

    if mini:
        loader = make_cifar10_dataloader(
            root="./data/cifar10",
            train_config=config.train_config,
        )
    else:
        dataset_specs = getattr(config, "datasets", None) or []
        if dataset_specs:
            loader = make_dataloader_from_dataset_specs(
                dataset_specs=list(dataset_specs),
                train_config=config.train_config,
                image_size=config.image_size,
            )
        else:
            loader = make_dataloader_from_classification_dataset(
                data_dirs=[data_root],
                train_config=config.train_config,
                image_size=config.image_size,
            )

    optimizer = build_optimizer(model, config)
    amp_context, scaler = _build_amp(device, use_amp, amp_dtype)

    if start_epoch > 0:
        previous = get_checkpoint_path(start_epoch - 1, dir_ckpts=config.train_config.dir_ckpts)
        _load_optimizer_state(previous, optimizer, scaler, device)

    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = max(steps, 1)
    if steps >= 0:
        steps_per_epoch = min(steps_per_epoch, steps)
    total_steps = max(1, steps_per_epoch * config.train_config.num_epoch)
    warmup_steps = int(config.warmup_epochs * steps_per_epoch) if start_epoch == 0 else 0
    fps_logger = FpsLogger(batch_size=config.train_config.batch_size)

    logger.info(
        f"MAE ViT-{config.variant}/{config.patch_size}, mask_ratio={config.mask_ratio}, "
        f"norm_pix_loss={config.norm_pix_loss}, lr_schedule={config.lr_schedule}"
    )

    global_step = 0
    for epoch_offset in range(config.train_config.num_epoch):
        epoch = start_epoch + epoch_offset
        logger.info(f"Starting epoch {epoch}")

        model.train()
        loss_sum = torch.zeros((), device=device)
        num_steps = 0
        lr = config.learning_rate

        for step, batch in enumerate(loader, start=1):
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device, non_blocking=True)

            lr = lr_at_step(global_step, total_steps, warmup_steps, config, warmup=start_epoch == 0)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with amp_context:
                loss, pred, target, mask = model(imgs)

            if scaler is not None:
                scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            fps_logger.tick()
            loss_sum += loss.detach().float()  # no host sync inside the loop
            num_steps += 1
            global_step += 1

            if steps >= 0 and step >= steps:
                break

        if num_steps == 0:
            raise RuntimeError("The data loader produced no batches (dataset smaller than batch_size?)")

        logger.info(
            f"epoch={epoch} step={num_steps} loss={(loss_sum / num_steps).item():.6f} lr={lr:.3e} "
            f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)}"
        )
        checkpoint_path = get_checkpoint_path(epoch, dir_ckpts=config.train_config.dir_ckpts)
        _save_training_state(checkpoint_path, model, optimizer, scaler, epoch)
        logger.info(f"Saved checkpoint for epoch {epoch}: {checkpoint_path}")

    model.eval()
    onnx_path = export_onnx(
        _EncoderExport(model),
        (torch.zeros(1, config.in_chans, config.image_size, config.image_size, device=device),),
        Path(config.train_config.dir_ckpts) / "image_mae_encoder.onnx",
        input_names=("images",),
        output_names=("encoded_tokens",),
    )
    logger.info("Exported MAE encoder to %s", onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MAE trainer")
    parser.add_argument("--path-config", type=str, default="./experiments/image_mae/mae_config.yaml")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=-1,
        help="Resume from the checkpoint of this epoch (-1 = start from scratch).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint in the checkpoint directory (overrides --start-epoch).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for datasets when no mixed dataset config is provided.",
    )
    parser.add_argument("--mini", action="store_true", help="Whether to use ciphar10 dataset and smaller model.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Autocast dtype on CUDA (bfloat16 needs no loss scaling)",
    )
    args = parser.parse_args()

    if not args.mini:
        path_config = Path(args.path_config)
        config = load_yaml(path_config, MAEConfig)
    else:
        config = MAE_MINI_CONFIG

    model = MAE(config)
    resume_epoch = find_latest_epoch(config.train_config.dir_ckpts) if args.resume else args.start_epoch
    checkpoint_path = get_checkpoint_path(resume_epoch, dir_ckpts=config.train_config.dir_ckpts)
    if resume_epoch >= 0 and checkpoint_path.exists():
        model.load_checkpoint(checkpoint_path)
        start_epoch = resume_epoch + 1
        logger.info(f"Loaded checkpoint from {checkpoint_path}, will start with epoch {start_epoch}")
    else:
        start_epoch = 0
        logger.info(f"No checkpoint found, will start with epoch {start_epoch}")

    train(
        model=model,
        config=config,
        steps=args.steps,
        data_root=args.data_root,
        start_epoch=start_epoch,
        mini=args.mini,
        use_amp=not args.disable_amp,
        amp_dtype=args.amp_dtype,
    )


if __name__ == "__main__":
    configure_logger("mae")
    main()
