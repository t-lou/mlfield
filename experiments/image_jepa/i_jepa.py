"""I-JEPA trainer."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from components.dataset.image_only_dataset import ImageOnlyDataset
from components.dataset.mask_prefetch import collate_fn_with_masks
from components.utils.config import load_yaml
from components.utils.fps_logger import FpsLogger
from components.utils.logger import configure_logger, logger
from components.vit.i_jepa import I_JEPA
from components.vit.i_jepa_defs import IJEPAConfig


def _build_transform(image_size: int):
    # Reuse ImageNet-style normalization as in other vision trainers.
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _compute_i_jepa_loss(outputs: dict[str, object]) -> torch.Tensor:
    """Compute I-JEPA loss using fully vectorized masked operations.

    Critical optimization: Keep all computations on GPU without CPU-GPU syncs.
    Never convert tensor values to Python scalars (no .item(), no += with Python ints).
    """
    pred_list = outputs["predicted_target_tokens"]
    tgt_list = outputs["target_tokens"]
    valid_list = outputs["target_valid_masks"]

    if len(pred_list) != len(tgt_list) or len(pred_list) != len(valid_list):
        raise ValueError("Mismatch between number of predicted blocks, target blocks, and validity masks")

    if len(pred_list) == 0:
        raise ValueError("No target blocks produced by model")

    device = pred_list[0].device
    total_loss = torch.tensor(0.0, device=device)
    total_valid_count = torch.tensor(0.0, device=device)

    for pred, tgt, valid in zip(pred_list, tgt_list, valid_list):
        # Compute MSE loss on full tensors without reshaping
        # Shape: pred (B, T, D), tgt (B, T, D), valid (B, T)
        loss_per_token = F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)  # (B, T)

        # Apply mask via element-wise multiplication (fully parallelized)
        # No advanced indexing - pure CUDA operations
        loss_masked = loss_per_token * valid.float()  # (B, T)

        # Accumulate loss and count (all GPU tensors, no conversion to Python)
        total_loss = total_loss + loss_masked.sum()
        total_valid_count = total_valid_count + valid.sum().float()

    # Safe division on GPU (no Python scalars involved)
    return total_loss / torch.clamp(total_valid_count, min=1.0)


def _save_checkpoint(path: Path, model: I_JEPA, optimizer: torch.optim.Optimizer) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def _load_checkpoint(path: Path, model: I_JEPA, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    state = torch.load(path, map_location=device)

    if "model" not in state or "optimizer" not in state:
        raise KeyError(f"Checkpoint {path} must contain 'model' and 'optimizer' entries")

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


def train(
    config: IJEPAConfig,
    start_epoch: int = -1,
    use_amp: bool = True,
    ema_momentum: float = 0.996,
    use_mask_prefetch: bool = False,
) -> None:
    """Train I-JEPA with a DINO-like orchestration loop.

    Args:
        use_mask_prefetch: If True, generate masks in DataLoader prefetch (experimental).
                          Can improve GPU utilization if mask sampling is bottleneck.
    """

    # SHARED FROM dino.py: checkpoint folder handling.
    dir_ckpt = Path(config.train_config.dir_ckpts)
    dir_ckpt.mkdir(parents=True, exist_ok=True)

    # SHARED FROM dino.py: device selection and worker resolution.
    device = config.train_config.get_device()

    # REPLACED FROM dino.py: no multi-crop collate, just one transformed image per sample.
    dataset = ImageOnlyDataset(
        root_dirs=list(config.data_dirs),
        transform=_build_transform(config.image_size),
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": config.train_config.batch_size,
        "shuffle": config.train_config.shuffle,
        "num_workers": config.train_config.num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": True,
    }

    # Use custom collate function if mask prefetching is enabled
    if use_mask_prefetch:
        loader_kwargs["collate_fn"] = lambda batch: collate_fn_with_masks(
            batch, config, device=torch.device("cpu")
        )  # cuda doesn't seem to be a good idea in dataloader

    if config.train_config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.train_config.prefetch_factor
        loader_kwargs["persistent_workers"] = config.train_config.persistent_workers
    loader = DataLoader(**loader_kwargs)

    model = I_JEPA(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)

    if start_epoch >= 0:
        ckpt_path = dir_ckpt / f"epoch_{start_epoch:03d}.pth"
        if ckpt_path.exists():
            _load_checkpoint(ckpt_path, model, optimizer, device)
            next_epoch = start_epoch + 1
            logger.info(f"Resuming training from epoch {start_epoch}; next epoch is {next_epoch}")
        else:
            logger.info(f"No checkpoint found at {ckpt_path}; starting from scratch at epoch 0")
            next_epoch = 0
    else:
        logger.info("Starting training from scratch at epoch 0")
        next_epoch = 0

    # SHARED FROM dino.py: AMP setup.
    if device.type == "cuda" and use_amp:
        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        amp_context = nullcontext()
        scaler = None

    stop_epoch = next_epoch + config.train_config.num_epoch
    fps_logger = FpsLogger(batch_size=config.train_config.batch_size)

    logger.info(f"Mask prefetching: {'ENABLED' if use_mask_prefetch else 'DISABLED'}")

    for epoch in range(next_epoch, stop_epoch):
        model.train()
        for batch_data in loader:
            # Handle both prefetched masks and on-the-fly sampling
            if use_mask_prefetch:
                imgs, context_mask_cpu, target_masks_cpu = batch_data
                # Move masks to GPU
                context_mask = context_mask_cpu.to(device)
                target_masks = [m.to(device) for m in target_masks_cpu]
            else:
                imgs = batch_data
                context_mask = None
                target_masks = None

            imgs = imgs.to(device, non_blocking=True)

            with amp_context:
                outputs = model(imgs, context_mask=context_mask, target_masks=target_masks)
                loss = _compute_i_jepa_loss(outputs)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # SHARED FROM dino.py: update target network via EMA after optimizer step.
            model.momentum_update_target_encoder(momentum=ema_momentum)
            fps_logger.tick()

        # SHARED FROM dino.py: epoch-level checkpointing and logging.
        _save_checkpoint(dir_ckpt / f"epoch_{epoch:03d}.pth", model, optimizer)
        logger.info(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    configure_logger("i_jepa")

    parser = argparse.ArgumentParser(description="I-JEPA training (patched from proposal scaffold)")
    parser.add_argument(
        "--path-config",
        type=str,
        default="./experiments/image_jepa/i_jepa_config.yaml",
        help="Path to I-JEPA config YAML",
    )
    parser.add_argument("--start-epoch", type=int, default=-1, help="Starting epoch for training")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--ema-momentum", type=float, default=0.996, help="EMA momentum for target encoder")
    parser.add_argument(
        "--use-mask-prefetch",
        action="store_true",
        help="Enable mask prefetching via DataLoader (experimental, may improve GPU utilization)",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.path_config), IJEPAConfig)
    train(
        cfg,
        start_epoch=args.start_epoch,
        use_amp=not args.disable_amp,
        ema_momentum=args.ema_momentum,
        use_mask_prefetch=args.use_mask_prefetch,
    )
