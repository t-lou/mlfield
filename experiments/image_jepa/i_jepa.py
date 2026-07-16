"""I-JEPA trainer."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from components.dataset.image_only_dataset import ImageOnlyDataset
from components.utils.config import load_yaml
from components.utils.device import get_device, resolve_num_workers
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
    pred_list = outputs["predicted_target_tokens"]
    tgt_list = outputs["target_tokens"]

    if len(pred_list) != len(tgt_list):
        raise ValueError("Mismatch between number of predicted and target blocks")

    if len(pred_list) == 0:
        raise ValueError("No target blocks produced by model")

    loss = torch.zeros((), device=pred_list[0].device)
    for pred, tgt in zip(pred_list, tgt_list):
        loss = loss + F.mse_loss(pred, tgt)
    return loss / len(pred_list)


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
    num_epochs: int = 100,
    start_epoch: int = -1,
    use_amp: bool = True,
    num_workers: int | None = 8,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    ema_momentum: float = 0.996,
) -> None:
    """Train I-JEPA with a DINO-like orchestration loop."""

    # SHARED FROM dino.py: checkpoint folder handling.
    dir_ckpt = Path("./i_jepa_checkpoints")
    dir_ckpt.mkdir(parents=True, exist_ok=True)

    # SHARED FROM dino.py: device selection and worker resolution.
    device = get_device()
    resolved_num_workers = resolve_num_workers(num_workers)

    # REPLACED FROM dino.py: no multi-crop collate, just one transformed image per sample.
    dataset = ImageOnlyDataset(
        root_dirs=list(config.data_dirs),
        transform=_build_transform(config.image_size),
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": resolved_num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": True,
    }
    if resolved_num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
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

    stop_epoch = next_epoch + num_epochs
    fps_logger = FpsLogger(batch_size=config.batch_size)

    for epoch in range(next_epoch, stop_epoch):
        model.train()
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)

            with amp_context:
                outputs = model(imgs)
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
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to run in this invocation")
    parser.add_argument("--start-epoch", type=int, default=-1, help="Starting epoch for training")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Defaults to automatic CPU-count based selection.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of batches prefetched per worker when num_workers > 0",
    )
    parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers",
    )
    parser.add_argument("--ema-momentum", type=float, default=0.996, help="EMA momentum for target encoder")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.path_config), IJEPAConfig)
    train(
        cfg,
        num_epochs=args.num_epochs,
        start_epoch=args.start_epoch,
        use_amp=not args.disable_amp,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.disable_persistent_workers,
        ema_momentum=args.ema_momentum,
    )
