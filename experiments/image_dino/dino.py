import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from components.dataset.image_only_dataset import ImageOnlyDataset
from components.utils.config import load_yaml
from components.utils.fps_logger import FpsLogger
from components.utils.logger import configure_logger, logger
from components.utils.onnx_export import export_onnx
from components.vit.dino_defs import DINOConfig
from components.vit.dino_session import DINOSession
from components.vit.dino_transform import DINOTransform, dino_collate_fn
from torch.utils.data import DataLoader


def _build_loader(config: DINOConfig, device: torch.device) -> DataLoader:
    dino_transform = DINOTransform(config)
    dataset = ImageOnlyDataset(
        root_dirs=config.data_dirs,
        transform=dino_transform,  # picklable, unlike a lambda (needed for spawn-based workers)
    )
    num_workers = config.train_config.num_workers
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": config.train_config.batch_size,
        "shuffle": config.train_config.shuffle,
        "num_workers": num_workers,
        "collate_fn": dino_collate_fn,
        "pin_memory": (device.type == "cuda"),
        "drop_last": True,  # equal batch sizes keep the teacher center statistics comparable
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.train_config.prefetch_factor
        loader_kwargs["persistent_workers"] = config.train_config.persistent_workers
    return DataLoader(**loader_kwargs)


def _build_optimizer(student: torch.nn.Module, lr: float = 1e-4, weight_decay: float = 0.04) -> torch.optim.Optimizer:
    """AdamW over the trainable parameters; no weight decay for biases, norms and tokens."""
    decay, no_decay = [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:  # e.g. the fixed weight-norm gain of the last layer
            continue
        (no_decay if p.ndim < 2 or "token" in name else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def _build_amp(device: torch.device, use_amp: bool, amp_dtype: str):
    """Return (autocast context, GradScaler or None). The scaler is only needed for float16."""
    if device.type != "cuda" or not use_amp:
        return nullcontext(), None
    if amp_dtype == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), None
    return torch.autocast(device_type="cuda", dtype=torch.float16), torch.amp.GradScaler("cuda", enabled=True)


def _train_one_epoch(
    dino_session: DINOSession,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_context,
    scaler: torch.amp.GradScaler | None,
    num_teachers: int,
    ema_momentum: float,
    fps_logger: FpsLogger,
) -> float:
    """Run one epoch and return the mean loss (a single host sync at the end)."""
    student = dino_session.student
    teacher = dino_session.teacher
    criterion = dino_session.loss_fn
    student.train()

    loss_sum = torch.zeros((), device=device)
    num_steps = 0

    for imgs in loader:
        imgs = [img.to(device, non_blocking=True) for img in imgs]

        with amp_context:
            # Student sees all crops, teacher only sees global crops.
            # Crops of equal size share one forward pass (2 global + 8 local crops -> 2 passes).
            student_outputs = student.forward_crops(imgs)

            with torch.inference_mode():
                teacher_outputs = teacher.forward_crops(imgs[:num_teachers])

            loss = criterion(student_outputs, teacher_outputs, global_crop_indices=list(range(num_teachers)))

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        dino_session.update_teacher(momentum=ema_momentum)
        fps_logger.tick()

        loss_sum += loss.detach().float()
        num_steps += 1

    if num_steps == 0:
        raise RuntimeError("The data loader produced no batches (dataset smaller than batch_size?)")
    return (loss_sum / num_steps).item()


def train(
    config: DINOConfig,
    start_epoch: int = -1,
    use_amp: bool = True,
    amp_dtype: str = "float16",
    ema_momentum: float = 0.996,
):
    """Train the DINO model with multi-crop augmentation.

    Args:
        amp_dtype: "float16" (with GradScaler) or "bfloat16" (no scaler needed) for CUDA autocast.
    """

    dir_ckpt = Path(config.train_config.dir_ckpts)
    dir_ckpt.mkdir(parents=True, exist_ok=True)

    device = config.train_config.get_device()
    loader = _build_loader(config, device)

    dino_session = DINOSession(config, device=device)
    optimizer = _build_optimizer(dino_session.student)
    amp_context, scaler = _build_amp(device, use_amp, amp_dtype)

    if start_epoch >= 0 and (dir_ckpt / f"epoch_{start_epoch:03d}.pth").exists():
        ckpt = dino_session.load(dir_ckpt / f"epoch_{start_epoch:03d}.pth")
        # optimizer / scaler state only exists in checkpoints written by this trainer
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                logger.warning("Optimizer state does not match the current optimizer; starting it from scratch")
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        next_epoch = start_epoch + 1
        logger.info(f"Resuming training from epoch {start_epoch}; next epoch is {next_epoch}")
    else:
        logger.info("Starting training from scratch at epoch 0")
        next_epoch = 0

    amp_info = "disabled" if isinstance(amp_context, nullcontext) else f"enabled ({amp_dtype})"
    logger.info(
        f"Backbone ViT-{config.variant}/{config.patch_size}, "
        f"training with {config.num_teachers} global crops, {config.num_students} local crops, amp={amp_info}"
    )

    stop_epoch = next_epoch + config.train_config.num_epoch
    fps_logger = FpsLogger(batch_size=config.train_config.batch_size)
    for epoch in range(next_epoch, stop_epoch):
        mean_loss = _train_one_epoch(
            dino_session, loader, optimizer, device, amp_context, scaler, config.num_teachers, ema_momentum, fps_logger
        )

        extra = {"optimizer": optimizer.state_dict()}
        if scaler is not None:
            extra["scaler"] = scaler.state_dict()
        dino_session.save(dir_ckpt / f"epoch_{epoch:03d}.pth", extra=extra)
        logger.info(f"Epoch {epoch:03d}, Loss: {mean_loss:.4f}")

    onnx_path = export_onnx(
        dino_session.teacher.backbone,
        (torch.zeros(1, 3, config.teacher_base_res, config.teacher_base_res, device=device),),
        dir_ckpt / "image_dino_teacher_backbone.onnx",
        input_names=("images",),
        output_names=("cls_features",),
    )
    logger.info("Exported DINO teacher backbone to %s", onnx_path)


if __name__ == "__main__":
    configure_logger("dino")

    argument_parser = argparse.ArgumentParser(description="DINO Training")
    argument_parser.add_argument(
        "--path-config", type=str, default="./experiments/image_dino/dino_config.yaml", help="Path for the configs"
    )
    argument_parser.add_argument("--start-epoch", type=int, default=-1, help="Starting epoch for training")
    argument_parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision training")
    argument_parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Autocast dtype on CUDA (bfloat16 needs no loss scaling and is more robust for large ViTs)",
    )
    argument_parser.add_argument("--ema-momentum", type=float, default=0.996, help="EMA momentum for the teacher")
    args = argument_parser.parse_args()

    path_config = Path(args.path_config)
    config = load_yaml(path_config, DINOConfig)

    train(
        config,
        start_epoch=args.start_epoch,
        use_amp=not args.disable_amp,
        amp_dtype=args.amp_dtype,
        ema_momentum=args.ema_momentum,
    )
