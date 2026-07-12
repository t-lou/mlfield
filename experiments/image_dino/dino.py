import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from components.dataset.image_only_dataset import ImageOnlyDataset
from components.utils.config import load_yaml
from components.utils.device import get_device, resolve_num_workers
from components.utils.fps_logger import FpsLogger
from components.utils.logger import configure_logger, logger
from components.vit.dino_defs import DINOConfig
from components.vit.dino_session import DINOSession
from components.vit.dino_transform import DINOTransform, dino_collate_fn


def train(
    config: DINOConfig,
    num_epochs=100,
    start_epoch=-1,
    use_amp=True,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
):
    """Train the DINO model on ImageNet 256x256 dataset with multi-crop augmentation."""

    dir_ckpt = Path("./dino_checkpoints")
    if not dir_ckpt.exists():
        dir_ckpt.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dino_transform = DINOTransform(config)
    dataset = ImageOnlyDataset(
        root_dirs=config.data_dirs,
        transform=lambda x: dino_transform(x),
    )
    num_workers = resolve_num_workers(num_workers)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": dino_collate_fn,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = DataLoader(**loader_kwargs)

    dino_session = DINOSession(config, device=device)

    if start_epoch > 0 and (dir_ckpt / f"epoch_{start_epoch:03d}.pth").exists():
        dino_session.load(dir_ckpt / f"epoch_{start_epoch:03d}.pth")
        next_epoch = start_epoch + 1
        logger.info(f"Resuming training from epoch {start_epoch}; next epoch is {next_epoch}")
    else:
        logger.info(f"Starting training from scratch at epoch {start_epoch}")
        next_epoch = 0

    student = dino_session.student
    teacher = dino_session.teacher
    criterion = dino_session.loss_fn
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.04)

    if device.type == "cuda" and use_amp:
        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        amp_context = nullcontext()
        scaler = None

    logger.info(
        f"Training with {config.num_teachers} global crops, {config.num_students} local crops, "
        f"amp={'enabled' if scaler is not None else 'disabled'}"
    )

    stop_epoch = next_epoch + num_epochs
    fps_logger = FpsLogger(batch_size=config.batch_size)
    for epoch in range(next_epoch, stop_epoch):
        for imgs in loader:
            imgs = [img.to(device, non_blocking=True) for img in imgs]

            with amp_context:
                # Student sees all crops, teacher only sees global crops.
                student_outputs = [student(crop) for crop in imgs]

                with torch.inference_mode():
                    teacher_outputs = [teacher(imgs[i]) for i in range(config.num_teachers)]

                loss = criterion(student_outputs, teacher_outputs, global_crop_indices=list(range(config.num_teachers)))

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            dino_session.update_teacher(momentum=0.996)
            fps_logger.tick()

        dino_session.save(dir_ckpt / f"epoch_{epoch:03d}.pth")
        logger.info(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    configure_logger("dino")

    argument_parser = argparse.ArgumentParser(description="DINO Training")
    argument_parser.add_argument(
        "--path-config", type=str, default="./experiments/image_dino/dino_config.yaml", help="Path for the configs"
    )
    argument_parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of epochs to run in this invocation",
    )
    argument_parser.add_argument("--start-epoch", type=int, default=-1, help="Starting epoch for training")
    argument_parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision training")
    argument_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Defaults to automatic CPU-count based selection.",
    )
    argument_parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of batches prefetched per worker when num_workers > 0",
    )
    argument_parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers",
    )
    args = argument_parser.parse_args()

    path_config = Path(args.path_config)
    config = load_yaml(path_config, DINOConfig)

    train(
        config,
        args.num_epochs,
        start_epoch=args.start_epoch,
        use_amp=not args.disable_amp,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.disable_persistent_workers,
    )
