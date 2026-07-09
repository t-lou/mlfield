import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from components.data.image_only_dataset import ImageOnlyDataset
from components.utils.device import get_device, resolve_num_workers
from components.utils.logger import configure_logger, logger
from components.vit.dino_defs import STUDENT_BASE_RES, TEACHER_BASE_RES
from components.vit.dino_session import DINOSession

# Try to reuse the MAE dataset.
DEFAULG_DATA_ROOT_DIR = "./data/kaggle/imagenet/"


# Global and local crop transformations for DINO
GLOBAL_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(TEACHER_BASE_RES, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
)

LOCAL_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(STUDENT_BASE_RES, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
)


def multicrop_augment(img, num_global_crops=2, num_local_crops=8):
    """Generate multiple crops of the input image for DINO training."""
    crops = []
    for _ in range(num_global_crops):
        crops.append(GLOBAL_TRANSFORM(img))
    for _ in range(num_local_crops):
        crops.append(LOCAL_TRANSFORM(img))
    return crops


def dino_collate_fn(batch):
    """Collate a batch of multi-crop samples into a list of crop tensors."""
    if not batch:
        return []
    num_crops = len(batch[0])
    if any(len(sample) != num_crops for sample in batch):
        raise ValueError("Inconsistent number of crops in batch")
    return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(num_crops)]


def train(
    num_epochs=100,
    bs=8,
    data_root=DEFAULG_DATA_ROOT_DIR,
    start_epoch=-1,
    num_global_crops=2,
    num_local_crops=8,
    use_amp=True,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
):
    """Train the DINO model on ImageNet 256x256 dataset with multi-crop augmentation."""
    if not Path(data_root).exists():
        raise ValueError(f"Data root directory {data_root} does not exist.")

    dir_ckpt = Path("./dino_checkpoints")
    if not dir_ckpt.exists():
        dir_ckpt.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dataset = ImageOnlyDataset(
        root_dir=data_root,
        transform=lambda x: multicrop_augment(x, num_global_crops, num_local_crops),
    )
    num_workers = resolve_num_workers(num_workers)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": bs,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": dino_collate_fn,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = DataLoader(**loader_kwargs)

    dino_session = DINOSession(device=device)

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
        f"Training with {num_global_crops} global crops, {num_local_crops} local crops, "
        f"amp={'enabled' if scaler is not None else 'disabled'}"
    )

    stop_epoch = next_epoch + num_epochs
    for epoch in range(next_epoch, stop_epoch):
        for imgs in loader:
            imgs = [img.to(device, non_blocking=True) for img in imgs]

            with amp_context:
                # Student sees all crops, teacher only sees global crops.
                student_outputs = [student(crop) for crop in imgs]

                with torch.inference_mode():
                    teacher_outputs = [teacher(imgs[i]) for i in range(num_global_crops)]

                loss = criterion(student_outputs, teacher_outputs, global_crop_indices=list(range(num_global_crops)))

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            dino_session.update_teacher(momentum=0.996)

        dino_session.save(dir_ckpt / f"epoch_{epoch:03d}.pth")
        logger.info(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    configure_logger("dino")

    argument_parser = argparse.ArgumentParser(description="DINO Training")
    argument_parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of epochs to run in this invocation",
    )
    argument_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    argument_parser.add_argument("--start-epoch", type=int, default=-1, help="Starting epoch for training")
    argument_parser.add_argument(
        "--num-global-crops",
        type=int,
        default=2,
        help="Number of global crops to generate per image",
    )
    argument_parser.add_argument(
        "--num-local-crops",
        type=int,
        default=8,
        help="Number of local crops to generate per image",
    )
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
    train(
        args.num_epochs,
        args.batch_size,
        start_epoch=args.start_epoch,
        num_global_crops=args.num_global_crops,
        num_local_crops=args.num_local_crops,
        use_amp=not args.disable_amp,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.disable_persistent_workers,
    )
