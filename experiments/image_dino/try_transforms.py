import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from components.dataset.image_only_dataset import ImageOnlyDataset
from components.utils.config import load_yaml
from components.vit.dino_defs import DINOConfig
from components.vit.dino_transform import DINOTransform, dino_collate_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a DINO checkpoint and extract a feature for one image")
    parser.add_argument("--config", default="experiments/image_dino/dino_config.yaml", help="Path to DINO config yaml")
    parser.add_argument("--image-dir", default="data/lenna/", help="Path to input images (with one image)")
    parser.add_argument("--output-prefix", default="data/dino_transform", help="Prefix for the output image")
    args = parser.parse_args()

    config_path = Path(args.config if args.config else "experiments/image_dino/dino_config.yaml")
    config = load_yaml(config_path, DINOConfig)

    device = torch.device("cpu")
    dino_transform = DINOTransform(config)
    dataset = ImageOnlyDataset(
        root_dirs=[args.image_dir],
        transform=lambda x: dino_transform(x),
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 1,
        "collate_fn": dino_collate_fn,
        "pin_memory": (device.type == "cuda"),
    }
    loader = DataLoader(**loader_kwargs)

    for imgs in loader:
        imgs = [img.to(device, non_blocking=True) for img in imgs]
        assert len(imgs) == (config.num_teachers + config.num_students)
        input_teacher = imgs[: config.num_teachers]
        input_student = imgs[config.num_teachers :]

        # plot input for teacher
        fig, axes = plt.subplots(1, config.num_teachers, figsize=(4 * config.num_teachers, 4 * 1), squeeze=False)
        for i, data in enumerate(input_teacher):
            data = data.cpu().numpy()[0, ...]
            data = data.transpose(1, 2, 0)
            ax = axes[0, i]
            ax.imshow(data)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.output_prefix + "_in_teacher.png")
        plt.close(fig)

        # plot input for students
        row = 2
        col = int(math.ceil(config.num_students / row))
        fig, axes = plt.subplots(row, col, figsize=(4 * col, 4 * row), squeeze=False)
        for i, data in enumerate(input_student):
            data = data.cpu().numpy()[0, ...]
            data = data.transpose(1, 2, 0)
            ax = axes[i // col, i % col]
            ax.imshow(data)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.output_prefix + "_in_student.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
