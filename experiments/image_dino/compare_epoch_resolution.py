import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from components.utils.config import load_yaml
from components.utils.device import get_device
from components.vit.dino_defs import DINOConfig
from components.vit.dino_inf import load_student_from_checkpoint
from components.vit.dino_vis import get_features


def render_feature_grids(
    rgbs: List[np.ndarray],
    attn_maps: List[Optional[np.ndarray]],
    token_maps: List[Optional[np.ndarray]],
    output_prefix: Path,
) -> None:
    if not rgbs:
        raise RuntimeError("No valid images found. Provide existing paths via IMAGE_PATHS or --images.")

    n = len(rgbs)

    # Figure 1: Attention map overlay grid.
    fig1, axes1 = plt.subplots(3, n, figsize=(4 * n, 4 * 3), squeeze=False)
    for i in range(n):
        if attn_maps[i] is not None:
            ax = axes1[1, i]
            ax.imshow(attn_maps[i], cmap="magma")
        ax.axis("off")

        ax = axes1[2, i]
        if token_maps[i] is not None:
            ax.imshow(token_maps[i], cmap="viridis")
        ax.axis("off")
    fig1.tight_layout()
    fig1.savefig(output_prefix, dpi=180)
    plt.close(fig1)

    print(f"Saved: {output_prefix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DINO attention and patch-token feature maps")
    parser.add_argument(
        "--path-config", type=str, default="./experiments/image_dino/dino_config.yaml", help="Path for the configs"
    )
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Path of folder to DINO checkpoints (*.pth)")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image path, only one image is allowed.",
    )
    parser.add_argument("--output-prefix", type=str, default="./example", help="Partial path to save the figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    ckpt_dir = Path(args.ckpt_dir)
    assert ckpt_dir.exists() and ckpt_dir.is_dir(), f"Checkpoint directory does not exist: {ckpt_dir}"

    # Find path of all checkpoints and sort by name, assuming they are sorted by index
    ckpt_paths = sorted(ckpt_dir.glob("*.pth"))

    image = Path(args.image)
    assert image.exists()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    config = load_yaml(Path(args.path_config), DINOConfig)

    image_sizes = [128, 256, 512, 768]

    n_rows = len(image_sizes)
    n_cols = len(ckpt_paths)

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for col, ckpt_path in enumerate(ckpt_paths):
        student = load_student_from_checkpoint(config=config, ckpt_path=ckpt_path, device=device)

        for row, image_size in enumerate(image_sizes):
            _, attn_maps, token_maps = get_features(
                student=student,
                image_paths=[image],
                image_size=image_size,
                device=device,
            )
            assert len(attn_maps) == len(token_maps) == 1

            ax1 = axes1[row, col]
            ax1.imshow(attn_maps[0], cmap="magma")
            ax1.set_title(f"size={image_size}, ckpt={ckpt_path.stem}")
            ax1.axis("off")

            ax2 = axes2[row, col]
            ax2.imshow(token_maps[0], cmap="viridis")
            ax2.set_title(f"size={image_size}, ckpt={ckpt_path.stem}")
            ax2.axis("off")

    plt.figure(fig1.number)
    plt.tight_layout()

    plt.figure(fig2.number)
    plt.tight_layout()

    fig1.tight_layout()
    fig1.savefig(str(output_prefix) + "_attn_maps.png", dpi=180)
    fig2.tight_layout()
    fig2.savefig(str(output_prefix) + "_token_maps.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
