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

# Replace these with your real image paths, or pass --images on CLI.
IMAGE_PATHS = [
    "./data/kaggle/coco/coco2017/val2017/000000043314.jpg",
    "./data/kaggle/coco/coco2017/val2017/000000052565.jpg",
    "./data/kaggle/coco/coco2017/val2017/000000060823.jpg",
    "./data/kaggle/coco/coco2017/val2017/000000074457.jpg",
]


def render_feature_grids(
    rgbs: List[np.ndarray],
    attn_maps: List[Optional[np.ndarray]],
    token_maps: List[Optional[np.ndarray]],
    output_path: Path,
) -> None:
    if not rgbs:
        raise RuntimeError("No valid images found. Provide existing paths via IMAGE_PATHS or --images.")

    n = len(rgbs)

    # Figure 1: Attention map overlay grid.
    fig1, axes1 = plt.subplots(3, n, figsize=(4 * n, 4 * 3), squeeze=False)
    for i in range(n):
        ax = axes1[0, i]
        ax.imshow(rgbs[i])
        ax.axis("off")

        if attn_maps[i] is not None:
            ax = axes1[1, i]
            ax.imshow(attn_maps[i], cmap="magma")
        ax.axis("off")

        ax = axes1[2, i]
        if token_maps[i] is not None:
            ax.imshow(token_maps[i], cmap="viridis")
        ax.axis("off")
    fig1.tight_layout()
    fig1.savefig(output_path, dpi=180)
    plt.close(fig1)

    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DINO attention and patch-token feature maps")
    parser.add_argument(
        "--path-config", type=str, default="./experiments/image_dino/dino_config.yaml", help="Path for the configs"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to DINO checkpoint (.pth)")
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        default=None,
        help="Image paths. If omitted, uses IMAGE_PATHS in this script.",
    )
    parser.add_argument("--output-path", type=str, default="./feature_viz.png", help="Path to save the figure")
    parser.add_argument("--image-size", type=int, default=None, help="Override model input size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    ckpt_path = Path(args.ckpt)
    output_path = Path(args.output_path)
    image_candidates = args.images if args.images is not None and len(args.images) > 0 else IMAGE_PATHS
    image_paths = [Path(p) for p in image_candidates if p.exists()]

    config = load_yaml(Path(args.path_config), DINOConfig)
    student = load_student_from_checkpoint(config=config, ckpt_path=ckpt_path, device=device)
    image_size = args.image_size if args.image_size is not None else config.model_base_res

    rgbs, attn_maps, token_maps = get_features(
        student=student,
        image_paths=image_paths,
        image_size=image_size,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_feature_grids(rgbs, attn_maps, token_maps, output_path)


if __name__ == "__main__":
    main()
