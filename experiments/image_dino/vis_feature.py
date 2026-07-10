import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from components.utils.config import load_yaml
from components.utils.device import get_device
from components.vit.dino_defs import DINOConfig
from components.vit.dino_session import DINOSession

# Replace these with your real image paths, or pass --images on CLI.
IMAGE_PATHS = [
    "./data/kaggle/coco/coco2017/val2017/000000043314.jpg",
    "./data/kaggle/coco/coco2017/val2017/000000052565.jpg",
    "./data/kaggle/coco/coco2017/val2017/000000060823.jpg",
    "./data/kaggle/coco/coco2017/val2017/000000074457.jpg",
]


def preprocess_image(path: Path, image_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and resize image to the DINO encoder input size."""
    image = Image.open(path).convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    tensor = tfm(image)
    rgb = np.array(image.resize((image_size, image_size)))
    return tensor, rgb


def _reshape_token_vector_to_grid(vec: torch.Tensor) -> Optional[torch.Tensor]:
    """Reshape [num_tokens] to [h, w] if token count is a square."""
    n = vec.numel()
    side = int(n**0.5)
    if side * side != n:
        return None
    return vec.reshape(side, side)


def extract_maps_from_activations(
    attn_tensor: Optional[torch.Tensor],
    token_tensor: Optional[torch.Tensor],
    out_h: int,
    out_w: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build attention and token-energy heatmaps from captured tensors."""
    attn_map = None
    token_map = None

    if attn_tensor is not None:
        # Accept common shapes: [B, H, N, N] or [B, N, N].
        a = attn_tensor.detach().float().cpu()
        if a.ndim == 4:
            # Average heads, use CLS-to-patch attention.
            cls_to_all = a[0, :, 0, :].mean(dim=0)
        elif a.ndim == 3:
            cls_to_all = a[0, 0, :]
        else:
            cls_to_all = None

        if cls_to_all is not None and cls_to_all.numel() > 1:
            cls_to_patches = cls_to_all[1:]  # drop CLS->CLS
            grid = _reshape_token_vector_to_grid(cls_to_patches)
            if grid is not None:
                grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, gh, gw]
                grid = F.interpolate(grid, size=(out_h, out_w), mode="bilinear", align_corners=False)
                g = grid[0, 0]
                g = (g - g.min()) / (g.max() - g.min() + 1e-8)
                attn_map = g.numpy()

    if token_tensor is not None:
        # Expect [B, N, C], optionally with CLS token.
        t = token_tensor.detach().float().cpu()
        if t.ndim == 3:
            tokens = t[0]  # [N, C]
            if tokens.shape[0] > 1:
                no_cls = tokens[1:]
                side = int(no_cls.shape[0] ** 0.5)
                if side * side == no_cls.shape[0]:
                    patch_strength = no_cls.norm(dim=-1)
                else:
                    side = int(tokens.shape[0] ** 0.5)
                    patch_strength = tokens.norm(dim=-1)
            else:
                patch_strength = tokens.norm(dim=-1)

            grid = _reshape_token_vector_to_grid(patch_strength)
            if grid is not None:
                grid = grid.unsqueeze(0).unsqueeze(0)
                grid = F.interpolate(grid, size=(out_h, out_w), mode="bilinear", align_corners=False)
                g = grid[0, 0]
                g = (g - g.min()) / (g.max() - g.min() + 1e-8)
                token_map = g.numpy()

    return attn_map, token_map


class ActivationCatcher:
    """Capture one representative attention tensor and one token tensor per forward pass."""

    def __init__(self):
        self.last_attn: Optional[torch.Tensor] = None
        self.last_tokens: Optional[torch.Tensor] = None

    def make_hook(self, module_name: str):
        def _hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return

            # Attention-like tensors usually 3D/4D and module names include "attn".
            if "attn" in module_name.lower() and out.ndim in (3, 4):
                self.last_attn = out

            # Token tensors are typically [B, N, C]. Keep last seen.
            if out.ndim == 3:
                self.last_tokens = out

        return _hook


@torch.no_grad()
def render_feature_grids(
    student: torch.nn.Module,
    image_paths: List[Path],
    output_path: Path,
    image_size: int,
    device: torch.device,
) -> None:
    """Create and save two grids: attention maps and patch-token feature maps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catcher = ActivationCatcher()
    hooks = []
    for name, module in student.backbone.vit.named_modules():
        hooks.append(module.register_forward_hook(catcher.make_hook(name)))

    rgbs = []
    attn_maps = []
    token_maps = []
    valid_paths = []

    for p in image_paths:
        if not p.exists():
            print(f"[skip] missing image: {p}")
            continue

        x, rgb = preprocess_image(p, image_size)
        x = x.unsqueeze(0).to(device)

        catcher.last_attn = None
        catcher.last_tokens = None
        _ = student(x)

        attn_map, token_map = extract_maps_from_activations(
            catcher.last_attn,
            catcher.last_tokens,
            out_h=image_size,
            out_w=image_size,
        )

        rgbs.append(rgb)
        attn_maps.append(attn_map)
        token_maps.append(token_map)
        valid_paths.append(p)

    for h in hooks:
        h.remove()

    if not valid_paths:
        raise RuntimeError("No valid images found. Provide existing paths via IMAGE_PATHS or --images.")

    n = len(valid_paths)

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


def load_student_from_checkpoint(config: DINOConfig, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load student via DINOSession checkpoint as requested."""
    session = DINOSession(config, device=device)
    session.load(ckpt_path)
    student = session.student.to(device)
    student.eval()
    return student


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
    image_paths = [Path(p) for p in (args.images if args.images is not None and len(args.images) > 0 else IMAGE_PATHS)]

    config = load_yaml(Path(args.path_config), DINOConfig)
    student = load_student_from_checkpoint(config=config, ckpt_path=ckpt_path, device=device)
    image_size = args.image_size if args.image_size is not None else config.model_base_res

    render_feature_grids(
        student=student,
        image_paths=image_paths,
        output_path=output_path,
        image_size=image_size,
        device=device,
    )


if __name__ == "__main__":
    main()
