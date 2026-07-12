from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


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
def get_features(
    encoder: torch.nn.Module,
    image_paths: List[Path],
    image_size: int,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """Create and save two grids: attention maps and patch-token feature maps."""

    catcher = ActivationCatcher()
    hooks = []
    for name, module in encoder.backbone.vit.named_modules():
        hooks.append(module.register_forward_hook(catcher.make_hook(name)))

    rgbs = []
    attn_maps = []
    token_maps = []

    for p in image_paths:
        x, rgb = preprocess_image(p, image_size)
        x = x.unsqueeze(0).to(device)

        catcher.last_attn = None
        catcher.last_tokens = None
        _ = encoder(x)

        attn_map, token_map = extract_maps_from_activations(
            catcher.last_attn,
            catcher.last_tokens,
            out_h=image_size,
            out_w=image_size,
        )

        rgbs.append(rgb)
        attn_maps.append(attn_map)
        token_maps.append(token_map)

    for h in hooks:
        h.remove()

    return rgbs, attn_maps, token_maps
