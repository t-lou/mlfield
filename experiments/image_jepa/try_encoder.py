"""
Downstream encoding with I-JEPA.

Loads a checkpoint into I_JEPA, uses the frozen target encoder (full-image,
no masking) as a feature extractor, runs one image through, and prints the output shape.

Usage
-----
python -m experiments.image_jepa.encode_image \
    --ckpt  path/to/checkpoint.pth \
    --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from components.utils.device import get_device
from components.vit.i_jepa import I_JEPA
from components.vit.i_jepa_defs import IJEPAConfig

# ---------------------------------------------------------------------------
# Downstream wrapper
# ---------------------------------------------------------------------------


class IJEPAEncoder(nn.Module):
    """
    Target encoder from I-JEPA.

    The target encoder always sees the full (unmasked) image and is EMA-
    updated during training, making it the natural feature extractor.
    Patch tokens are mean-pooled to produce a fixed-size representation.
    """

    def __init__(self, model: I_JEPA) -> None:
        super().__init__()
        self.target_encoder = model.target_encoder
        self.cfg = model.cfg

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings: (B, N_patches, embed_dim)."""
        self.target_encoder.eval()
        tokens = self.target_encoder.forward_full(
            imgs,
            patch_keep_mask=None,
            add_cls_token=False,
        )
        assert self.cfg.image_size % self.cfg.patch_size == 0, "Image size must be divisible by patch size"
        grid_h = self.cfg.image_size // self.cfg.patch_size
        grid_w = self.cfg.image_size // self.cfg.patch_size
        tokens = tokens.view(tokens.size(0), grid_h, grid_w, tokens.size(2))
        return tokens

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode the image"""
        features = self.encode(imgs)
        return features


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(model: I_JEPA, ckpt_path: Path, device: torch.device) -> None:
    """Load weights into model, handling common checkpoint wrapper formats."""
    raw = torch.load(str(ckpt_path), map_location=device)

    assert "model" in raw, f"Checkpoint {ckpt_path} must contain 'model' entry"
    state: dict[str, torch.Tensor] = raw["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    """Load, resize, and centre-crop an image to (1, 3, image_size, image_size)."""
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    with Image.open(path) as img:
        tensor = tfm(img.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="I-JEPA downstream encoding demo")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to I-JEPA checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--vis-output", type=str, default=None, help="Path to save visualized output (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    image_path = Path(args.image)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()

    # Build model from default config and load weights.
    cfg = IJEPAConfig()
    model = I_JEPA(cfg).to(device)
    load_checkpoint(model, ckpt_path, device)

    # Build encoder wrapper for downstream use.
    encoder = IJEPAEncoder(model).to(device)

    # Load and preprocess the image.
    imgs = load_image(image_path, cfg.image_size, device)
    print(f"Input image tensor shape : {tuple(imgs.shape)}")

    # Intermediate feature shape.
    with torch.no_grad():
        features = encoder.encode(imgs)
    print(f"Encoded feature shape    : {tuple(features.shape)}")

    # Full downstream output shape.
    with torch.no_grad():
        logits = encoder(imgs)
    print(f"Output shape : {tuple(logits.shape)}")

    if args.vis_output:
        vis_path = Path(args.vis_output)
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        # Visualize the patch embeddings as a grid.
        # Note: This is a simple visualization; for better results, consider using PCA or t-SNE.
        import matplotlib.pyplot as plt

        features_grid = features.squeeze(0).cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        # show original image
        plt.imshow(imgs.squeeze(0).permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("Input Image")
        plt.subplot(1, 3, 2)
        plt.imshow(features_grid.mean(axis=-1), cmap="viridis")
        plt.axis("off")
        plt.title("Mean Patch Embeddings")
        plt.subplot(1, 3, 3)
        plt.imshow(features_grid.max(axis=-1), cmap="magma")
        plt.axis("off")
        plt.title("Max Patch Embeddings")
        plt.savefig(vis_path)
        plt.close()
        print(f"Visualization saved to: {vis_path}")


if __name__ == "__main__":
    main()
