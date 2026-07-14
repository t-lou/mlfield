import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from components.dataset.image_only_dataset import ImageOnlyDataset
from components.utils.config import load_yaml
from components.utils.device import get_device
from components.utils.logger import configure_logger
from components.vit.mae import MAE
from components.vit.mae_defs import MAEConfig


def visualize_mae(model: MAE, imgs: torch.Tensor, save_path: Path) -> None:
    """
    Visualize MAE reconstruction quality by showing original, masked, and reconstructed images.

    Creates a 3xN subplot figure where:
    - Row 0: Original images
    - Row 1: Images with masked patches zeroed out
    - Row 2: Reconstructed images from encoder-decoder

    Visually demonstrates what MAE learned to reconstruct masked regions.
    Good diagnostic tool for evaluating model training progress.

    Args:
        model: Trained MAE model (will be set to eval mode)
        imgs: Batch of input images (batch, 3, H, W)
        save_path: Path to save visualization image

    Note:
        - Shows maximum 6 images to keep figure readable
        - Clipped to [0, 1] range for display
        - Uses model.eval() and torch.no_grad() for inference

    Improvement: Consider adding:
        - Difference maps (original - reconstruction)
        - Uncertainty/confidence estimates
        - Progressive reconstruction frames (decoder layers)
        - Histogram of reconstruction errors
    """
    model.eval()
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(imgs, model.cfg.mask_ratio)
        pred = model.forward_decoder(latent, ids_restore)
        rec_imgs = model.unpatchify(pred)

        B, C, H, W = imgs.shape
        patch = model.cfg.patch_size

        mask = mask.unsqueeze(-1).repeat(1, 1, patch * patch * C)
        mask = model.unpatchify(mask)
        masked_imgs = imgs * (1 - mask)

    num_show = min(6, imgs.shape[0])
    fig, axes = plt.subplots(3, num_show, figsize=(3 * num_show, 9))

    for i in range(num_show):
        axes[0, i].imshow(imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(masked_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[1, i].set_title("Masked")
        axes[1, i].axis("off")

        axes[2, i].imshow(rec_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[2, i].set_title("Reconstructed")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MAE debug trainer with CIFAR-10 and ImageNet presets")
    parser.add_argument("--path-config", type=Path, default="./experiments/image_mae/mae_config.yaml")
    parser.add_argument("--path-ckpt", type=Path, default="./mae_checkpoints/final.pth")
    parser.add_argument("--path-output", type=Path, default="./data/temp/mae_vis.png")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for datasets.",
    )
    args = parser.parse_args()

    device = get_device()

    config = load_yaml(args.path_config, MAEConfig)

    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageOnlyDataset([args.data_root], transform=transform)

    loader_kwargs = {
        "batch_size": min(len(dataset), 8),
        "shuffle": False,
    }
    loader = DataLoader(dataset, **loader_kwargs)

    model = MAE(config)
    model.load_checkpoint(path=args.path_config, device=device)
    model = model.to(device)

    model.eval()

    for imgs in loader:
        visualize_mae(model, imgs, save_path=args.path_output)
        break


if __name__ == "__main__":
    configure_logger("mae")
    main()
