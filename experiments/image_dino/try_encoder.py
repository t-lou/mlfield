import argparse
from pathlib import Path

import torch

from components.utils.config import load_yaml
from components.utils.device import get_device
from components.vit.dino_defs import DINOConfig
from components.vit.dino_inf import preprocess_image
from components.vit.dino_session import DINOSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a DINO checkpoint and extract a feature for one image")
    parser.add_argument("--config", default="experiments/image_dino/dino_config.yaml", help="Path to DINO config yaml")
    parser.add_argument("--ckpt", default="dino256_checkpoints/epoch_040.pth", help="Path to DINO checkpoint")
    parser.add_argument("--image", default="data/lenna/Lenna.png", help="Path to input image")
    args = parser.parse_args()

    config_path = Path(args.config if args.config else "experiments/image_dino/dino_config.yaml")
    ckpt_path = Path(args.ckpt if args.ckpt else "dino256_checkpoints/epoch_040.pth")
    image_path = Path(args.image if args.image else "data/lenna/Lenna.png")

    config = load_yaml(config_path, DINOConfig)
    device = get_device()

    encoder = DINOSession.build_encoder(config=config, path_ckpt=ckpt_path, device=device)

    x, _ = preprocess_image(image_path, config.model_base_res)
    print(f"Image size is {x.shape}")

    x = x.unsqueeze(0).to(device)

    print(f"Input size is {x.shape}")

    with torch.inference_mode():
        feature_cls = encoder(x)
        feature_full = encoder.forward_full(x)

    print(f"Feature CLS size is {feature_cls.shape}")
    print(f"Feature full is {feature_full.shape}")

    out_path = Path("data/temp/dino_feature.pt")
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"feature_cls": feature_cls.cpu(), "feature_full": feature_full.cpu()}, out_path)


if __name__ == "__main__":
    main()
