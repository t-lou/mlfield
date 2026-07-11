import argparse
from pathlib import Path

import torch

from components.utils.config import load_yaml
from components.utils.device import get_device
from components.vit.dino_defs import DINOConfig
from components.vit.dino_session import DINOSession


def load_student_from_checkpoint(config: DINOConfig, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load student via DINOSession checkpoint as requested."""
    session = DINOSession(config, device=device)
    session.load(ckpt_path)
    student = session.student.to(device)
    student.eval()
    return student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try to load all checkpoints as a smoketest for model integraty")
    parser.add_argument(
        "--path-config", type=str, default="./experiments/image_dino/dino_config.yaml", help="Path for the configs"
    )
    parser.add_argument("--ckpts", type=str, required=True, help="Path to DINO checkpoints (with *.pth)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    config = load_yaml(Path(args.path_config), DINOConfig)
    root_ckpt_dir = Path(args.ckpts)

    for ckpt_path in root_ckpt_dir.glob("*.pth"):
        print(f"Loading checkpoint: {ckpt_path}")
        _ = load_student_from_checkpoint(config=config, ckpt_path=ckpt_path, device=device)

    print("All look fine")


if __name__ == "__main__":
    main()
