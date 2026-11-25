"""Export encoder and full-teacher PyTorch/ONNX models from a Sonata training checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.utils.config import load_yaml, load_yaml_recursive  # noqa: E402
from components.utils.device import get_device  # noqa: E402
from experiments.ssl.sonata import (  # noqa: E402
    PointTransformerV3Config,
    Sonata,
    SonataConfig,
    SonataExperimentConfig,
    _small_gpu_profile,
    export_sonata_encoder_onnx,
    export_sonata_encoder_pt,
    export_sonata_full_onnx,
    export_sonata_full_pt,
)


def _load_checkpoint_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
    small_gpu: bool = False,
) -> tuple[Sonata, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a Sonata training checkpoint with a 'model' state dict")
    state = checkpoint["model"]
    stem_weight_key = "teacher.backbone.embed_conv.weight"
    stem_weight = state.get(stem_weight_key) if isinstance(state, dict) else None
    if not isinstance(stem_weight, torch.Tensor) or stem_weight.ndim < 2:
        raise ValueError(f"Checkpoint is missing a valid {stem_weight_key!r} tensor")
    in_channels = stem_weight.shape[1]

    if "model_config" in checkpoint:
        model_config_data = checkpoint["model_config"]
        if not isinstance(model_config_data, dict):
            raise ValueError(f"{checkpoint_path} has an invalid 'model_config'")
        model_config = load_yaml_recursive(model_config_data, SonataConfig)
        model_config.backbone = load_yaml_recursive(
            model_config_data["backbone"],
            PointTransformerV3Config,
        )
    else:
        experiment_config = load_yaml(config_path, SonataExperimentConfig)
        model_config = _small_gpu_profile(experiment_config.model) if small_gpu else experiment_config.model
        model_config.backbone = replace(model_config.backbone, in_channels=in_channels)

    model = Sonata(model_config)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, in_channels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a Sonata training checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("sonata_config.yml"),
        help="Training YAML config used to create the checkpoint architecture",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Export directory (default: <checkpoint directory>/exports)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=4096,
        help="Fixed number of points for ONNX exports (default: 4096)",
    )
    parser.add_argument(
        "--small-gpu",
        action="store_true",
        help="Recreate the compact architecture for legacy checkpoints trained with --small-gpu",
    )
    parser.add_argument("--device", help="Torch device (defaults to the project device selection)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    model, in_channels = _load_checkpoint_model(args.checkpoint, args.config, device, args.small_gpu)
    if args.num_points < 1:
        parser.error("--num-points must be positive")
    output_dir = args.output_dir or args.checkpoint.parent / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_pt = output_dir / "sonata_teacher_backbone.pt"
    encoder_onnx = output_dir / "sonata_teacher_backbone.onnx"
    full_pt = output_dir / "sonata_full.pt"
    full_onnx = output_dir / "sonata_full.onnx"

    export_sonata_encoder_pt(model, encoder_pt)
    export_sonata_encoder_onnx(model, encoder_onnx, in_channels, args.num_points)
    export_sonata_full_pt(model, full_pt)
    export_sonata_full_onnx(model, full_onnx, in_channels, args.num_points)

    print(f"Exported teacher encoder PyTorch weights: {encoder_pt}")
    print(f"Exported teacher encoder ONNX model: {encoder_onnx}")
    print(f"Exported full teacher PyTorch weights: {full_pt}")
    print(f"Exported full teacher ONNX model: {full_onnx}")


if __name__ == "__main__":
    main()
