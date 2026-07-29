import argparse
import sys
from pathlib import Path

from components.definitions.mmperc_params import MmpercParams
from components.mmperc.decoder.decode_a2d2 import ModelInferenceWrapper
from components.utils.config import load_yaml
from components.utils.device import get_device
from components.utils.logger import configure_logger, logger


def main(params: MmpercParams, ckpt: Path, output: Path):
    device = get_device()

    logger.info("Instantiating ModelInferenceWrapper...")

    model_inference_wrapper = ModelInferenceWrapper(ckpt=ckpt, params=params, device=device)
    logger.info("ModelInferenceWrapper instantiated successfully.")

    results = model_inference_wrapper.infer_a2d2_dataset(params, output)
    logger.info(results)


if __name__ == "__main__":
    configure_logger("mmperc_inf")

    parser = argparse.ArgumentParser(description="MMPERC inference")
    parser.add_argument(
        "--path-config",
        type=Path,
        default="./experiments/mmperc/mmperc_config.yaml",
        help="Path to MMPERC config YAML",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default="./mmperc_checkpoints/last.pth",
        help="Path to the checkpoint",
    )
    parser.add_argument("--output", type=Path, default="./results.npz", help="Path to the output file")

    args = parser.parse_args()

    if not args.ckpt.exists():
        logger.error(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(Path(args.path_config), MmpercParams)
    main(cfg, args.ckpt, args.output)
