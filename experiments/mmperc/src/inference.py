import argparse
import sys
from pathlib import Path

from components.definitions.mmperc import MmpercParams
from components.mmperc.decoder.decode_a2d2 import ModelInferenceWrapper
from components.utils.config import load_yaml
from components.utils.device import get_device
from components.utils.logger import configure_logger, logger


def main(params: MmpercParams, ckpt: Path):
    device = get_device()

    logger.info("Instantiating ModelInferenceWrapper...")

    model_inference_wrapper = ModelInferenceWrapper(ckpt=ckpt, params=params, device=device)
    logger.info("ModelInferenceWrapper instantiated successfully.")

    results = model_inference_wrapper.infer_a2d2_dataset(params, "./mmperc_inference_out/results.npz")
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

    args = parser.parse_args()

    if not args.ckpt.exists():
        logger.error(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    cfg = load_yaml(Path(args.path_config), MmpercParams)
    main(cfg, args.ckpt)
