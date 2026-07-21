import argparse
import sys
from pathlib import Path

from components.definitions.mmperc import MmpercParams
from components.mmperc.decoder.decode_a2d2.bbox import ModelInferenceWrapper
from components.utils.config import load_yaml
from components.utils.device import get_device
from components.utils.logger import configure_logger, logger

if __name__ == "__main__":
    configure_logger("mmperc_inf")

    parser = argparse.ArgumentParser(description="MMPERC training (patched from proposal scaffold)")
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

    assert args.ckpt.exists()

    cfg = load_yaml(Path(args.path_config), MmpercParams)
    device = get_device()

    logger.info("Testing ModelInferenceWrapper instantiation...")

    try:
        model_inference_wrapper = ModelInferenceWrapper(ckpt=args.ckpt, device=device)
        logger.info("ModelInferenceWrapper instantiated successfully.")

        results = model_inference_wrapper.infer_a2d2_dataset(cfg, "/workspace/mmperc/data/a2d2_output.npz")
        logger.info(results)

    except Exception as e:
        logger.info(f"Error instantiating ModelInferenceWrapper: {e}")
        sys.exit(1)
