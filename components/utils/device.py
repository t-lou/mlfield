import os
from typing import Optional

import torch

from components.utils.logger import logger


def get_device() -> torch.device:
    """Return the best available device (GPU, MPS, or CPU) for PyTorch."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal Performance Shaders (MPS) backend")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def resolve_num_workers(num_workers: Optional[int] = None, max_num_workers: int = 20) -> int:
    """Return a usable number of DataLoader workers, preferring the CPU count by default."""
    if num_workers is not None:
        ret = max(0, int(num_workers))
    else:
        ret = os.cpu_count() or 1
        ret = max(1, ret)

    ret = min(max_num_workers, ret)

    logger.info(f"Resolved number of DataLoader workers: {ret}")
    return ret
