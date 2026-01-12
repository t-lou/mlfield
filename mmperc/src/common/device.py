# device_utils.py

import os

import torch


def get_best_device():
    """
    Returns the best available device:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)

    You can override by setting:
        export DEVICE=cuda
        export DEVICE=cpu
        export DEVICE=mps
    """
    # Manual override
    override = os.environ.get("DEVICE", "").lower()
    if override in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[device_utils] WARNING: CUDA override requested but not available. Falling back to auto.")
    elif override == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device_utils] WARNING: MPS override requested but not available. Falling back to auto.")
    elif override == "cpu":
        return torch.device("cpu")

    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def to_device(obj, device):
    """
    Moves tensors, dicts, lists, and modules to the target device.
    Useful for keeping your pipeline consistent.
    """
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)

    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)

    return obj  # leave untouched if not a tensor/module
