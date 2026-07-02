from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import common.params as params


def move_to_device(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    """
    Move a batch of tensors to the target device.
    """
    return {k: v.to(device) for k, v in batch.items()}


def reduce_output_to_scalar(out: Any) -> Tensor:
    """
    Convert model output (tensor or dict of tensors) into a single scalar.
    """
    if torch.is_tensor(out):
        return out.sum()

    if isinstance(out, dict):
        return sum(v.sum() for v in out.values() if torch.is_tensor(v))

    raise TypeError(f"Unsupported model output type: {type(out)}")


def measure_memory(
    model: torch.nn.Module,
    batch: Dict[str, Tensor],
    training: bool,
    device: str,
) -> int:
    """
    Run a forward/backward pass on a REAL batch and return peak GPU memory in bytes.
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    batch = move_to_device(batch, device)

    if training:
        model.train()
        out = model(**batch)
        loss = reduce_output_to_scalar(out)
        loss.backward()
    else:
        model.eval()
        with torch.no_grad():
            _ = model(**batch)

    return torch.cuda.max_memory_allocated(device)


def try_batch_size(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    training: bool,
    device: str,
) -> int:
    """
    Attempt to load a REAL batch of size `batch_size` and measure memory.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    return measure_memory(model, batch, training, device)


def find_max_batch_size(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    training: bool = True,
    max_bs: int = 1024,
    device: str = "cuda",
) -> int:
    """
    Binary search for the largest REAL batch size that fits in GPU memory.
    """
    low, high = 1, max_bs
    best = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            print(f"Trying batch size: {mid}...")
            _ = try_batch_size(model, dataset, mid, training, device)
            best = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise e

    return best


def get_parameter_size(model: torch.nn.Module) -> int:
    """
    Return total parameter memory in bytes.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


if __name__ == "__main__":
    from common.device import get_best_device
    from datasets.a2d2_dataset import A2D2Dataset
    from model.simple_model import SimpleModel

    device: str = get_best_device()
    print(f"Using device: {device}")

    dataset = A2D2Dataset(params.PATH_TRAIN)
    model = SimpleModel().to(device)

    param_mem_mb: float = get_parameter_size(model) / 1024**2
    print(f"Parameter memory: {param_mem_mb:.2f} MB")

    max_bs: int = find_max_batch_size(model, dataset, training=True, device=device)
    print(f"Max REAL batch size that fits: {max_bs}")
