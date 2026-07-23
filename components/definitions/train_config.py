from dataclasses import dataclass

import torch

from components.utils.device import get_device, resolve_num_workers


@dataclass
class TrainConfig:
    """Configuration for training a model."""

    batch_size: int = 32
    # number of DataLoader workers; if None, will be resolved automatically
    num_workers: int | None = None
    # number of samples loaded in advance by each worker
    prefetch_factor: int = 2
    # whether to pin memory in DataLoader (recommended for GPU training)
    pin_memory: bool = True
    # whether to keep DataLoader workers alive between epochs (recommended for GPU training)
    persistent_workers: bool = True
    # whether to shuffle the training data
    shuffle: bool = True
    num_epoch: int = 10
    dir_ckpts: str = "./checkpoints"
    device: str | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = get_device().type
        if self.num_workers is None:
            self.num_workers = resolve_num_workers()

    def get_device(self) -> torch.device:
        """Return the device as a torch.device object."""
        return torch.device(self.device)
