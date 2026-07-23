from dataclasses import dataclass


@dataclass
class TrainConfig:
    # DataLoader
    batch_size: int = 2
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = True
    num_epoch: int = 10
    dir_ckpts: str = "./checkpoints"
