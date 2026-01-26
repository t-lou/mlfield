from dataclasses import dataclass

import torch.optim as optim
from torch.utils.data import DataLoader

import common.params as params
from common.device import get_best_device
from datasets.a2d2_dataset import A2D2Dataset, bev_collate
from model.simple_model import SimpleModel
from pipeline.train_bbox2d import train_model


@dataclass
class TrainConfig:
    # DataLoader
    batch_size: int = 2
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = True


def main():
    device = get_best_device()
    settings = {
        "lite": TrainConfig(
            batch_size=2,
            num_workers=1,
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=False,
            shuffle=True,
        ),
        "normal": TrainConfig(
            batch_size=16,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        ),
    }
    chosen_setting = settings["lite"]

    path_dataset = params.PATH_TRAIN
    dataset = A2D2Dataset(root=path_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=chosen_setting.batch_size,
        shuffle=True,
        collate_fn=bev_collate,
        num_workers=chosen_setting.num_workers,
        pin_memory=chosen_setting.pin_memory,
        persistent_workers=chosen_setting.persistent_workers,
        prefetch_factor=chosen_setting.prefetch_factor,
    )

    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, dataloader, optimizer, device, num_epochs=5, ckpt_dir="checkpoints")


if __name__ == "__main__":
    main()
