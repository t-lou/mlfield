import sys

import torch.optim as optim
from components.definitions.train import TrainConfig
from components.mmperc.model.simple_model import SimpleModel
from components.mmperc.pipeline.train_a2d2 import train_model
from components.utils.device import get_device
from datasets.a2d2_dataset import A2D2Dataset, bev_collate
from torch.utils.data import DataLoader


def main(config_name: str = "lite"):
    path_data = ""
    device = get_device()
    settings = {
        "lite": TrainConfig(
            batch_size=2,
            num_workers=1,
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=False,
            shuffle=True,
            num_epoch=1,
        ),
        "normal": TrainConfig(
            batch_size=16,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            num_epoch=10,
        ),
    }
    assert config_name in settings, f"Config {config_name} not found."

    chosen_setting = settings[config_name]

    dataset_train = A2D2Dataset(root=path_data)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=chosen_setting.batch_size,
        shuffle=True,
        collate_fn=bev_collate,
        num_workers=chosen_setting.num_workers,
        pin_memory=chosen_setting.pin_memory,
        persistent_workers=chosen_setting.persistent_workers,
        prefetch_factor=chosen_setting.prefetch_factor,
    )

    dataset_eval = A2D2Dataset(root=path_data)
    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=chosen_setting.batch_size * 2,
        shuffle=True,
        collate_fn=bev_collate,
        num_workers=chosen_setting.num_workers,
        pin_memory=chosen_setting.pin_memory,
        persistent_workers=chosen_setting.persistent_workers,
        prefetch_factor=chosen_setting.prefetch_factor,
    )

    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(
        model,
        dataloader_train,
        dataloader_eval,
        optimizer,
        device,
        num_epochs=chosen_setting.batch_size,
        ckpt_dir="checkpoints",
    )


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "normal"
    main(config_name)
