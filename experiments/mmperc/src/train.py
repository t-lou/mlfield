import argparse
from functools import partial
from pathlib import Path

import torch.optim as optim
from components.dataset.a2d2_dataset import A2D2Dataset, Split, bev_collate
from components.definitions.mmperc import MmpercParams
from components.mmperc.model.simple_model import SimpleModel
from components.mmperc.pipeline.train_a2d2 import train_model
from components.utils.config import load_yaml
from components.utils.device import get_device
from components.utils.logger import configure_logger
from torch.utils.data import DataLoader


def main(params: MmpercParams):

    train_config = params.train_config
    device = get_device()

    dataset_train = A2D2Dataset(path_tar=Path(params.path_data), params=params, split=Split.TRAIN)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=partial(bev_collate, params=params),
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        persistent_workers=train_config.persistent_workers,
        prefetch_factor=train_config.prefetch_factor,
    )

    dataset_eval = A2D2Dataset(path_tar=Path(params.path_data), params=params, split=Split.VAL)
    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=train_config.batch_size * 2,
        shuffle=True,
        collate_fn=partial(bev_collate, params=params),
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        persistent_workers=train_config.persistent_workers,
        prefetch_factor=train_config.prefetch_factor,
    )

    model = SimpleModel(params=params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(
        model,
        dataloader_train,
        dataloader_eval,
        optimizer,
        device,
        num_epochs=train_config.batch_size,
        ckpt_dir="checkpoints",
    )


if __name__ == "__main__":
    configure_logger("mmperc")

    parser = argparse.ArgumentParser(description="MMPERC training (patched from proposal scaffold)")
    parser.add_argument(
        "--path-config",
        type=str,
        default="./experiments/mmperc/mmperc_config.yaml",
        help="Path to MMPERC config YAML",
    )

    args = parser.parse_args()

    cfg = load_yaml(Path(args.path_config), MmpercParams)
    main(cfg)
