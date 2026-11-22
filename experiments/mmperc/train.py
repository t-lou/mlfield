import argparse
import logging
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch.optim as optim
from components.dataset.a2d2_dataset import A2D2Dataset, Split, bev_collate
from components.definitions.mmperc_params import MmpercParams
from components.mmperc.model.transformer_model import SMALL_GPU_PTV3_CONFIG, SMALL_GPU_VIT_CONFIG, TransformerModel
from components.mmperc.pipeline.train_a2d2 import train_model
from components.utils.config import load_yaml
from components.utils.logger import configure_logger
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


def _build_dataloader(
    dataset: Dataset,
    params: MmpercParams,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    config = params.train_config
    num_workers = config.num_workers or 0
    if num_workers < 0:
        raise ValueError("train_config.num_workers must be non-negative")
    if batch_size < 1:
        raise ValueError("DataLoader batch_size must be positive")
    if num_workers > 0 and config.prefetch_factor < 1:
        raise ValueError("train_config.prefetch_factor must be positive when num_workers is non-zero")

    device = config.get_device()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(bev_collate, params=params),
        num_workers=num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        persistent_workers=config.persistent_workers and num_workers > 0,
        prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
    )


def main(params: MmpercParams, skip_eval: bool = False) -> None:
    train_config = params.train_config
    if train_config.batch_size < 1:
        raise ValueError("train_config.batch_size must be positive")
    if train_config.num_epoch < 1:
        raise ValueError("train_config.num_epoch must be positive")

    # TransformerModel currently requires full-resolution images. Use a separate
    # config copy so calling main does not mutate the caller's parameters.
    model_params = replace(params, image_scale=1.0)
    if params.image_scale != model_params.image_scale:
        log.warning(
            "TransformerModel requires image_scale=1.0; overriding configured value %.3f",
            params.image_scale,
        )

    device = train_config.get_device()

    dataset_train = A2D2Dataset(params=model_params, split=Split.TRAIN)
    dataloader_train = _build_dataloader(
        dataset_train,
        model_params,
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
    )

    dataloader_eval = None
    if not skip_eval:
        dataset_eval = A2D2Dataset(params=model_params, split=Split.VAL)
        dataloader_eval = _build_dataloader(
            dataset_eval,
            model_params,
            batch_size=train_config.batch_size * 2,
            shuffle=False,
        )

    def train_one_model(model: nn.Module, ckpt_dir: str = "checkpoints") -> None:
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        scheduler = CosineAnnealingLR(optimizer, T_max=train_config.num_epoch, eta_min=1e-6)

        train_model(
            model,
            dataloader_train,
            dataloader_eval,
            optimizer,
            scheduler,
            device,
            num_epochs=train_config.num_epoch,
            ckpt_dir=ckpt_dir,
            skip_eval=skip_eval,
        )

    train_one_model(
        TransformerModel(
            params=model_params,
            vit_config=SMALL_GPU_VIT_CONFIG,
            ptv3_config=SMALL_GPU_PTV3_CONFIG,
        ),
        ckpt_dir=str(Path(train_config.dir_ckpts) / "transformer_model"),
    )


if __name__ == "__main__":
    configure_logger("mmperc_train")

    parser = argparse.ArgumentParser(description="MMPERC training (patched from proposal scaffold)")
    parser.add_argument(
        "--path-config",
        type=str,
        default="./experiments/mmperc/mmperc_config.yaml",
        help="Path to MMPERC config YAML",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation during training")

    args = parser.parse_args()

    cfg = load_yaml(Path(args.path_config), MmpercParams)
    main(cfg, skip_eval=args.skip_eval)
