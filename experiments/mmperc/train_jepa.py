from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from components.dataset.a2d2_dataset_unlabeled import A2D2DatasetUnlabeled
from components.definitions.mmperc_params import MmpercParams
from components.mmperc.encoder.can_encoder import CANEncoder
from components.mmperc.encoder.jepa_encoder import JEPAEncoder, LeJEPA
from components.mmperc.encoder.multi_camera_encoder import MultiCameraEncoder
from components.mmperc.encoder.point_transformer_v3 import PointTransformerV3BEV
from components.mmperc.encoder.tiny_camera_encoder import TinyCameraEncoder
from components.mmperc.fusion.latent_fusion import LatentFusion
from components.utils.config import load_yaml
from components.utils.logger import configure_logger
from torch.utils.data import DataLoader

SENSOR_NAMES = [
    "front_center",
    "front_right",
    "front_left",
    "side_right",
    "side_left",
    "rear_center",
]


def _points_from_npz(value: dict) -> torch.Tensor:
    if "points" in value:
        points = value["points"]
    else:
        names = [name for name in ("x", "y", "z", "intensity") if name in value]
        if len(names) < 3:
            raise KeyError("Lidar archive must contain 'points' or x/y/z arrays")
        points = np.column_stack([value[name] for name in names])
    return torch.as_tensor(points, dtype=torch.float32)


def ssl_collate(
    samples: list[dict],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cameras: dict[str, torch.Tensor] = {}
    lidars: dict[str, list[torch.Tensor]] = {name: [] for name in SENSOR_NAMES}
    can_values: dict[str, list[float]] = {name: [] for name in CANEncoder.DEFAULT_KEYS}

    for sample in samples:
        for sensor_name in SENSOR_NAMES:
            camera = sample.get(("camera", sensor_name))
            lidar = sample.get(("lidar", sensor_name))
            if camera is None or lidar is None:
                raise ValueError(f"Missing camera/lidar pair for {sensor_name}")
            camera_tensor = torch.as_tensor(camera, dtype=torch.float32).permute(2, 0, 1) / 255.0
            cameras.setdefault(sensor_name, []).append(camera_tensor)
            lidars[sensor_name].append(_points_from_npz(lidar))
        can_in = sample.get("can_in")
        if can_in is None:
            raise ValueError("Missing can_in data")
        for name in CANEncoder.DEFAULT_KEYS:
            value = can_in.get(name)
            if value is None:
                raise ValueError(f"Missing CAN value for {name}")
            can_values[name].append(float(value))

    camera_batch = {name: torch.stack(values) for name, values in cameras.items()}
    lidar_batch = {name: torch.nn.utils.rnn.pad_sequence(values, batch_first=True) for name, values in lidars.items()}
    can_batch = {name: torch.tensor(values, dtype=torch.float32) for name, values in can_values.items()}
    return lidar_batch, camera_batch, can_batch


def build_model(params: MmpercParams) -> LeJEPA:
    camera_encoder = MultiCameraEncoder(
        {name: TinyCameraEncoder(params=params, sensor_name=name) for name in SENSOR_NAMES}
    )
    lidar_encoder = PointTransformerV3BEV(params=params, sensor_names=SENSOR_NAMES)
    dim = params.bev_params.bev_channels
    can_encoder = CANEncoder(dim=dim)
    fusion = LatentFusion(
        dim=dim,
        camera_names=SENSOR_NAMES,
        camera_channels=dim,
        lidar_channels=dim,
        can_channels=dim,
        num_latents=32,
        num_heads=8,
        depth=2,
        share_weights=True,
    )
    encoder = JEPAEncoder(lidar_encoder, camera_encoder, fusion, can_encoder=can_encoder)
    return LeJEPA(encoder, dim=dim, sigreg_weight=0.1)


def train(params: MmpercParams, checkpoint_dir: str = "checkpoints-jepa") -> None:
    device = params.train_config.get_device()
    dataset = A2D2DatasetUnlabeled(params=params, decode_images=True)
    loader = DataLoader(
        dataset,
        batch_size=params.train_config.batch_size,
        shuffle=params.train_config.shuffle,
        num_workers=params.train_config.num_workers,
        pin_memory=params.train_config.pin_memory,
        collate_fn=ssl_collate,
    )
    model = build_model(params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(params.train_config.num_epoch):
        model.train()
        batches = 0
        epoch_totals = {name: 0.0 for name in ("loss", "prediction_loss", "sigreg_loss")}
        for lidar_points, camera_images, can_values in loader:
            lidar_points = {name: value.to(device, non_blocking=True) for name, value in lidar_points.items()}
            camera_images = {name: value.to(device, non_blocking=True) for name, value in camera_images.items()}
            can_values = {name: value.to(device, non_blocking=True) for name, value in can_values.items()}
            output = model(lidar_points, camera_images, can_tokens=can_values, mask_ratio=0.5)
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            optimizer.step()
            batches += 1
            for name in epoch_totals:
                epoch_totals[name] += output[name].item()

        if batches == 0:
            raise RuntimeError("The training DataLoader produced no batches")
        epoch_means = {name: value / batches for name, value in epoch_totals.items()}

        print(
            f"epoch={epoch} loss={epoch_means['loss']:.4f} "
            f"prediction={epoch_means['prediction_loss']:.4f} "
            f"sigreg={epoch_means['sigreg_loss']:.4f}"
        )
        torch.save(model.state_dict(), Path(checkpoint_dir) / f"lejepa_epoch_{epoch:04d}.pt")


if __name__ == "__main__":
    configure_logger("mmperc_jepa_train")
    parser = argparse.ArgumentParser(description="Multimodal LeJEPA pretraining")
    parser.add_argument(
        "--path-config",
        type=str,
        default="./experiments/mmperc/mmperc_config.yaml",
        help="Path to MMPERC config YAML",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints-jepa")
    args = parser.parse_args()
    train(load_yaml(Path(args.path_config), MmpercParams), checkpoint_dir=args.checkpoint_dir)
