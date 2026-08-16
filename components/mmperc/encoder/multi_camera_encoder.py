from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class MultiCameraEncoder(nn.Module):
    """
    Generic multi-camera encoder adapter.

    Each camera branch is expected to implement the same encoder API as
    TinyCameraEncoder: forward(image) -> (tokens, feat, skip_feats).

    The wrapper is intentionally independent from the concrete camera encoder and
    only adds per-camera identity and pose metadata. The pose metadata should
    follow the calibration convention: a 4x4 vehicle-frame transform or a valid
    flattened equivalent.
    """

    def __init__(self, cam_encoders: dict[str, nn.Module]):
        super().__init__()
        self.cam_encoders = nn.ModuleDict(cam_encoders)
        self.cam_ids = list(cam_encoders.keys())

        self.use_meta = True
        C = next(iter(cam_encoders.values())).out_channels
        self.cam_id_embed = nn.Embedding(len(self.cam_ids), C)
        self.pose_mlp = nn.Linear(16, C)
        self.pose_mlp_legacy = nn.Linear(6, C)

    def _extract_pose_vector(self, meta: dict[str, Any] | Any, device: torch.device) -> Tensor:
        pose_obj = None
        if isinstance(meta, dict):
            pose_obj = meta.get("pose")
            if pose_obj is None and "calibration" in meta:
                calibration = meta["calibration"]
                if hasattr(calibration, "pose"):
                    pose_obj = calibration.pose
                else:
                    pose_obj = calibration
            if pose_obj is None and "extrinsics" in meta:
                pose_obj = meta["extrinsics"]
        else:
            pose_obj = meta

        if pose_obj is not None:
            if hasattr(pose_obj, "sensor_from_vehicle"):
                pose_obj = pose_obj.sensor_from_vehicle
            elif hasattr(pose_obj, "vehicle_from_sensor"):
                pose_obj = pose_obj.vehicle_from_sensor
            elif isinstance(pose_obj, dict):
                for key in ("sensor_from_vehicle", "vehicle_from_sensor", "extrinsics"):
                    if key in pose_obj:
                        pose_obj = pose_obj[key]
                        break

        if pose_obj is None:
            raise ValueError("MultiCameraEncoder requires camera pose metadata in each camera entry.")

        pose = torch.as_tensor(pose_obj, device=device, dtype=torch.float32)
        if pose.dim() == 2:
            if pose.shape == (3, 4):
                pose = torch.cat(
                    [pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)], dim=0
                )
            pose = pose.reshape(-1)
        elif pose.dim() == 1:
            pose = pose.reshape(-1)

        if pose.numel() == 6:
            return pose
        if pose.numel() == 16:
            return pose
        raise ValueError(
            "MultiCameraEncoder pose must be a 4x4 matrix, a flattened 16-vector, or a legacy 6-vector. "
            f"Got {tuple(pose.shape)} with {pose.numel()} elements."
        )

    def forward(self, images: dict[str, Tensor], cam_meta: dict[str, dict[str, Any]]):
        """
        images: {camera_id: (B, 3, H, W)}
        cam_meta: {
            camera_id: {
                "pose": SensorPose or 4x4 transform,
                "extrinsics": optional legacy 3x4 / 4x4 / 6-vector,
                "camera_id": int or str,
            }
        }
        """
        tokens_list: list[Tensor] = []

        for idx, cam_id in enumerate(self.cam_ids):
            img = images[cam_id]
            encoder = self.cam_encoders[cam_id]

            tokens, _, _ = encoder(img)

            if self.use_meta:
                meta = cam_meta[cam_id]
                cam_id_tensor = torch.tensor([idx], device=img.device, dtype=torch.long)
                cam_id_emb = self.cam_id_embed(cam_id_tensor).view(1, 1, -1)

                pose_vec = self._extract_pose_vector(meta, img.device)
                if pose_vec.numel() == 6:
                    pose_emb = self.pose_mlp_legacy(pose_vec).view(1, 1, -1)
                else:
                    pose_emb = self.pose_mlp(pose_vec).view(1, 1, -1)

                tokens = tokens + cam_id_emb + pose_emb

            tokens_list.append(tokens)

        if not tokens_list:
            raise ValueError("MultiCameraEncoder received an empty camera list.")

        return torch.cat(tokens_list, dim=1)
