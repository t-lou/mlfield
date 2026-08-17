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
        if any(enc.out_channels != C for enc in cam_encoders.values()):
            raise ValueError("MultiCameraEncoder requires all camera encoders to share out_channels.")
        self.out_channels = C
        self.cam_id_embed = nn.Embedding(len(self.cam_ids), C)
        self.pose_mlp = nn.Linear(16, C)
        self.pose_mlp_legacy = nn.Linear(6, C)

        for idx, cam_id in enumerate(self.cam_ids):
            pose_vec = cam_encoders[cam_id].cam_pose_vector
            self.register_buffer(f"_default_pose_{idx}", pose_vec.detach().clone())

    def forward(
        self, images: dict[str, Tensor], cam_meta: dict[str, dict[str, Any]] | None = None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, dict[str, Tensor]]]:
        """
        images: {camera_id: (B, 3, H, W)}
        cam_meta: {
            camera_id: {
                "pose": SensorPose or 4x4 transform,
                "extrinsics": optional legacy 3x4 / 4x4 / 6-vector,
                "camera_id": int or str,
            }
        }

        Returns per-camera outputs, kept separate (not concatenated/ensembled) so
        callers can decide how to combine them — e.g. simple concatenation for
        SimpleModel, or per-camera masking/prediction for JEPA-style pretraining:

            tokens_per_cam: {camera_id: (B, N_cam, C)}
            feats:          {camera_id: (B, C, H', W')}
            skip_feats:     {camera_id: {"s2": ..., "s4": ...}}
        """
        if not self.cam_ids:
            raise ValueError("MultiCameraEncoder received an empty camera list.")

        tokens_per_cam: dict[str, Tensor] = {}
        feats: dict[str, Tensor] = {}
        skip_feats: dict[str, dict[str, Tensor]] = {}

        for idx, cam_id in enumerate(self.cam_ids):
            img = images[cam_id]
            encoder = self.cam_encoders[cam_id]

            tokens, feat, skips = encoder(img)

            if self.use_meta:
                cam_id_tensor = torch.tensor([idx], device=img.device, dtype=torch.long)
                cam_id_emb = self.cam_id_embed(cam_id_tensor).view(1, 1, -1)

                pose_vec = getattr(self, f"_default_pose_{idx}").to(img.device)
                if pose_vec.numel() == 6:
                    pose_emb = self.pose_mlp_legacy(pose_vec).view(1, 1, -1)
                else:
                    pose_emb = self.pose_mlp(pose_vec).view(1, 1, -1)

                tokens = tokens + cam_id_emb + pose_emb

            tokens_per_cam[cam_id] = tokens
            feats[cam_id] = feat
            skip_feats[cam_id] = skips

        return tokens_per_cam, feats, skip_feats
