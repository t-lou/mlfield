from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams
from components.mmperc.encoder.multi_camera_encoder import MultiCameraEncoder
from components.mmperc.encoder.point_transformer_v3 import PointTransformerV3Encoder
from components.mmperc.encoder.vit_camera_encoder import ViTCameraEncoder
from components.mmperc.fusion.futr_fusion import FuTrFusionBlock
from components.mmperc.head.bbox2d_head import BBox3dHead
from components.mmperc.head.semantics_head import FullResSemHead
from components.utils.calibration import load_sensor_calibration


class _PointTransformerBEVEncoder(nn.Module):
    """Adapt PTv3's point features to the dense BEV contract used by fusion."""

    def __init__(self, params: MmpercParams, sensor_names: list[str]) -> None:
        super().__init__()
        self.sensor_names = list(sensor_names)
        self.calibrations = {
            name: load_sensor_calibration(
                Path(params.path_calibration),
                sensor_name=name,
                sensor_type="camera",
            )
            for name in self.sensor_names
        }

        bev = params.bev_params
        self.x_min, self.x_max = bev.x_range
        self.y_min, self.y_max = bev.y_range
        self.z_min, self.z_max = bev.z_range
        self.vx, self.vy, self.vz = bev.voxel_size
        self.bev_h = bev.bev_h
        self.bev_w = bev.bev_w
        self.out_h = self.bev_h // 2
        self.out_w = self.bev_w // 2

        self.point_transformer = PointTransformerV3Encoder(in_channels=4)
        self.feature_proj = nn.Linear(self.point_transformer.out_channels, bev.bev_channels)

    @staticmethod
    def _to_vehicle(calibration, points: Tensor) -> Tensor:
        xyz = points[..., :3]
        ones = torch.ones(*xyz.shape[:-1], 1, device=points.device, dtype=points.dtype)
        xyz_h = torch.cat([xyz, ones], dim=-1)
        transform = torch.as_tensor(
            calibration.pose.vehicle_from_sensor,
            device=points.device,
            dtype=points.dtype,
        )
        return torch.cat([(xyz_h @ transform.T)[..., :3], points[..., 3:4]], dim=-1)

    def _voxelize(self, points: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        batch_size, num_points, _ = points.shape
        device = points.device
        xyz = points[..., :3].reshape(-1, 3)
        raw_features = points[..., :4].reshape(-1, 4)
        batch = torch.arange(batch_size, device=device).repeat_interleave(num_points)

        valid = (
            (xyz[:, 0] >= self.x_min)
            & (xyz[:, 0] < self.x_max)
            & (xyz[:, 1] >= self.y_min)
            & (xyz[:, 1] < self.y_max)
            & (xyz[:, 2] >= self.z_min)
            & (xyz[:, 2] < self.z_max)
        )
        xyz = xyz[valid]
        raw_features = raw_features[valid]
        batch = batch[valid]

        nx = int((self.x_max - self.x_min) / self.vx) + 1
        ny = int((self.y_max - self.y_min) / self.vy) + 1
        nz = int((self.z_max - self.z_min) / self.vz) + 1
        if max(nx, ny, nz) >= 1 << self.point_transformer.serial_depth:
            raise ValueError("LiDAR voxel grid exceeds the PTv3 serialization depth.")

        if xyz.numel():
            grid_coord = torch.stack(
                (
                    ((xyz[:, 0] - self.x_min) / self.vx).floor().long(),
                    ((xyz[:, 1] - self.y_min) / self.vy).floor().long(),
                    ((xyz[:, 2] - self.z_min) / self.vz).floor().long(),
                ),
                dim=-1,
            )
            keys = (((batch * nx + grid_coord[:, 0]) * ny + grid_coord[:, 1]) * nz) + grid_coord[:, 2]
            _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            num_voxels = int(inverse.max()) + 1

            counts = xyz.new_zeros(num_voxels)
            counts.index_add_(0, inverse, xyz.new_ones(xyz.shape[0]))
            voxel_coord = xyz.new_zeros(num_voxels, 3)
            voxel_feat = raw_features.new_zeros(num_voxels, 4)
            voxel_coord.index_add_(0, inverse, xyz)
            voxel_feat.index_add_(0, inverse, raw_features)
            voxel_coord = voxel_coord / counts.unsqueeze(-1)
            voxel_feat = voxel_feat / counts.unsqueeze(-1)

            voxel_batch = torch.empty(num_voxels, dtype=torch.long, device=device)
            voxel_batch[inverse] = batch
            voxel_grid = torch.empty(num_voxels, 3, dtype=torch.long, device=device)
            voxel_grid[inverse] = grid_coord
            is_real = torch.ones(num_voxels, dtype=torch.bool, device=device)
        else:
            voxel_coord = xyz.new_zeros(0, 3)
            voxel_feat = raw_features.new_zeros(0, 4)
            voxel_batch = torch.empty(0, dtype=torch.long, device=device)
            voxel_grid = torch.empty(0, 3, dtype=torch.long, device=device)
            is_real = torch.empty(0, dtype=torch.bool, device=device)

        # PTv3 expects at least one voxel in every batch item. Add masked-out
        # placeholders for empty clouds so the encoder can still serialize them.
        present = torch.bincount(voxel_batch, minlength=batch_size)
        empty_batches = torch.where(present == 0)[0]
        if empty_batches.numel():
            dummy_xyz = xyz.new_tensor(
                [self.x_min + self.vx / 2, self.y_min + self.vy / 2, self.z_min + self.vz / 2]
            ).expand(empty_batches.numel(), -1)
            voxel_coord = torch.cat([voxel_coord, dummy_xyz])
            dummy_feat = torch.cat([dummy_xyz, dummy_xyz.new_zeros(empty_batches.numel(), 1)], dim=-1)
            voxel_feat = torch.cat([voxel_feat, dummy_feat])
            voxel_batch = torch.cat([voxel_batch, empty_batches])
            voxel_grid = torch.cat([voxel_grid, torch.zeros(empty_batches.numel(), 3, dtype=torch.long, device=device)])
            is_real = torch.cat([is_real, torch.zeros(empty_batches.numel(), dtype=torch.bool, device=device)])

        return {
            "coord": voxel_coord,
            "feat": voxel_feat,
            "grid_coord": voxel_grid,
            "batch": voxel_batch,
        }, is_real

    def forward(self, points_by_sensor: dict[str, Tensor]) -> Tensor:
        if not points_by_sensor:
            raise ValueError("No point clouds were provided to TransformerModel.")

        transformed = []
        for sensor_name, points in points_by_sensor.items():
            if points.ndim != 3 or points.shape[-1] < 4:
                raise ValueError(f"Expected {sensor_name} points with shape (B, N, >=4).")
            transformed.append(self._to_vehicle(self.calibrations[sensor_name], points))
        points = torch.cat(transformed, dim=1)

        data, is_real = self._voxelize(points)
        encoded = self.point_transformer(data)
        features = self.feature_proj(encoded.feat[is_real])

        batch = encoded.batch[is_real]
        xyz = encoded.coord[is_real]
        ix = ((xyz[:, 0] - self.x_min) / (self.x_max - self.x_min) * self.out_w).long()
        iy = ((xyz[:, 1] - self.y_min) / (self.y_max - self.y_min) * self.out_h).long()
        valid = (ix >= 0) & (ix < self.out_w) & (iy >= 0) & (iy < self.out_h)

        flat_index = batch[valid] * (self.out_h * self.out_w) + iy[valid] * self.out_w + ix[valid]
        batch_size = points.shape[0]
        bev_flat = features.new_zeros(batch_size * self.out_h * self.out_w, features.shape[-1])
        counts = features.new_zeros(bev_flat.shape[0], 1)
        bev_flat.index_add_(0, flat_index, features[valid])
        counts.index_add_(0, flat_index, features.new_ones(flat_index.shape[0], 1))
        bev_flat = bev_flat / counts.clamp_min(1)
        return bev_flat.reshape(batch_size, self.out_h, self.out_w, -1).permute(0, 3, 1, 2).contiguous()


class _ViTCameraAdapter(nn.Module):
    """Project ViT outputs to the channel and skip-feature API expected downstream."""

    def __init__(self, params: MmpercParams, sensor_name: str, out_channels: int) -> None:
        super().__init__()
        depth = 12
        self.encoder = ViTCameraEncoder(params=params, sensor_name=sensor_name, depth=depth, extract_layers=[depth])
        self.out_channels = out_channels
        self.token_proj = nn.Linear(self.encoder.embed_dim, out_channels)
        self.feature_proj = nn.Conv2d(self.encoder.embed_dim, out_channels, kernel_size=1)
        self.s4_refine = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU())
        self.s2_refine = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU())
        self.register_buffer("cam_pose_vector", self.encoder.cam_pose_vector.detach().clone())

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        tokens, feat, _ = self.encoder(image)
        feat = self.feature_proj(feat)
        feat_h, feat_w = feat.shape[-2:]
        s4_size = (feat_h * 4, feat_w * 4)
        s2_size = (feat_h * 8, feat_w * 8)
        s4 = self.s4_refine(F.interpolate(feat, size=s4_size, mode="bilinear", align_corners=False))
        s2 = self.s2_refine(F.interpolate(feat, size=s2_size, mode="bilinear", align_corners=False))
        return self.token_proj(tokens), feat, {"s4": s4, "s2": s2}


class TransformerModel(nn.Module):
    """Multimodal model using PTv3 for LiDAR and ViT for camera features."""

    def __init__(self, params: MmpercParams) -> None:
        super().__init__()
        if not params.use_lidar and not params.use_camera:
            raise ValueError("At least one of LiDAR or camera must be enabled.")

        self._sensor_name = "front_center"
        channels = params.bev_params.bev_channels

        assert params.image_scale == 1.0, "Image scaling not support in ViT"

        if params.use_lidar:
            self.lidar_encoder = _PointTransformerBEVEncoder(params, [self._sensor_name])
        if params.use_camera:
            camera_encoder = _ViTCameraAdapter(params, self._sensor_name, channels)
            self.cam_encoder = MultiCameraEncoder({self._sensor_name: camera_encoder})

        self._use_fusion = params.use_lidar and params.use_camera
        if self._use_fusion:
            self.fusion = FuTrFusionBlock(params=params)

        if params.pred_bbox:
            self.bbox_head = BBox3dHead(channels)
        if params.pred_semantics:
            if not params.use_camera:
                raise ValueError("Semantic segmentation requires camera input.")
            self.sem_head = FullResSemHead(
                in_channels=self.cam_encoder.out_channels,
                num_classes=params.num_sem_classes,
                skip_channels=self.cam_encoder.out_channels,
            )
        self._params = params

    def forward(
        self,
        points: Tensor,
        image: Tensor,
        cam_meta: dict[str, dict] | None = None,
    ) -> dict[str, Tensor]:
        lidar_bev = None
        camera_tokens = None
        camera_feat = None
        camera_skips = None

        if self._params.use_lidar:
            lidar_bev = self.lidar_encoder({self._sensor_name: points})

        if self._params.use_camera:
            tokens, feats, skips = self.cam_encoder({self._sensor_name: image}, cam_meta)
            camera_tokens = tokens[self._sensor_name]
            camera_feat = feats[self._sensor_name]
            camera_skips = skips[self._sensor_name]

        if self._use_fusion:
            cam_hw = camera_feat.shape[-2:]
            bev_features = self.fusion(lidar_bev, camera_tokens, cam_hw=cam_hw)
        elif lidar_bev is not None:
            bev_features = lidar_bev
        else:
            bev_features = camera_feat

        outputs: dict[str, Tensor] = {}
        if hasattr(self, "bbox_head"):
            bbox = self.bbox_head(bev_features)
            outputs["bbox_heatmap"] = bbox["heatmap"]
            outputs["bbox_reg"] = bbox["reg"]
        if hasattr(self, "sem_head"):
            outputs["sem_logits"] = self.sem_head(camera_feat, camera_skips, out_size=image.shape[-2:])
        return outputs


def _smoke_test():
    """
    Smoke test for the model.
    """
    from components.definitions.mmperc_params import MmpercParams
    from components.utils.device import get_device

    device = get_device()

    params = MmpercParams()
    params.path_data = "./data/camera_lidar_semantic_bboxes.tar"
    params.image_scale = 1.0
    params.path_calibration = "./data/cams_lidars.json"
    model = TransformerModel(params=params).to(device)

    # Create dummy inputs
    B, N, H_img, W_img = 2, 1000, 256, 256
    points = torch.rand(B, N, 4).to(device)  # (B, N, 4)
    images = torch.rand(B, 3, H_img, W_img).to(device)  # (B, 3, H_img, W_img)

    # Forward pass
    outputs = model(points=points, image=images)

    # Print output shapes
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    print("TransformerModel smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
