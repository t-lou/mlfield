from pathlib import Path

import torch
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams
from components.mmperc.backbone.tiny_bev_backbone import TinyBEVBackbone
from components.mmperc.encoder.simple_pfn import SimplePFN
from components.mmperc.scatter.scatter import scatter_to_bev
from components.mmperc.voxelizer.pointpillar_lite import PointpillarLite
from components.utils.calibration import load_sensor_calibration
from components.utils.logger import logger
from components.vit.position_embedding import PosEmbdCache


class BEVTokenizer(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self._pos_cache = PosEmbdCache()  # keyed by grid size, same idea as TinyCameraEncoder._camera_geometry_cache

    def forward(self, bev_feat):  # (B, C, H, W)
        x = self.proj(bev_feat)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        pos = self._pos_cache.get_2d(H, W, C, device=x.device, dtype=x.dtype)  # sincos, no cls
        tokens = self.norm(tokens + pos)
        return tokens


class PointPillarBEV(nn.Module):
    """
    Full lidar → BEV encoder:
        - voxelization (PointPillars-style)
        - PFN feature extraction
        - scatter to BEV grid
        - lightweight BEV backbone

    Output:
        (B, params.BEV_CHANNELS, BEV_H/2, BEV_W/2)
    """

    @staticmethod
    def transform_points_to_vehicle(calibration, points: Tensor) -> Tensor:
        """Apply the sensor-to-vehicle frame transform while preserving feature channels."""
        if points.dim() == 2:
            points = points.unsqueeze(0)

        if points.size(-1) < 3:
            raise ValueError("Point cloud must contain x, y, z as the first 3 columns.")

        xyz = points[..., :3]
        ones = torch.ones(*xyz.shape[:-1], 1, device=points.device, dtype=points.dtype)
        xyz_h = torch.cat([xyz, ones], dim=-1)
        vehicle_T = torch.as_tensor(
            calibration.pose.vehicle_from_sensor,
            device=points.device,
            dtype=points.dtype,
        )
        xyz_vehicle_h = xyz_h @ vehicle_T.T
        return torch.cat([xyz_vehicle_h[..., :3], points[..., 3:]], dim=-1)

    def __init__(self, params: MmpercParams, sensor_names: list[str]) -> None:
        super().__init__()

        self.sensor_names = list(sensor_names)

        self.calibrations = {
            sensor_name: load_sensor_calibration(
                Path(params.path_calibration),
                sensor_name=sensor_name,
                sensor_type="camera",  # in a2d2 dataset lidar data seems to be projected to camera
            )
            for sensor_name in self.sensor_names
        }

        # Shared voxelization + backbone pipeline across all sensors.
        self.voxelizer = PointpillarLite(params=params)
        self.pfn = SimplePFN(in_channels=9, out_channels=64)
        self.backbone = TinyBEVBackbone(params=params, out_channels=params.bev_params.bev_channels)
        self.bev_h = params.bev_params.bev_h
        self.bev_w = params.bev_params.bev_w

    def _encode_points(self, points: Tensor) -> Tensor:
        """Voxelize, encode, and project the merged LiDAR points into a BEV feature map."""
        vox = self.voxelizer(points)
        pillars = vox["pillars"]
        pillar_coords = vox["pillar_coords"]
        logger.debug(f"pillars.shape: {pillars.shape}, pillar_coords.shape: {pillar_coords.shape}")

        pillar_feats = self.pfn(pillars)
        logger.debug(f"pillar_feats.shape: {pillar_feats.shape}")

        bev = scatter_to_bev(
            pillar_feats,
            pillar_coords,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
        )
        logger.debug(f"bev.shape: {bev.shape}")

        bev_backbone = self.backbone(bev)
        logger.debug(f"bev_backbone.shape: {bev_backbone.shape}")
        return bev_backbone

    def forward(self, points_by_sensor: dict[str, Tensor]) -> Tensor:
        """
        Args:
            points_by_sensor: {
                "front_center": Tensor[(B, N, 5)],
                "front_left": Tensor[(B, N, 5)],
                ...
            }

        Returns:
            BEV feature map: (B, params.BEV_CHANNELS, H/2, W/2)
        """
        transformed: list[Tensor] = [
            self.transform_points_to_vehicle(calibration=self.calibrations[sensor_name], points=points)
            for sensor_name, points in points_by_sensor.items()
        ]

        if not transformed:
            raise ValueError("No point clouds were provided to MultiPointPillarBEV.")

        merged_points = torch.cat(transformed, dim=1)
        return self._encode_points(merged_points)


def _smoke_test():
    params = MmpercParams()
    model = PointPillarBEV(params=params, sensor_names=["front_center", "front_left"])

    B = 2
    points = {
        "front_center": torch.rand((B, 1000, 5)),
        "front_left": torch.rand((B, 1000, 5)),
    }
    out = model(points)
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    _smoke_test()
