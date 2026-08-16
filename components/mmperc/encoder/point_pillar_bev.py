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

    def __init__(self, params: MmpercParams, sensor_name: str = "front_center") -> None:
        super().__init__()

        # Raw point cloud → pillars
        self.voxelizer = PointpillarLite(params=params)

        # Single-sensor calibration relative to the vehicle frame.
        self.lidar_calibration = load_sensor_calibration(
            Path(params.path_calibration),
            sensor_name=sensor_name,
            sensor_type="lidar",
        )

        # Pillar Feature Network (per-pillar feature extraction)
        self.pfn = SimplePFN(in_channels=9, out_channels=64)

        # BEV backbone (expands 64 → params.BEV_CHANNELS)
        self.backbone = TinyBEVBackbone(params=params, out_channels=params.bev_params.bev_channels)

        # Precomputed BEV grid resolution
        self.bev_h = params.bev_params.bev_h
        self.bev_w = params.bev_params.bev_w

    def forward(self, points: Tensor) -> Tensor:
        """
        Args:
            points: (B, N, 5)
                Raw lidar points: x, y, z, intensity, timestamp

        Returns:
            BEV feature map: (B, params.BEV_CHANNELS, H/2, W/2)
        """

        # Convert each LiDAR point from its sensor frame into the vehicle frame.
        points = self.transform_points_to_vehicle(self.lidar_calibration, points)

        # 1. Voxelization
        vox = self.voxelizer(points)
        pillars = vox["pillars"]  # (B, P, M, C_in)
        pillar_coords = vox["pillar_coords"]  # (B, P, 2)
        logger.debug(f"pillars.shape: {pillars.shape}, pillar_coords.shape: {pillar_coords.shape}")

        # 2. PFN → per-pillar features
        pillar_feats = self.pfn(pillars)  # (B, P, 64)
        logger.debug(f"pillar_feats.shape: {pillar_feats.shape}")

        # 3. Scatter to BEV grid
        bev = scatter_to_bev(
            pillar_feats,
            pillar_coords,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
        )  # (B, 64, H, W)
        logger.debug(f"bev.shape: {bev.shape}")

        # 4. BEV backbone, downsampling H/2, W/2
        bev_backbone = self.backbone(bev)
        logger.debug(f"bev_backbone.shape: {bev_backbone.shape}")

        return bev_backbone


def _smoke_test():
    """
    Smoke test for PointPillarBEV
    """
    params = MmpercParams()
    model = PointPillarBEV(params=params)

    # Random input: (B, N, 5)
    points = torch.rand((2, 1000, 5))
    output = model(points)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    _smoke_test()
