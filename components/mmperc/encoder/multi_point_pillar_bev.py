from pathlib import Path

import torch
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams
from components.mmperc.backbone.tiny_bev_backbone import TinyBEVBackbone
from components.mmperc.encoder.point_pillar_bev import PointPillarBEV
from components.mmperc.encoder.simple_pfn import SimplePFN
from components.mmperc.scatter.scatter import scatter_to_bev
from components.mmperc.voxelizer.pointpillar_lite import PointpillarLite
from components.utils.calibration import load_sensor_calibration
from components.utils.logger import logger


class MultiPointPillarBEV(nn.Module):
    """
    Multi-sensor LiDAR encoder.

    It keeps each sensor in its own local calibration, transforms all points into the
    vehicle frame, concatenates them, and then runs the same voxelization/PFN/scatter
    stack used by the single-sensor PointPillarBEV.
    """

    def __init__(self, params: MmpercParams, sensor_names: list[str]) -> None:
        super().__init__()

        self.sensor_names = list(sensor_names)

        self.calibrations = {
            sensor_name: load_sensor_calibration(
                Path(params.path_calibration),
                sensor_name=sensor_name,
                sensor_type="lidar",
            )
            for sensor_name in self.sensor_names
        }

        # Shared voxelization + backbone pipeline across all sensors.
        self.voxelizer = PointpillarLite(params=params)
        self.pfn = SimplePFN(in_channels=9, out_channels=64)
        self.backbone = TinyBEVBackbone(params=params, out_channels=params.bev_params.bev_channels)
        self.bev_h = params.bev_params.bev_h
        self.bev_w = params.bev_params.bev_w

    def _transform_points_to_vehicle(self, points: Tensor, sensor_name: str) -> Tensor:
        """Use the shared point transform logic for each sensor frame."""
        return PointPillarBEV.transform_points_to_vehicle(self.calibrations[sensor_name], points)

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
        transformed: list[Tensor] = []

        for sensor_name in self.sensor_names:
            if sensor_name not in points_by_sensor:
                raise KeyError(f"Missing point cloud for sensor '{sensor_name}'.")
            points = points_by_sensor[sensor_name]
            transformed.append(self._transform_points_to_vehicle(points, sensor_name=sensor_name))

        if not transformed:
            raise ValueError("No point clouds were provided to MultiPointPillarBEV.")

        merged_points = torch.cat(transformed, dim=1)
        return self._encode_points(merged_points)


def _smoke_test():
    params = MmpercParams()
    model = MultiPointPillarBEV(params=params, sensor_names=["front_center", "front_left"])

    B = 2
    points = {
        "front_center": torch.rand((B, 1000, 5)),
        "front_left": torch.rand((B, 1000, 5)),
    }
    out = model(points)
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    _smoke_test()
