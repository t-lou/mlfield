"""
Shared geometric parameters for the entire project.

These values define:
- the 3D region of interest for lidar voxelization
- the voxel size used by PointPillars
- the resulting BEV grid resolution (bev_h, bev_w)
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BevParams:
    """
    Parameters for BEV grid and voxelization.
    """

    # -----------------------------
    # Lidar region of interest (meters)
    # -----------------------------
    x_range: Tuple[float, float] = (0.0, 120.0)  # forward/backward
    y_range: Tuple[float, float] = (-60.0, 60.0)  # left/right
    z_range: Tuple[float, float] = (-5.0, 3.0)  # vertical range

    # Combined point cloud range (x_min, y_min, z_min, x_max, y_max, z_max)
    pc_range: Tuple[float, float, float, float, float, float] = (
        x_range[0],
        y_range[0],
        z_range[0],
        x_range[1],
        y_range[1],
        z_range[1],
    )

    # -----------------------------
    # Derived BEV grid resolution
    # -----------------------------
    bev_h: int = 156
    bev_w: int = 156

    # backbone downsampling factor
    backbone_stride: int = 2

    # -----------------------------
    # Voxel size (meters)
    # -----------------------------
    voxel_size: Tuple[float, float, float] = (
        (x_range[1] - x_range[0]) / bev_w,
        (y_range[1] - y_range[0]) / bev_h,
        z_range[1] - z_range[0],  # vz (full height, pillar)
    )

    bev_channels: int = 128  # Number of channels in BEV feature map
