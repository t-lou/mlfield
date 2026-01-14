"""
Shared geometric parameters for the entire project.

These values define:
- the 3D region of interest for lidar voxelization
- the voxel size used by PointPillars
- the resulting BEV grid resolution (BEV_H, BEV_W)
"""

BEV_CHANNELS = 128  # Number of channels in BEV feature map

# -----------------------------
# Lidar region of interest (meters)
# -----------------------------
X_RANGE: tuple[float, float] = (-50.0, 50.0)  # forward/backward
Y_RANGE: tuple[float, float] = (-50.0, 50.0)  # left/right
Z_RANGE: tuple[float, float] = (-5.0, 3.0)  # vertical range

# -----------------------------
# Voxel size (meters)
# -----------------------------
# (vx, vy, vz)
VOXEL_SIZE: tuple[float, float, float] = (0.32, 0.32, 8.0)

# -----------------------------
# Derived BEV grid resolution
# -----------------------------
# BEV_W = (x_max - x_min) / voxel_size_x
# BEV_H = (y_max - y_min) / voxel_size_y
# These define the spatial resolution of the BEV feature map.
# -----------------------------
BEV_W: int = int((X_RANGE[1] - X_RANGE[0]) / VOXEL_SIZE[0])
BEV_H: int = int((Y_RANGE[1] - Y_RANGE[0]) / VOXEL_SIZE[1])

# Optional: print for debugging when running this file directly
if __name__ == "__main__":
    print(f"BEV grid resolution: H={BEV_H}, W={BEV_W}")
