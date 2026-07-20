"""
Shared geometric parameters for the entire project.

These values define:
- the 3D region of interest for lidar voxelization
- the voxel size used by PointPillars
- the resulting BEV grid resolution (BEV_H, BEV_W)
"""

PATH_TRAIN = "/workspace/mmperc/data/a2d2/training"
PATH_EVAL = "/workspace/mmperc/data/a2d2/evaluation"
PATH_VALID = "/workspace/mmperc/data/a2d2/validation"

# -----------------------------
# Number of semantic classes
# -----------------------------
NUM_SEM_CLASSES = 38

# Whether to activate debug plotting
DEBUG_PLOT_ON = True

# Optional: print for debugging when running this file directly
