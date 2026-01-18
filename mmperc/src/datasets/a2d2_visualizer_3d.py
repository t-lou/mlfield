"""
A2D2 NPZ Visualizer

Usage:
    python a2d2_visualizer.py batch_0001.npz  17
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# ================================================================
# 3D Box Construction
# ================================================================


def make_box_from_corners(corners, color=[1, 0, 0]):
    corners = np.array(corners)

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # verticals
    ]

    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(corners)
    box.lines = o3d.utility.Vector2iVector(edges)
    box.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return box


def create_3d_boxes_from_gt(gt_boxes: np.ndarray) -> list[o3d.geometry.LineSet]:
    """
    Convert (num_boxes, 7) array into Open3D LineSets.
    Format per box: [x, y, z, dx, dy, dz, yaw]
    """
    objects = []

    for box in gt_boxes:
        if np.allclose(box, 0):
            continue  # padded box

        x, y, z, dx, dy, dz, yaw = box

        # Compute 8 corners of the oriented box
        # A2D2 uses yaw around Z axis
        c, s = np.cos(yaw), np.sin(yaw)

        # Local corners before rotation
        local = np.array(
            [
                [-dx / 2, -dy / 2, -dz / 2],
                [dx / 2, -dy / 2, -dz / 2],
                [dx / 2, dy / 2, -dz / 2],
                [-dx / 2, dy / 2, -dz / 2],
                [-dx / 2, -dy / 2, dz / 2],
                [dx / 2, -dy / 2, dz / 2],
                [dx / 2, dy / 2, dz / 2],
                [-dx / 2, dy / 2, dz / 2],
            ]
        )

        # Rotation matrix around Z
        R = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )

        corners = (R @ local.T).T + np.array([x, y, z])

        objects.append(make_box_from_corners(corners, color=[1, 0, 0]))

    return objects


# ================================================================
# Visualizer Class
# ================================================================


class A2D2Visualizer3D:
    def __init__(
        self,
        window_name="A2D2 3D Viewer",
        width=1280,
        height=720,
        point_size=2.0,
        line_width=1.0,
        bg_color=(0.05, 0.05, 0.05),
    ):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width, height)

        opt = self.vis.get_render_option()
        opt.background_color = bg_color
        opt.point_size = point_size
        opt.line_width = line_width

        self.geometries = []

    def add(self, geom):
        self.vis.add_geometry(geom)
        self.geometries.append(geom)

    def add_geometries(self, geoms):
        for g in geoms:
            self.add(g)

    def run(self):
        self.vis.run()

    def close(self):
        self.vis.destroy_window()


# ================================================================
# High-level NPZ Visualizer
# ================================================================


def visualize_npz_frame(npz_path: Path, frame_idx: int):
    """
    Load a single frame from an NPZ chunk and visualize it.
    """

    npz = np.load(npz_path)

    # Extract frame
    points = npz["points"][frame_idx][:, :3]  # XYZ only
    gt_boxes = npz["gt_boxes"][frame_idx]

    print(f"Loaded frame {frame_idx} from {npz_path}")
    print(f"Points shape: {points.shape}")
    print(f"GT boxes shape: {gt_boxes.shape}")

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    # Build 3D boxes
    boxes = create_3d_boxes_from_gt(gt_boxes)

    # Visualize
    viz = A2D2Visualizer3D(point_size=3, line_width=2)
    viz.add_geometries([pcd] + boxes)
    viz.run()
    viz.close()


# ================================================================
# CLI Entry Point
# ================================================================

if __name__ == "__main__":
    npz_path = Path(sys.argv[1])
    frame_idx = int(sys.argv[2])

    assert npz_path.is_file(), f"NPZ file not found: {npz_path}"

    visualize_npz_frame(npz_path, frame_idx)
