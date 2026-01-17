"""
a2d2_visualizer.py

A clean, extensible 3D visualizer for A2D2 LiDAR scenes.
Supports:
- Loading LiDAR point clouds (.npy)
- Loading 3D bounding boxes from A2D2 JSON
- Rendering point clouds + 3D boxes
- Adjustable point size, line width, background color
- Easy extension for keyboard controls or animation
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# ================================================================
# Utility: Find a single file by extension + optional partial name
# ================================================================


def find_only_file_with_ext_and_partial_name(
    directory: Path,
    extensions: str | list[str] | tuple[str, ...],
    partial_name: str | list[str] | tuple[str, ...] | None = None,
    recursive: bool = True,
) -> Path:
    """
    Search for exactly one file in `directory` (optionally recursive) matching:
      - one or more extensions
      - optional partial name(s)

    Returns:
        Path to the unique matching file.

    Raises:
        ValueError if zero or multiple files match.
    """

    # Normalize extensions
    if isinstance(extensions, str):
        extensions = [extensions]

    # Build glob patterns
    patterns = [str(directory / "**" / f"*.{ext}") if recursive else str(directory / f"*.{ext}") for ext in extensions]

    # Collect matching files
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=recursive))

    # Normalize partial name(s)
    if partial_name is not None:
        if isinstance(partial_name, str):
            partials = [partial_name]
        else:
            partials = list(partial_name)

        # Filter: must contain all partial substrings
        for p in partials:
            files = [f for f in files if p in f]

    # Enforce exactly one match
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one file with extensions {extensions} in {directory}, "
            f"found {len(files)} after filtering with {partial_name}"
        )

    return Path(files[0])


def load_lidar_points(path: Path, partial_name: str | None = None) -> np.ndarray:
    """
    Loads LiDAR points from either .npy or .npz.
    Returns Nx3 XYZ points.
    """

    lidar_path = find_only_file_with_ext_and_partial_name(path, ["npy", "npz"], partial_name)
    if lidar_path.suffix == ".npy":
        data = np.load(lidar_path)
        raise ValueError(
            f"Loaded LiDAR data in single array, please check contents: shape={data.shape}, dtype={data.dtype}"
        )
    elif lidar_path.suffix == ".npz":
        npz = np.load(lidar_path)
        assert "points" in npz, f"NPZ contains keys: {[k for k in npz.files]}"
        data = npz["points"]
    else:
        raise ValueError(f"Unsupported LiDAR file extension: {lidar_path.suffix}")

    # Ensure Nx3
    print(f"Loaded LiDAR data shape: {data.shape}")
    if data.shape[1] >= 3:
        return data[:, :3]

    raise ValueError(f"Loaded LiDAR data has invalid shape {data.shape}, expected at least 3 columns")


def load_labels(path: Path, partial_name: str | None = None) -> dict:
    """
    Loads the A2D2 JSON label file.
    """
    label_partial_name = ["label"] + ([partial_name] if partial_name else [])
    json_path = find_only_file_with_ext_and_partial_name(path, "json", label_partial_name)
    with open(json_path, "r") as f:
        return json.load(f)


# ================================================================
# 3D Box Construction
# ================================================================


def make_box_from_corners(corners, color=[1, 0, 0]):
    """
    Create an Open3D LineSet from 8 corner points.
    corners: list of 8 (x, y, z)
    """
    corners = np.array(corners)

    # 12 edges of a cuboid
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


def create_3d_boxes_from_labels(labels: dict) -> list[o3d.geometry.LineSet]:
    """
    Convert A2D2 JSON labels into Open3D LineSets.
    """
    objects = []

    for key, box in labels.items():
        corners = box["3d_points"]
        cls = box["class"]

        # Class‑based colors
        color = {
            "Pedestrian": [1, 0, 0],  # red
            "Car": [0, 1, 0],  # green
            "Truck": [0, 0, 1],  # blue
        }.get(cls, [1, 1, 0])  # default: yellow

        objects.append(make_box_from_corners(corners, color))

    return objects


# ================================================================
# Visualizer Class
# ================================================================


class A2D2Visualizer3D:
    """
    A general-purpose 3D visualizer for LiDAR scenes.

    Usage:
        viz = LidarSceneVisualizer(point_size=3, line_width=2)
        viz.add_geometries([pcd] + boxes)
        viz.run()
    """

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

        self.render_opt = self.vis.get_render_option()
        self.render_opt.background_color = bg_color
        self.render_opt.point_size = point_size
        self.render_opt.line_width = line_width

        self.geometries = []

    # ---------------- Geometry Management ----------------

    def add(self, geom):
        self.vis.add_geometry(geom)
        self.geometries.append(geom)

    def add_geometries(self, geoms):
        for g in geoms:
            self.add(g)

    # ---------------- Rendering Controls ----------------

    def set_point_size(self, size: float):
        self.render_opt.point_size = size

    def set_line_width(self, width: float):
        self.render_opt.line_width = width

    def set_background(self, color):
        self.render_opt.background_color = color

    def update(self):
        for g in self.geometries:
            self.vis.update_geometry(g)
        self.vis.poll_events()
        self.vis.update_renderer()

    # ---------------- Main Loop ----------------

    def run(self):
        self.vis.run()

    def close(self):
        self.vis.destroy_window()


# ================================================================
# High-level Scene Visualizer
# ================================================================


def visualize_scene(path: Path, partial_name: str | None = None):
    """
    Loads LiDAR + labels from a directory and visualizes them.
    """
    points = load_lidar_points(path, partial_name)
    labels = load_labels(path, partial_name)

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    # Build 3D boxes
    boxes = create_3d_boxes_from_labels(labels)

    # Visualize
    viz = A2D2Visualizer3D(point_size=3, line_width=2)
    viz.add_geometries([pcd] + boxes)
    viz.run()
    viz.close()


# ================================================================
# CLI Entry Point
# ================================================================

if __name__ == "__main__":
    scene_path = Path(sys.argv[1])
    assert scene_path.is_dir(), f"Provided path {scene_path} is not a directory"
    visualize_scene(scene_path)
