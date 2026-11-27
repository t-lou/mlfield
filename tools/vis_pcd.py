#!/usr/bin/env python3
"""Minimal PCD viewer based on pypcd4 + pyvista.

Requires only x, y, z in the file. If an `intensity` field exists it is used
for coloring, otherwise points are colored by height (z).

Usage:
    python vis_pcd.py cloud.pcd --point-size 3 --cmap viridis
    python vis_pcd.py --point-size 3 --cmap viridis  # open a file dialog
    python vis_pcd.py cloud.pcd --screenshot out.png      # off-screen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from pypcd4 import PointCloud


def load_pcd(path: Path, attribute: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load finite XYZ coordinates and optional intensity values from a PCD.

    Args:
        path: Path to the PCD file to load.
        attribute: The attribute to use for coloring.

    Returns:
        A tuple containing a contiguous ``(N, 3)`` float32 array and either a
        contiguous ``(N,)`` intensity array or ``None``. Non-finite XYZ rows
        (for example, placeholders in organized point clouds) are removed.

    Raises:
        FileNotFoundError: If ``path`` is not a regular file.
        ValueError: If the PCD has no x/y/z fields or returns invalid data.
        Exception: Other failures raised while decoding the PCD are left to the
            caller so they can include file-specific context.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: file does not exist")

    try:
        pc = PointCloud.from_path(str(path))
    except Exception as exc:
        raise ValueError(f"{path}: failed to read point cloud: {exc}") from exc

    if not all(field in pc.fields for field in ("x", "y", "z")):
        raise ValueError(f"{path}: missing x/y/z fields (found {pc.fields})")

    names = ("x", "y", "z")
    has_attribute = attribute in pc.fields
    if has_attribute:
        names += (attribute,)

    x = pc["x"].numpy().flatten()
    y = pc["y"].numpy().flatten()
    z = pc["z"].numpy().flatten()

    # Organized clouds often contain NaN placeholders.
    finite_xyz = np.isfinite(np.column_stack([x, y, z])).all(axis=1)
    x = x[finite_xyz]
    y = y[finite_xyz]
    z = z[finite_xyz]

    xyz = np.column_stack([x, y, z]).astype(np.float32)
    intensity = pc[attribute].numpy().flatten() if has_attribute else None
    if intensity is not None:
        intensity = np.ascontiguousarray(intensity)
        intensity[~np.isfinite(intensity)] = np.nan
    return xyz, intensity


def build_mesh(xyz: np.ndarray, intensity: np.ndarray | None) -> tuple[pv.PolyData, str]:
    """Build a PyVista point mesh and select a scalar field.

    Args:
        xyz: Point coordinates with shape ``(N, 3)``.
        intensity: Optional intensity values with shape ``(N,)``. If the
            array has no finite values, height is used instead.

    Returns:
        A pair of the constructed mesh and the name of its scalar field.

    Raises:
        ValueError: If XYZ or intensity has an invalid shape or size.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"invalid XYZ shape: {xyz.shape}")
    if len(xyz) == 0:
        raise ValueError("cannot build a mesh without points")
    if not np.isfinite(xyz).all():
        raise ValueError("XYZ coordinates must contain only finite values")

    xyz = np.ascontiguousarray(xyz, dtype=np.float32)

    mesh = pv.PolyData(xyz)
    if intensity is not None:
        intensity = np.asarray(intensity)
        if intensity.ndim != 1 or len(intensity) != len(xyz):
            raise ValueError(f"invalid intensity shape: {intensity.shape}; expected ({len(xyz)},)")
        intensity = np.ascontiguousarray(intensity, dtype=np.float32)
        if np.isfinite(intensity).any():
            mesh["intensity"] = intensity
            return mesh, "intensity"
    mesh["z"] = xyz[:, 2]
    return mesh, "z"


def positive_float(value: str) -> float:
    """Parse an argparse value that must be finite and greater than zero.

    Args:
        value: String representation of a floating-point number.

    Returns:
        The validated positive number.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a finite positive number.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value}") from exc
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than zero")
    return result


def scalar_range(value: str) -> tuple[float, float]:
    """Parse an argparse scalar range in ``MIN,MAX`` form.

    Args:
        value: Comma-separated lower and upper limits.

    Returns:
        The validated ``(minimum, maximum)`` tuple.

    Raises:
        argparse.ArgumentTypeError: If the value is malformed or its limits are
            non-finite or not strictly increasing.
    """
    try:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError
        low, high = (float(part.strip()) for part in parts)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected MIN,MAX") from exc
    if not np.isfinite([low, high]).all():
        raise argparse.ArgumentTypeError("limits must be finite numbers")
    if low >= high:
        raise argparse.ArgumentTypeError("MIN must be smaller than MAX")
    return low, high


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PCD viewer.

    Returns:
        Parsed arguments, including input paths and PyVista display options.
    """
    ap = argparse.ArgumentParser(description="View PCD point clouds with pyvista.")
    ap.add_argument("file", nargs="?", type=Path, help="a .pcd file (opens a file dialog if omitted)")
    ap.add_argument(
        "--point-size",
        type=positive_float,
        default=2.0,
        help="rendered point size (default: 2)",
    )
    ap.add_argument("--attribute", default="intensity", help="PCD field to use for coloring (default: intensity)")
    ap.add_argument("--cmap", default="turbo", help="matplotlib colormap name (default: turbo)")
    ap.add_argument("--background", default="black", help="plotter background color (default: black)")
    ap.add_argument("--no-axes", action="store_true", help="hide orientation axes")
    ap.add_argument("--screenshot", type=Path, help="save image and exit (off-screen)")
    ap.add_argument(
        "--scalar-range",
        type=scalar_range,
        metavar="MIN,MAX",
        help="fixed scalar range for coloring",
    )
    return ap.parse_args()


def main() -> int:
    """Load all requested clouds, render them, and return an exit status.

    Invalid individual files are reported to stderr and skipped. The process
    exits with status 1 only when no valid point-cloud data can be displayed.

    Returns:
        Zero on success, or one if no finite points were loaded.
    """
    args = parse_args()

    plotter = pv.Plotter(off_screen=args.screenshot is not None)
    try:
        plotter.set_background(args.background)

        total = 0
        if args.file is None:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                selected_file = filedialog.askopenfilename(
                    title="Open PCD/PLY file",
                    filetypes=[("PCD files", "*.pcd"), ("PLY files", "*.ply"), ("All files", "*")],
                )
            finally:
                root.destroy()
            if not selected_file:
                print("no PCD file selected", file=sys.stderr)
                return 1
            file_path = Path(selected_file)
        else:
            file_path = args.file

        try:
            xyz, intensity = load_pcd(file_path, args.attribute)
            if len(xyz) == 0:
                print(f"skip {file_path}: no finite points", file=sys.stderr)
                return 1
            mesh, scalars = build_mesh(xyz, intensity)
        except Exception as exc:
            print(f"skip {file_path}: {exc}", file=sys.stderr)
            return 1

        scalar_kwargs = {}
        if args.scalar_range is not None:
            scalar_kwargs["clim"] = args.scalar_range

        plotter.add_mesh(
            mesh,
            scalars=scalars,
            cmap=args.cmap,
            point_size=args.point_size,
            render_points_as_spheres=False,
            name=str(file_path),
            show_scalar_bar=True,
            scalar_bar_args={"title": scalars},
            **scalar_kwargs,
        )
        total += len(xyz)
        print(f"{file_path}: {len(xyz):,} points, colored by {scalars}")

        if total == 0:
            print("nothing to show", file=sys.stderr)
            return 1

        if not args.no_axes:
            plotter.add_axes()
        plotter.reset_camera()

        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            plotter.show(screenshot=str(args.screenshot))
            print(f"saved screenshot: {args.screenshot}")
        else:
            plotter.show(title="PCD Viewer")
        return 0
    finally:
        plotter.close()


if __name__ == "__main__":
    sys.exit(main())
