#!/usr/bin/env python3
"""Minimal PCD viewer based on pypcd4 + pyvista.

Requires only x, y, z in the file. If an `intensity` field exists it is used
for coloring, otherwise points are colored by height (z).

Usage:
    python vis_pcd.py cloud.pcd
    python vis_pcd.py a.pcd b.pcd --point-size 3 --cmap viridis
    python vis_pcd.py cloud.pcd --screenshot out.png      # off-screen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from pypcd4 import PointCloud


def load_pcd(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return finite XYZ points and an optional finite intensity array."""
    if not path.is_file():
        raise FileNotFoundError(f"{path}: file does not exist")

    pc = PointCloud.from_path(str(path))

    if not all(f in pc.fields for f in ("x", "y", "z")):
        raise ValueError(f"{path}: missing x/y/z fields (found {pc.fields})")

    has_intensity = "intensity" in pc.fields
    names = ("x", "y", "z", "intensity") if has_intensity else ("x", "y", "z")
    data = pc.numpy(names).astype(np.float32, copy=False)
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(f"{path}: unexpected point-cloud data shape {data.shape}")

    # Organized clouds often contain NaN placeholders.
    mask = np.isfinite(data[:, :3]).all(axis=1)
    data = data[mask]

    xyz = np.ascontiguousarray(data[:, :3])
    intensity = data[:, 3] if has_intensity else None
    if intensity is not None:
        intensity = np.ascontiguousarray(intensity)
        intensity[~np.isfinite(intensity)] = np.nan
    return xyz, intensity


def build_mesh(xyz: np.ndarray, intensity: np.ndarray | None) -> tuple[pv.PolyData, str]:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"invalid XYZ shape: {xyz.shape}")
    if len(xyz) == 0:
        raise ValueError("cannot build a mesh without points")

    mesh = pv.PolyData(xyz)
    if intensity is not None and np.isfinite(intensity).any():
        mesh["intensity"] = intensity
        return mesh, "intensity"
    mesh["z"] = xyz[:, 2]
    return mesh, "z"


def positive_float(value: str) -> float:
    """Argparse type for values that must be greater than zero."""
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value}") from exc
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than zero")
    return result


def scalar_range(value: str) -> tuple[float, float]:
    """Argparse type for a pair of scalar limits."""
    try:
        low, high = (float(part) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected MIN,MAX") from exc
    if not np.isfinite([low, high]).all():
        raise argparse.ArgumentTypeError("limits must be finite numbers")
    if low >= high:
        raise argparse.ArgumentTypeError("MIN must be smaller than MAX")
    return low, high


def main() -> int:
    ap = argparse.ArgumentParser(description="View PCD point clouds with pyvista.")
    ap.add_argument("files", nargs="+", type=Path, help="one or more .pcd files")
    ap.add_argument("--point-size", type=positive_float, default=2.0)
    ap.add_argument("--cmap", default="turbo", help="matplotlib colormap name")
    ap.add_argument("--background", default="black")
    ap.add_argument("--no-axes", action="store_true", help="hide orientation axes")
    ap.add_argument("--screenshot", type=Path, help="save image and exit (off-screen)")
    ap.add_argument(
        "--scalar-range",
        type=scalar_range,
        metavar="MIN,MAX",
        help="fixed scalar range for coloring",
    )
    args = ap.parse_args()

    plotter = pv.Plotter(off_screen=args.screenshot is not None)
    try:
        plotter.set_background(args.background)

        total = 0
        for path in args.files:
            try:
                xyz, intensity = load_pcd(path)
                if len(xyz) == 0:
                    print(f"skip {path}: no finite points", file=sys.stderr)
                    continue
                mesh, scalars = build_mesh(xyz, intensity)
            except Exception as exc:
                print(f"skip {path}: {exc}", file=sys.stderr)
                continue

            scalar_kwargs = {}
            if args.scalar_range is not None:
                scalar_kwargs["clim"] = args.scalar_range

            plotter.add_mesh(
                mesh,
                scalars=scalars,
                cmap=args.cmap,
                point_size=args.point_size,
                render_points_as_spheres=False,
                name=str(path),
                show_scalar_bar=True,
                scalar_bar_args={"title": scalars},
                **scalar_kwargs,
            )
            total += len(xyz)
            print(f"{path}: {len(xyz):,} points, colored by {scalars}")

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
