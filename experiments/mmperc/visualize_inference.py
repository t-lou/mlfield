#!/usr/bin/env python3
"""
Visualize a single inference-output .npz file (as produced by
ModelInferenceWrapper.infer_a2d2_dataset in decode_a2d2.py) in either:

  - 2D: matplotlib top-down (BEV / XY) plot of the point cloud with
    ground-truth and predicted boxes drawn as rectangles (+ a semantic BEV
    panel next to it, if the .npz has one).
  - 3D: a PyVista window with the point cloud and 3D wireframe boxes.

Usage:
    python visualize_inference.py sample_000000.npz --mode 2d
    python visualize_inference.py sample_000000.npz --mode 3d
    python visualize_inference.py sample_000000.npz --mode 2d --score-thresh 0.5 --output bev.png

Requires: numpy, matplotlib (2D mode), pyvista (3D mode).

This is the inference-side counterpart to visualize_dataset.py; geometry and
plotting primitives shared by both live in viz_common.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from components.mmperc.common.visualization import (
    add_boxes_3d,
    add_point_cloud,
    as_boxes,
    filter_pred_boxes,
    plot_bev,
    semantic_ids_to_rgb,
)


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ================================================================
# 2D (matplotlib)
# ================================================================


def visualize_2d(data: dict, score_thresh: float, output: Path | None) -> None:
    import matplotlib.pyplot as plt

    points = np.asarray(data["points"])[:, :3]
    gt_boxes = as_boxes(data.get("gt_boxes"))
    pred_boxes = filter_pred_boxes(as_boxes(data.get("pred_boxes")), data.get("pred_scores"), score_thresh)

    has_sem = "sem_pred" in data and "semantics_mapping_color" in data
    fig, axes = plt.subplots(1, 2 if has_sem else 1, figsize=(16 if has_sem else 9, 8))
    ax = axes[0] if has_sem else axes

    plot_bev(
        ax,
        points,
        box_sets=[(gt_boxes, "limegreen", "GT"), (pred_boxes, "red", "Pred")],
        title=f"BEV — points={len(points)}, gt={len(gt_boxes)}, pred={len(pred_boxes)}",
    )

    if has_sem:
        sem_pred = np.asarray(data["sem_pred"])
        sem_rgb = semantic_ids_to_rgb(sem_pred, data["semantics_mapping_color"])
        axes[1].imshow(sem_rgb, origin="lower")
        axes[1].set_title("Semantic BEV prediction")
        axes[1].axis("off")

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved 2D visualization to {output}")
    else:
        plt.show()


# ================================================================
# 3D (PyVista)
# ================================================================


def visualize_3d(data: dict, score_thresh: float) -> None:
    try:
        import pyvista as pv
    except ImportError as e:
        raise SystemExit("3D mode requires pyvista. Install it with: pip install pyvista") from e

    points = np.asarray(data["points"])[:, :3]
    gt_boxes = as_boxes(data.get("gt_boxes"))
    pred_boxes = filter_pred_boxes(as_boxes(data.get("pred_boxes")), data.get("pred_scores"), score_thresh)

    plotter = pv.Plotter()
    add_point_cloud(plotter, points)
    plotter.add_axes_at_origin(labels_off=True, line_width=3)
    add_boxes_3d(plotter, gt_boxes, color=(0.0, 1.0, 0.0))  # green = GT
    add_boxes_3d(plotter, pred_boxes, color=(1.0, 0.0, 0.0))  # red = pred

    print(f"points={len(points)}, gt boxes={len(gt_boxes)}, pred boxes={len(pred_boxes)} (green=GT, red=Pred)")
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a single inference .npz file in 2D or 3D")
    parser.add_argument("npz_path", type=Path, help="Path to the .npz file to visualize")
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="3d",
        help="Visualization mode (default: 3d)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Score threshold to filter predicted boxes (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="(2D only) save the figure here instead of showing it interactively",
    )
    args = parser.parse_args()

    if not args.npz_path.exists():
        raise FileNotFoundError(args.npz_path)

    data = load_npz(args.npz_path)

    if args.mode == "2d":
        visualize_2d(data, score_thresh=args.score_thresh, output=args.output)
    else:
        visualize_3d(data, score_thresh=args.score_thresh)


if __name__ == "__main__":
    main()
