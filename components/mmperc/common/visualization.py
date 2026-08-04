"""
viz_common.py
=============

Reusable building blocks shared by the visualizer scripts in this package:

  - visualize_inference.py  -> visualizes ModelInferenceWrapper .npz outputs
  - visualize_dataset.py    -> visualizes a single raw A2D2Dataset sample (GT only)
  - visualize_browser.py    -> interactive frame-by-frame browser over A2D2Dataset

Convention used everywhere: 2D mode is matplotlib-only (camera image + BEV
plot, if a camera image exists). 3D mode always uses PyVista for the point
cloud / box view; when a camera image also needs to be shown alongside it,
that goes in a *separate* matplotlib window synced by frame index, since
mixing an image panel directly into a PyVista renderer is not worth the
fragility.

Nothing in here knows about .npz files or the A2D2Dataset class specifically -
it only operates on plain numpy arrays, so every visualizer (and any future
one) can import it without pulling in unrelated dependencies.

Box column convention (shared across the whole codebase):
    [x, y, z, l, w, h, yaw]
    x, y, z : box center in world coordinates (meters)
    l       : extent along the box's local X axis (heading direction)
    w       : extent along the box's local Y axis (left/right)
    h       : extent along Z (height)
    yaw     : rotation around Z, radians
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Generic array helpers
# ----------------------------------------------------------------------


def to_numpy(x: Any) -> np.ndarray:
    """Convert a torch.Tensor (or anything array-like) to a plain numpy array."""
    if x is None:
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def as_boxes(value: Any, ncols: int = 7) -> np.ndarray:
    """Normalize a boxes container (possibly a torch tensor, a 1D empty array,
    or a single flat box) into a well-formed (N, ncols) numpy array."""
    boxes = to_numpy(value) if value is not None else np.zeros((0, ncols))
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(0, ncols) if boxes.size == 0 else boxes.reshape(1, -1)
    return boxes


def strip_zero_padded_rows(arr: np.ndarray) -> np.ndarray:
    """Drop the trailing block of all-zero rows from a fixed-size, zero-padded
    array (points or boxes padded up to a fixed capacity, as A2D2Dataset does).

    Caveat: this assumes padding is a contiguous trailing block of exact
    all-zero rows, which holds for A2D2Dataset's padding scheme but would be
    wrong if a genuine row happens to be all zeros.
    """
    if arr.size == 0:
        return arr
    nonzero_rows = np.flatnonzero(np.any(arr != 0, axis=-1))
    if nonzero_rows.size == 0:
        return arr[:0]
    last = nonzero_rows[-1] + 1
    return arr[:last]


def filter_pred_boxes(pred_boxes: np.ndarray, pred_scores: Optional[np.ndarray], score_thresh: float) -> np.ndarray:
    """Drop predicted boxes below score_thresh. Falls back to no filtering if
    the scores don't line up 1:1 with the boxes (rather than guessing wrong)."""
    if pred_scores is None or pred_boxes.shape[0] == 0:
        return pred_boxes
    scores_flat = to_numpy(pred_scores).reshape(-1)
    if scores_flat.shape[0] != pred_boxes.shape[0]:
        return pred_boxes
    return pred_boxes[scores_flat >= score_thresh]


def camera_chw_to_uint8_hwc(camera_chw: np.ndarray) -> np.ndarray:
    """Convert an A2D2Dataset-style (3, H, W) float camera tensor in [0, 1]
    into a plain (H, W, 3) uint8 image, ready for matplotlib/imshow."""
    camera_hwc = np.asarray(camera_chw).transpose(1, 2, 0)
    return np.clip(camera_hwc * 255.0, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Box geometry
# ----------------------------------------------------------------------


def get_box_corners_3d(box: np.ndarray) -> np.ndarray:
    """Return the 8 corners of a single [x, y, z, l, w, h, yaw] box, shape (8, 3)."""
    x, y, z, l_, h, w, yaw = box[:7]

    corners_local = np.array(
        [
            [l_ / 2, w / 2, -h / 2],
            [l_ / 2, -w / 2, -h / 2],
            [-l_ / 2, -w / 2, -h / 2],
            [-l_ / 2, w / 2, -h / 2],
            [l_ / 2, w / 2, h / 2],
            [l_ / 2, -w / 2, h / 2],
            [-l_ / 2, -w / 2, h / 2],
            [-l_ / 2, w / 2, h / 2],
        ]
    )

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])

    corners = corners_local @ rot.T
    corners += np.array([x, y, z])
    return corners


# ----------------------------------------------------------------------
# 2D (matplotlib, top-down XY view + camera image)
# ----------------------------------------------------------------------


def draw_box_2d(ax, box: np.ndarray, color: str, label: Optional[str] = None) -> None:
    """Draw a single box's bottom face + heading tick onto a matplotlib Axes."""
    from matplotlib.patches import Polygon

    corners = get_box_corners_3d(box)[:4, :2]  # bottom face, xy only
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.5, label=label))
    # heading tick from center to the midpoint of the "front" edge (corners 0-1)
    center = box[:2]
    front_mid = corners[[0, 1]].mean(axis=0)
    ax.plot([center[0], front_mid[0]], [center[1], front_mid[1]], color=color, linewidth=1.5)


def draw_boxes_2d(ax, boxes: np.ndarray, color: str, label: Optional[str] = None) -> None:
    """Draw a whole set of boxes, only labeling the first one so the legend
    doesn't get one entry per box."""
    for i, box in enumerate(boxes):
        draw_box_2d(ax, box, color, label if i == 0 else None)


def plot_bev(
    ax,
    points: np.ndarray,
    box_sets: Iterable[Tuple[np.ndarray, str, Optional[str]]] = (),
    title: Optional[str] = None,
) -> None:
    """Render a top-down (BEV) scatter of `points` plus any number of box sets.
    Clears `ax` first so this can be reused frame-to-frame in an interactive
    browser without accumulating stale artists.

    box_sets: iterable of (boxes, color, label) tuples, e.g.
        [(gt_boxes, "limegreen", "GT"), (pred_boxes, "red", "Pred")]
    """
    ax.clear()
    ax.scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap="viridis", alpha=0.6)

    any_labeled = False
    for boxes, color, label in box_sets:
        if len(boxes):
            draw_boxes_2d(ax, boxes, color, label)
            any_labeled = any_labeled or label is not None

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    if any_labeled:
        ax.legend(loc="upper right")


def show_camera_image(ax, image: np.ndarray, title: str = "Camera (front center)") -> None:
    """Render a camera image (H, W, 3) onto a matplotlib Axes. Clears `ax`
    first so this can be reused frame-to-frame in an interactive browser."""
    ax.clear()
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


# ----------------------------------------------------------------------
# Semantics
# ----------------------------------------------------------------------


def semantic_ids_to_rgb(sem_ids: np.ndarray, class_to_color: Iterable[Tuple[int, Tuple[int, int, int]]]) -> np.ndarray:
    """Map a (H, W) array of class ids to an (H, W, 3) uint8 color image using
    a (class_id, (r, g, b)) mapping."""
    sem_rgb = np.zeros((*sem_ids.shape, 3), dtype=np.uint8)
    for cid, rgb in class_to_color:
        sem_rgb[sem_ids == cid] = rgb
    return sem_rgb


# ----------------------------------------------------------------------
# 3D (PyVista)
# ----------------------------------------------------------------------


def box_lines_polydata(box: np.ndarray):
    """Build a PyVista PolyData containing the 12 wireframe edges of a box."""
    import pyvista as pv

    corners = get_box_corners_3d(box)
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # verticals
    ]
    # VTK "lines" connectivity format: [n_points_in_cell, p0, p1, n_points_in_cell, p0, p1, ...]
    lines = np.hstack([[2, a, b] for a, b in edges])
    return pv.PolyData(corners, lines=lines)


def add_point_cloud(plotter, points: np.ndarray, point_size: float = 2.0):
    """Add a z-colored point cloud to a PyVista plotter/subplot. Returns the actor."""
    import pyvista as pv

    point_cloud = pv.PolyData(points)
    point_cloud["z"] = points[:, 2]
    return plotter.add_mesh(
        point_cloud, scalars="z", cmap="viridis", point_size=point_size, render_points_as_spheres=False
    )


def add_boxes_3d(plotter, boxes: np.ndarray, color: Tuple[float, float, float], line_width: float = 2) -> List:
    """Add a set of wireframe boxes to a PyVista plotter/subplot. Returns the list of actors."""
    actors = []
    for box in boxes:
        actors.append(plotter.add_mesh(box_lines_polydata(box), color=color, line_width=line_width))
    return actors


def build_3d_scene(
    plotter, points: np.ndarray, box_sets: Iterable[Tuple[np.ndarray, Tuple[float, float, float]]] = ()
) -> List:
    """Populate a PyVista plotter/subplot with a point cloud, an origin axes
    triad, and any number of box sets. Returns the full list of actors added
    (handy for later removal when redrawing in an interactive browser).

    box_sets: iterable of (boxes, rgb_color) tuples.
    """
    actors = [add_point_cloud(plotter, points)]
    plotter.add_axes_at_origin(labels_off=True, line_width=3)
    for boxes, color in box_sets:
        if len(boxes):
            actors.extend(add_boxes_3d(plotter, boxes, color))
    return actors
