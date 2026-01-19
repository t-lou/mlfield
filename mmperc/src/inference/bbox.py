from __future__ import annotations

from typing import Dict, List, Tuple

import common.params as params
import torch

# TODO use the params from common.params
_ = params
GRID_RES: float = 0.2  # meters per pixel
X_OFFSET: float = -50.0  # world origin offset in X
Y_OFFSET: float = -50.0  # world origin offset in Y


# ================================================================
# 1. Heatmap Top-K Extraction
# ================================================================


def topk_heatmap(heatmap: torch.Tensor, K: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract top-K heatmap peaks per class.

    Args:
        heatmap: Tensor of shape (B, C, H, W)
        K: number of top peaks per class

    Returns:
        scores: (B, C, K) top-K scores
        xs:     (B, C, K) x pixel indices
        ys:     (B, C, K) y pixel indices
    """
    B, C, H, W = heatmap.shape

    # Flatten spatial dims → (B, C, H*W)
    heatmap_flat = heatmap.view(B, C, -1)

    # Top-K per class
    scores, indices = torch.topk(heatmap_flat, K, dim=-1)

    # Convert flat indices back to (x, y)
    ys = indices // W
    xs = indices % W

    return scores, xs, ys


# ================================================================
# 2. Gather Regression Values at Peak Locations
# ================================================================


def gather_regression(reg_pred: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """
    Gather regression values at the top-K heatmap peak positions.

    Args:
        reg_pred: (B, 7, H, W)
        xs:       (B, C, K)
        ys:       (B, C, K)

    Returns:
        reg_vals: (B, C, K, 7)
    """
    B, C, K = xs.shape
    _, num_reg, H, W = reg_pred.shape
    assert num_reg == 7, "Regression head must output 7 channels (dx, dy, dz, w, l, h, yaw)"

    reg_out = torch.zeros((B, C, K, 7), device=reg_pred.device, dtype=reg_pred.dtype)

    # Vectorized gather (much faster than nested Python loops)
    for b in range(B):
        for c in range(C):
            x = xs[b, c]  # (K,)
            y = ys[b, c]  # (K,)
            reg_out[b, c] = reg_pred[b, :, y, x].T  # (K, 7)

    return reg_out


# ================================================================
# 3. Decode Boxes into World Coordinates
# ================================================================


def decode_boxes(xs: torch.Tensor, ys: torch.Tensor, reg_vals: torch.Tensor) -> List[List[List[float]]]:
    """
    Decode predicted boxes into world coordinates.

    Args:
        xs:       (B, C, K)
        ys:       (B, C, K)
        reg_vals: (B, C, K, 7)

    Returns:
        boxes: list of length B, each containing a list of boxes:
               [x, y, z, w, l, h, yaw]
    """
    B, C, K = xs.shape
    boxes: List[List[List[float]]] = []

    for b in range(B):
        boxes_b: List[List[float]] = []

        for c in range(C):
            for k in range(K):
                x_pix = xs[b, c, k].item()
                y_pix = ys[b, c, k].item()

                dx, dy, dz, w, l_, h, yaw = reg_vals[b, c, k].tolist()

                # Convert pixel → world coordinates
                x = x_pix * GRID_RES + X_OFFSET + dx
                y = y_pix * GRID_RES + Y_OFFSET + dy
                z = dz

                boxes_b.append([x, y, z, w, l_, h, yaw])

        boxes.append(boxes_b)

    return boxes


# ================================================================
# 4. High-Level Inference Wrapper
# ================================================================


def model_inference(
    model: torch.nn.Module, points: torch.Tensor, images: torch.Tensor, K: int = 50
) -> Tuple[List[List[List[float]]], torch.Tensor]:
    """
    Run inference on a batch and decode bounding boxes.

    Args:
        model:  detection model
        points: (B, N, C) lidar input
        images: (B, 3, H, W) camera input
        K:      number of top-K peaks per class

    Returns:
        boxes:  list of decoded boxes per batch
        scores: (B, C, K) heatmap scores
    """
    model.eval()

    with torch.no_grad():
        pred: Dict[str, torch.Tensor] = model(points, images)

        heatmap = pred["heatmap"]  # (B, C, H, W)
        reg = pred["reg"]  # (B, 7, H, W)

        scores, xs, ys = topk_heatmap(heatmap, K)
        reg_vals = gather_regression(reg, xs, ys)
        boxes = decode_boxes(xs, ys, reg_vals)

    return boxes, scores
