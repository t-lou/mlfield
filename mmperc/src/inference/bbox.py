from __future__ import annotations

from typing import List, Tuple

import common.params as params
import torch
from common.bev_utils import get_res, grid_to_xy_stride

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
    stride = params.BACKBONE_STRIDE
    res_x, res_y = get_res()

    boxes: List[List[List[float]]] = []

    for b in range(B):
        boxes_b: List[List[float]] = []

        for c in range(C):
            for k in range(K):
                ix = xs[b, c, k].item()
                iy = ys[b, c, k].item()

                dx, dy, dz, w, l_, h, yaw = reg_vals[b, c, k].tolist()

                # 1. Convert grid index → world coordinate of cell origin
                cell_x, cell_y = grid_to_xy_stride(ix, iy)

                # 2. Convert normalized offsets → meters
                x = cell_x + dx * (res_x * stride)
                y = cell_y + dy * (res_y * stride)
                z = dz  # unchanged

                boxes_b.append([x, y, z, w, l_, h, yaw])

        boxes.append(boxes_b)

    return boxes


# ================================================================
# 4. High-Level Inference Wrapper
# ================================================================


def model_inference(
    model: torch.nn.Module,
    points: torch.Tensor,
    images: torch.Tensor,
    K: int = 50,
):
    """
    Run inference on a batch and decode bounding boxes.

    Args:
        model:  detection model
        points: (B, N, 4) lidar input
        images: (B, 3, H, W) camera input
        K:      number of top-K peaks per class

    Returns:
        boxes:  list of decoded boxes per batch
        scores: (B, C, K) heatmap scores
    """
    model.eval()

    with torch.no_grad():
        # ---------------------------------------------------------
        # 1. Forward pass through your full multimodal model
        # ---------------------------------------------------------
        pred = model(points, images)

        heatmap = pred["heatmap"]  # (B, C, H_bev, W_bev)
        reg = pred["reg"]  # (B, 6, H_bev, W_bev)

        # ---------------------------------------------------------
        # 2. Top-K heatmap peaks
        # ---------------------------------------------------------
        scores, xs, ys = topk_heatmap(heatmap, K)

        # ---------------------------------------------------------
        # 3. Gather regression values at peak locations
        # ---------------------------------------------------------
        reg_vals = gather_regression(reg, xs, ys)

        # ---------------------------------------------------------
        # 4. Decode boxes into world coordinates
        # ---------------------------------------------------------
        boxes = decode_boxes(xs, ys, reg_vals)

    return boxes, scores
