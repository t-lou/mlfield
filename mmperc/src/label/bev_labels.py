import math
from typing import List

import common.params as params
import torch
from common.bev_utils import get_res, xy_to_grid_stride


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    """
    Draw a 2D Gaussian on the heatmap centered at (cx, cy).

    heatmap: (H, W)
    cx, cy: integer center coordinates (x = width, y = height)
    radius: Gaussian radius in pixels
    """
    H, W = heatmap.shape
    diameter = 2 * radius + 1

    gaussian = torch.zeros((diameter, diameter), dtype=heatmap.dtype, device=heatmap.device)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist = dx * dx + dy * dy
            gaussian[dy + radius, dx + radius] = math.exp(-dist / (2 * (radius**2)))

    x0 = cx - radius
    y0 = cy - radius
    x1 = cx + radius + 1
    y1 = cy + radius + 1

    gx0 = max(0, -x0)
    gy0 = max(0, -y0)
    gx1 = diameter - max(0, x1 - W)
    gy1 = diameter - max(0, y1 - H)

    hx0 = max(0, x0)
    hy0 = max(0, y0)
    hx1 = min(W, x1)
    hy1 = min(H, y1)

    heatmap[hy0:hy1, hx0:hx1] = torch.max(
        heatmap[hy0:hy1, hx0:hx1],
        gaussian[gy0:gy1, gx0:gx1],
    )


def generate_bev_labels_bbox2d(
    gt_boxes: List[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    gt_boxes: list of tensors, each (N, 7) with [x, y, z, w, l_, h, yaw]

    Returns:
        heatmap: (B, 1, H_out, W_out)
        reg:     (B, 6, H_out, W_out)
        mask:    (B, 1, H_out, W_out)
    """
    stride = params.BACKBONE_STRIDE
    bev_h = params.BEV_H // stride
    bev_w = params.BEV_W // stride

    B = len(gt_boxes)
    heatmap = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.float32)
    reg = torch.zeros((B, 6, bev_h, bev_w), dtype=torch.float32)
    mask = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.float32)

    for b in range(B):
        boxes = gt_boxes[b]  # (N, 7)

        for box in boxes:
            if box.abs().sum() == 0:
                continue

            x, y, z, w, l_, h, yaw = box.tolist()

            # -------------------------------
            # 1. World → BEV grid (stride-aware)
            # -------------------------------
            ix, iy = xy_to_grid_stride(x, y)

            if ix < 0 or ix >= bev_w or iy < 0 or iy >= bev_h:
                continue

            # -------------------------------
            # 2. Heatmap (Gaussian peak)
            # -------------------------------
            # cx = ix (x index, width), cy = iy (y index, height)
            draw_gaussian(heatmap[b, 0], ix, iy, radius=2)

            # -------------------------------
            # 3. Regression targets
            # -------------------------------
            # Use the same convention as before: offsets normalized by voxel size.
            # Note: ix, iy are on the downsampled grid, so multiply by stride.
            res_x, res_y = get_res()

            cell_x = params.X_RANGE[0] + ix * (res_x * stride)
            cell_y = params.Y_RANGE[0] + iy * (res_y * stride)

            dx = (x - cell_x) / (res_x * stride)
            dy = (y - cell_y) / (res_y * stride)

            reg[b, :, iy, ix] = torch.tensor(
                [
                    dx,
                    dy,
                    math.log(w),
                    math.log(l_),
                    math.sin(yaw),
                    math.cos(yaw),
                ],
                dtype=torch.float32,
            )

            # -------------------------------
            # 4. Mask
            # -------------------------------
            mask[b, 0, iy, ix] = 1.0

    return heatmap, reg, mask
