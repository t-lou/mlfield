import math
from typing import List

import torch

import common.params as params
from common.bev_utils import get_res, grid_to_xy, xy_to_grid

GAUSSIAN_CACHE = {}


def get_gaussian(radius: int, device, dtype):
    key = (radius, device, dtype)
    if key not in GAUSSIAN_CACHE:
        xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        ys = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dist = xx * xx + yy * yy
        GAUSSIAN_CACHE[key] = torch.exp(-dist / (2 * (radius**2))).to(device=device, dtype=dtype)
    return GAUSSIAN_CACHE[key]


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    """
    Draw a 2D Gaussian on the heatmap centered at (cx, cy).

    heatmap: (H, W)
    cx, cy: integer center coordinates (x = width, y = height)
    radius: Gaussian radius in pixels
    """
    H, W = heatmap.shape
    gaussian = get_gaussian(radius, heatmap.device, heatmap.dtype)
    diameter = gaussian.shape[0]

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

    heatmap[hy0:hy1, hx0:hx1] = torch.max(heatmap[hy0:hy1, hx0:hx1], gaussian[gy0:gy1, gx0:gx1])


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
    bev_h = params.BEV_H
    bev_w = params.BEV_W

    B = len(gt_boxes)
    heatmap = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.float16)
    reg = torch.zeros((B, 6, bev_h, bev_w), dtype=torch.float32)
    mask = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.uint8)

    for b in range(B):
        boxes = gt_boxes[b]  # (N, 7)

        for box in boxes:
            if box.abs().sum() == 0:
                continue

            x, y, z, w, l_, h, yaw = box.tolist()

            # -------------------------------
            # 1. World → BEV grid
            # -------------------------------
            ix, iy = xy_to_grid(x, y)

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
            res_x, res_y = get_res()

            cell_x, cell_y = grid_to_xy(ix, iy)

            dx = (x - cell_x) / res_x
            dy = (y - cell_y) / res_y

            reg[b, 0, iy, ix] = dx
            reg[b, 1, iy, ix] = dy
            reg[b, 2, iy, ix] = math.log(w)
            reg[b, 3, iy, ix] = math.log(l_)
            reg[b, 4, iy, ix] = math.sin(yaw)
            reg[b, 5, iy, ix] = math.cos(yaw)

            # -------------------------------
            # 4. Mask
            # -------------------------------
            mask[b, 0, iy, ix] = 1

    return heatmap, reg, mask
