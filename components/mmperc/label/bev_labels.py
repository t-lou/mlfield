import math
from typing import List

import torch

from components.definitions.mmperc import MmpercParams
from components.utils.bev_utils import get_res, grid_to_xy, xy_to_grid

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
    mmperc_params: MmpercParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    gt_boxes: list of tensors, each (N, 7) with [x, y, z, w, l_, h, yaw]

    Returns:
        heatmap: (B, 1, H_out, W_out)
        reg:     (B, 8, H_out, W_out)  [dx, dy, dz, log_w, log_l, log_h, sin_yaw, cos_yaw]
        mask:    (B, 1, H_out, W_out)
    """
    bev_h = mmperc_params.bev_params.bev_h
    bev_w = mmperc_params.bev_params.bev_w

    B = len(gt_boxes)
    heatmap = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.float16)
    reg = torch.zeros((B, 8, bev_h, bev_w), dtype=torch.float32)
    mask = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.uint8)

    # z reference: center of the z range for stable dz targets
    z_min, z_max = mmperc_params.bev_params.z_range
    z_ref = (z_min + z_max) / 2.0

    for b in range(B):
        boxes = gt_boxes[b]  # (N, 7)

        for box in boxes:
            if box.abs().sum() == 0:
                continue

            x, y, z, w, l_, h, yaw = box.tolist()

            # -------------------------------
            # 1. World → BEV grid
            # -------------------------------
            ix, iy = xy_to_grid(x, y, mmperc_params=mmperc_params)

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
            res_x, res_y = get_res(mmperc_params=mmperc_params)

            cell_x, cell_y = grid_to_xy(ix, iy, mmperc_params=mmperc_params)

            dx = (x - cell_x) / res_x
            dy = (y - cell_y) / res_y
            dz = z - z_ref

            reg[b, 0, iy, ix] = dx
            reg[b, 1, iy, ix] = dy
            reg[b, 2, iy, ix] = dz
            reg[b, 3, iy, ix] = math.log(w)
            reg[b, 4, iy, ix] = math.log(l_)
            reg[b, 5, iy, ix] = math.log(h)
            reg[b, 6, iy, ix] = math.sin(yaw)
            reg[b, 7, iy, ix] = math.cos(yaw)

            # -------------------------------
            # 4. Mask
            # -------------------------------
            mask[b, 0, iy, ix] = 1

    return heatmap, reg, mask
