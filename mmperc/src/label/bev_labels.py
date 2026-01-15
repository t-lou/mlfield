import math

import common.params as params
import torch


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int):
    """
    Draw a 2D Gaussian on the heatmap centered at (cx, cy).

    heatmap: (H, W)
    cx, cy: integer center coordinates
    radius: Gaussian radius in pixels
    """
    H, W = heatmap.shape
    diameter = 2 * radius + 1

    # Precompute Gaussian kernel
    gaussian = torch.zeros((diameter, diameter), dtype=heatmap.dtype, device=heatmap.device)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist = dx * dx + dy * dy
            gaussian[dy + radius, dx + radius] = math.exp(-dist / (2 * (radius**2)))

    # Paste Gaussian into heatmap
    x0 = cx - radius
    y0 = cy - radius
    x1 = cx + radius + 1
    y1 = cy + radius + 1

    # Compute valid region inside heatmap
    gx0 = max(0, -x0)
    gy0 = max(0, -y0)
    gx1 = diameter - max(0, x1 - W)
    gy1 = diameter - max(0, y1 - H)

    hx0 = max(0, x0)
    hy0 = max(0, y0)
    hx1 = min(W, x1)
    hy1 = min(H, y1)

    # Apply max() to avoid overwriting stronger peaks
    heatmap[hy0:hy1, hx0:hx1] = torch.max(heatmap[hy0:hy1, hx0:hx1], gaussian[gy0:gy1, gx0:gx1])


def generate_bev_labels_bbox2d(
    gt_boxes, bev_h=params.BEV_H, bev_w=params.BEV_W, voxel_size=params.VOXEL_SIZE, pc_range=params.PC_RANGE
):
    """
    gt_boxes: list of tensors, each (N, 7) with [x, y, z, w, l_, h, yaw]
    Returns:
        heatmap: (B, 1, H, W)
        reg:     (B, 6, H, W)
        mask:    (B, 1, H, W)
    """

    B = len(gt_boxes)
    heatmap = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.float32)
    reg = torch.zeros((B, 6, bev_h, bev_w), dtype=torch.float32)
    mask = torch.zeros((B, 1, bev_h, bev_w), dtype=torch.float32)

    vx, vy = voxel_size
    x_min, y_min = pc_range[0], pc_range[1]

    for b in range(B):
        boxes = gt_boxes[b]  # (N, 7)

        for box in boxes:
            x, y, z, w, l_, h, yaw = box.tolist()

            # Convert world coords → BEV grid
            ix = int((x - x_min) / vx)
            iy = int((y - y_min) / vy)

            if ix < 0 or ix >= bev_w or iy < 0 or iy >= bev_h:
                continue

            # -------------------------------
            # 1. Heatmap (Gaussian peak)
            # -------------------------------
            draw_gaussian(heatmap[b, 0], ix, iy, radius=2)

            # -------------------------------
            # 2. Regression targets
            # -------------------------------
            dx = (x - (ix * vx + x_min)) / vx
            dy = (y - (iy * vy + y_min)) / vy

            reg[b, :, iy, ix] = torch.tensor(
                [
                    dx,
                    dy,
                    math.log(w),
                    math.log(l_),
                    math.sin(yaw),
                    math.cos(yaw),
                ]
            )

            # -------------------------------
            # 3. Mask
            # -------------------------------
            mask[b, 0, iy, ix] = 1.0

    return heatmap, reg, mask
