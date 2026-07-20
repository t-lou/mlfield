from __future__ import annotations

from typing import Tuple

from components.definitions.mmperc import MmpercParams


# ================================================================
# Resolution helpers
# ================================================================
def get_res(mmperc_params: MmpercParams) -> Tuple[float, float]:
    """
    Returns (res_x, res_y) in meters per pixel.
    res_x: meters per pixel along X (width)
    res_y: meters per pixel along Y (height)
    """
    res_x = (mmperc_params.x_range[1] - mmperc_params.x_range[0]) / mmperc_params.bev_params.bev_w
    res_y = (mmperc_params.y_range[1] - mmperc_params.y_range[0]) / mmperc_params.bev_params.bev_h
    return res_x, res_y


# ================================================================
# World → Grid
# ================================================================
def xy_to_grid(x: float, y: float, mmperc_params: MmpercParams) -> Tuple[int, int]:
    """
    Convert world coordinates (x, y) in meters into BEV grid indices (ix, iy).

    ix indexes width  (X direction)
    iy indexes height (Y direction)

    Use the returned indices as map[b, c, iy, ix] because BEV tensors follow (H, W) layout.

    Returns:
        (ix, iy) as integer grid indices.
    """
    res_x, res_y = get_res()

    # Convert meters → pixel index
    gx = (x - mmperc_params.x_range[0]) / res_x
    gy = (y - mmperc_params.y_range[0]) / res_y

    ix = int(gx)
    iy = int(gy)

    return ix, iy


# ================================================================
# World → Grid (stride-aware)
# ================================================================


def xy_to_grid_stride(x: float, y: float, mmperc_params: MmpercParams) -> Tuple[int, int]:
    """
    Convert world coordinates (x, y) into BEV grid indices (ix, iy)
    for a downsampled grid with the given stride.

    Example:
        stride = 4 → heatmap is BEV_H/4 x BEV_W/4.

    Note: the stride in params.py is used.

    ix indexes width  (X direction)
    iy indexes height (Y direction)
    """
    stride = float(mmperc_params.backbone_stride)
    res_x, res_y = get_res(mmperc_params)

    gx = (x - mmperc_params.x_range[0]) / (res_x * stride)
    gy = (y - mmperc_params.y_range[0]) / (res_y * stride)

    ix = int(gx)
    iy = int(gy)
    return ix, iy


# ================================================================
# Grid → World
# ================================================================
def grid_to_xy(ix: int, iy: int, mmperc_params: MmpercParams) -> Tuple[float, float]:
    """
    Convert BEV grid indices (ix, iy) back into world coordinates (x, y).

    ix indexes width  (X direction)
    iy indexes height (Y direction)

    Returns:
        (x, y) in meters.
    """
    res_x, res_y = get_res(mmperc_params)

    x = mmperc_params.x_range[0] + ix * res_x
    y = mmperc_params.y_range[0] + iy * res_y

    return x, y


# ================================================================
# World → Grid (stride-aware)
# ================================================================


def grid_to_xy_stride(ix: int, iy: int, mmperc_params: MmpercParams) -> tuple[float, float]:
    """
    Convert BEV grid indices (ix, iy) at a downsampled stride
    back into world coordinates (x, y).
    """
    stride = float(mmperc_params.backbone_stride)
    res_x, res_y = get_res(mmperc_params)

    x = mmperc_params.x_range[0] + ix * (res_x * stride)
    y = mmperc_params.y_range[0] + iy * (res_y * stride)

    return x, y
