import torch
from torch import Tensor


def scatter_to_bev(
    pillar_features: Tensor,
    pillar_coords_xy: Tensor,  # world coords (x, y) for each pillar
    bev_h: int,
    bev_w: int,
) -> Tensor:
    """
    Scatter pillar features into a dense BEV grid.

    Args:
        pillar_features: (B, P, C)
            Per‑pillar feature vectors.
        pillar_coords_xy: (B, P, 2)
            World coordinates (x, y) for each pillar.
        bev_h: Height of the BEV grid.
        bev_w: Width of the BEV grid.

    Returns:
        bev: (B, C, bev_h, bev_w)
            Dense BEV feature map.
    """
    B, P, C = pillar_features.shape
    device = pillar_features.device

    bev = torch.zeros(B, C, bev_h, bev_w, device=device)

    for b in range(B):
        feats = pillar_features[b]  # (P, C)
        coords_xy = pillar_coords_xy[b]  # (P, 2)

        # Convert all (x, y) → (ix, iy) using your unified helper
        ix = coords_xy[:, 0].long()
        iy = coords_xy[:, 1].long()

        # Keep only valid pillars
        valid = (ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h)
        if not valid.any():
            continue

        ix = ix[valid]
        iy = iy[valid]
        feats = feats[valid]  # (P_valid, C)

        # Scatter into BEV grid
        bev[b, :, iy, ix] = feats.t()

    return bev
