import torch
from torch import Tensor


def scatter_to_bev(
    pillar_features: Tensor,
    pillar_coords: Tensor,
    bev_h: int,
    bev_w: int,
) -> Tensor:
    """
    Scatter pillar features into a dense BEV grid.

    Args:
        pillar_features: (B, P, C)
            Per‑pillar feature vectors.
        pillar_coords:   (B, P, 2)
            Integer pillar coordinates (ix, iy) for each pillar.
        bev_h: Height of the BEV grid.
        bev_w: Width of the BEV grid.

    Returns:
        bev: (B, C, bev_h, bev_w)
            Dense BEV feature map.
    """
    B, P, C = pillar_features.shape
    device = pillar_features.device

    # Initialize empty BEV grid
    bev = torch.zeros(B, C, bev_h, bev_w, device=device)

    for b in range(B):
        coords = pillar_coords[b]  # (P, 2)
        feats = pillar_features[b]  # (P, C)

        ix = coords[:, 0].long()
        iy = coords[:, 1].long()

        # Keep only coordinates inside the BEV grid
        valid = (ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h)
        if not valid.any():
            continue

        ix = ix[valid]
        iy = iy[valid]
        feats = feats[valid]  # (P_valid, C)

        # Scatter features into BEV grid
        bev[b, :, iy, ix] = feats.t()  # (C, H, W)

    return bev
