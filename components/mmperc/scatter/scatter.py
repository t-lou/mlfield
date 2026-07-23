import torch
from torch import Tensor


def scatter_to_bev(
    pillar_features: Tensor,
    pillar_coords_xy: Tensor,  # grid coords (ix, iy) for each pillar
    bev_h: int,
    bev_w: int,
) -> Tensor:
    """
    Scatter pillar features into a dense BEV grid.
    Vectorized implementation (no Python loop over batch) for better performance.

    Args:
        pillar_features: (B, P, C)
            Per‑pillar feature vectors.
        pillar_coords_xy: (B, P, 2)
            Grid coordinates (ix, iy) for each pillar.
        bev_h: Height of the BEV grid.
        bev_w: Width of the BEV grid.

    Returns:
        bev: (B, C, bev_h, bev_w)
            Dense BEV feature map.
    """
    B, P, C = pillar_features.shape
    device = pillar_features.device

    # Vectorized scatter: avoid Python loop over batch
    bev = torch.zeros(B, C, bev_h, bev_w, device=device, dtype=pillar_features.dtype)

    # Get coordinates
    ix = pillar_coords_xy[..., 0].long()  # (B, P)
    iy = pillar_coords_xy[..., 1].long()  # (B, P)

    # Create batch indices
    batch_idx = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(B, P)

    # Validate coordinates
    valid = (ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h)

    # Flatten features and indices
    batch_idx_flat = batch_idx.reshape(B * P)
    ix_flat = ix.reshape(B * P)
    iy_flat = iy.reshape(B * P)
    valid_flat = valid.reshape(B * P)
    feats_flat = pillar_features.reshape(B * P, C)

    # Filter to valid entries
    batch_idx_valid = batch_idx_flat[valid_flat]
    ix_valid = ix_flat[valid_flat]
    iy_valid = iy_flat[valid_flat]
    feats_valid = feats_flat[valid_flat]  # (P_valid, C)

    # Scatter into BEV grid
    bev[batch_idx_valid, :, iy_valid, ix_valid] = feats_valid

    return bev
