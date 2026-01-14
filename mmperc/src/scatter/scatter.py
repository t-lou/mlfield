import torch


def scatter_to_bev(pillar_features, pillar_coords, bev_h, bev_w):
    """
    pillar_features: (B, P, C)
    pillar_coords:   (B, P, 2) with (ix, iy)
    returns: bev (B, C, H, W)
    """
    B, P, C = pillar_features.shape
    device = pillar_features.device

    bev = torch.zeros(B, C, bev_h, bev_w, device=device)

    for b in range(B):
        coords = pillar_coords[b]  # (P, 2)
        feats = pillar_features[b]  # (P, C)

        ix = coords[:, 0].long()
        iy = coords[:, 1].long()

        valid = (ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h)
        ix = ix[valid]
        iy = iy[valid]
        feats = feats[valid]

        bev[b, :, iy, ix] = feats.t()  # (C, H, W)

    return bev
