from typing import Dict, Tuple

import torch
from torch import Tensor

from components.definitions.mmperc_params import MmpercParams


class PointpillarLite:
    """
    Lightweight voxelizer for PointPillars.

    Converts raw lidar points (B, N, 4) into:
        - pillars:       (B, P, M, 4)
        - pillar_coords: (B, P, 2)  # (ix, iy)
        - pillar_count:  (B, P)
    """

    def __init__(
        self,
        params: MmpercParams,
        max_points_per_pillar: int = 20,
        max_pillars: int = 12000,
    ) -> None:

        x_range: Tuple[float, float] = params.bev_params.x_range
        y_range: Tuple[float, float] = params.bev_params.y_range
        z_range: Tuple[float, float] = params.bev_params.z_range
        voxel_size: Tuple[float, float, float] = params.bev_params.voxel_size

        # Spatial bounds
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

        # Voxel size
        self.vx, self.vy, self.vz = voxel_size

        # Pillar limits
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

    @torch.no_grad()
    def __call__(self, points: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            points: (B, N, 4) tensor  # x, y, z, intensity

        Returns:
            dict with:
                pillars:       (B, P, M, 4)
                pillar_coords: (B, P, 2)
                pillar_count:  (B, P)
        """
        if points.dim() != 3 or points.shape[-1] < 4:
            raise ValueError(f"Expected points of shape (B, N, >=4), got {tuple(points.shape)}")

        points = points[..., :4]
        B, N, C = points.shape
        device, dtype = points.device, points.dtype

        # Work on a fresh flat copy — never mutate the caller's tensor.
        pts = points.reshape(B * N, C)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        valid_mask = (
            (pts[:, 0] >= self.x_min)
            & (pts[:, 0] < self.x_max)
            & (pts[:, 1] >= self.y_min)
            & (pts[:, 1] < self.y_max)
            & (pts[:, 2] >= self.z_min)
            & (pts[:, 2] < self.z_max)
        )

        pillars = torch.zeros((B, self.max_pillars, self.max_points_per_pillar, C), dtype=dtype, device=device)
        pillar_count = torch.zeros(B, self.max_pillars, dtype=torch.long, device=device)
        pillar_coords = torch.zeros(B, self.max_pillars, 2, dtype=torch.long, device=device)

        if not valid_mask.any():
            return {"pillars": pillars, "pillar_coords": pillar_coords, "pillar_count": pillar_count}

        # Drop invalid points entirely instead of zeroing them in place.
        v_pts = pts[valid_mask]
        v_batch = batch_idx[valid_mask]
        v_ix = ((v_pts[:, 0] - self.x_min) / self.vx).long()
        v_iy = ((v_pts[:, 1] - self.y_min) / self.vy).long()

        coords = torch.stack([v_batch, v_ix, v_iy], dim=1)  # (V, 3)
        unique_coords, inverse, counts = torch.unique(
            coords, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        ub, uix, uiy = unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]
        num_groups = unique_coords.size(0)

        # Stable sort keeps each pillar's points in their original relative
        # order, matching the "keep first `count` points" behavior of the
        # original loop.
        order = torch.argsort(inverse, stable=True)
        sorted_pts = v_pts[order]
        sorted_group = inverse[order]

        group_start = torch.zeros(num_groups, dtype=torch.long, device=device)
        group_start[1:] = torch.cumsum(counts, 0)[:-1]
        point_slot = torch.arange(sorted_group.size(0), device=device) - group_start[sorted_group]

        # Local pillar id within each batch (0-based, in ascending (ix,iy) order).
        batch_pillar_counts = torch.bincount(ub, minlength=B)
        batch_pillar_offset = torch.zeros(B, dtype=torch.long, device=device)
        batch_pillar_offset[1:] = torch.cumsum(batch_pillar_counts, 0)[:-1]
        local_pillar_id = torch.arange(num_groups, device=device) - batch_pillar_offset[ub]

        keep_pillar = local_pillar_id < self.max_pillars
        if not bool(keep_pillar.all()):
            dropped = int((~keep_pillar).sum())
            print(f"[PointpillarLite] dropping {dropped} pillars beyond max_pillars={self.max_pillars}")

        keep_point = (point_slot < self.max_points_per_pillar) & keep_pillar[sorted_group]

        sel_b = ub[sorted_group[keep_point]]
        sel_pid = local_pillar_id[sorted_group[keep_point]]
        sel_slot = point_slot[keep_point]
        pillars[sel_b, sel_pid, sel_slot] = sorted_pts[keep_point]

        kb = ub[keep_pillar]
        klid = local_pillar_id[keep_pillar]
        pillar_count[kb, klid] = counts[keep_pillar].clamp(max=self.max_points_per_pillar)
        pillar_coords[kb, klid, 0] = uix[keep_pillar]
        pillar_coords[kb, klid, 1] = uiy[keep_pillar]

        return {
            "pillars": pillars,
            "pillar_coords": pillar_coords,
            "pillar_count": pillar_count,
        }
