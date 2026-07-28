from typing import Dict, Tuple

import torch
from torch import Tensor

from components.definitions.mmperc_params import MmpercParams
from components.utils.logger import logger


class PointpillarLite:
    """
    Lightweight voxelizer for PointPillars.

    Converts raw lidar points (B, N, 4) into:
        - pillars:       (B, P, M, 4)
        - pillar_coords: (B, P, 2)  # (ix, iy)
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

        # Number of grid bins along each axis, used to collapse the
        # (batch, ix, iy) triple into a single integer key below. +1 guards
        # against floating point edge cases pushing ix/iy to the last bin.
        self.ix_bins = int((self.x_max - self.x_min) / self.vx) + 1
        self.iy_bins = int((self.y_max - self.y_min) / self.vy) + 1

    @torch.no_grad()
    def __call__(self, points: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            points: (B, N, 4) tensor  # x, y, z, intensity

        Returns:
            dict with:
                pillars:       (B, P, M, 9)
                pillar_coords: (B, P, 2)
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

        NUM_FEATURES = 9  # xyzi, pillar mean 3d and pillar 2d center
        pillars = torch.zeros(
            (B, self.max_pillars, self.max_points_per_pillar, NUM_FEATURES), dtype=dtype, device=device
        )
        pillar_coords = torch.zeros(B, self.max_pillars, 2, dtype=torch.long, device=device)

        if not valid_mask.any():
            return {"pillars": pillars, "pillar_coords": pillar_coords}

        # Drop invalid points entirely instead of zeroing them in place.
        v_pts = pts[valid_mask]
        v_batch = batch_idx[valid_mask]
        v_ix = ((v_pts[:, 0] - self.x_min) / self.vx).long()
        v_iy = ((v_pts[:, 1] - self.y_min) / self.vy).long()

        # Collapse (batch, ix, iy) into a single integer key. torch.unique on
        # a 1D tensor is a plain sort; torch.unique(..., dim=0) on a (V, 3)
        # tensor does a lexicographic multi-column sort, which is meaningfully
        # slower (especially on GPU) for the same result. Batch stays the
        # most-significant component of the key, so downstream ordering
        # assumptions (e.g. local_pillar_id below) are unaffected.
        cell_stride = self.ix_bins * self.iy_bins
        keys = v_batch * cell_stride + v_ix * self.iy_bins + v_iy  # (V,)
        unique_keys, inverse, counts = torch.unique(keys, sorted=True, return_inverse=True, return_counts=True)
        ub = unique_keys // cell_stride
        rem = unique_keys % cell_stride
        uix = rem // self.iy_bins
        uiy = rem % self.iy_bins
        num_groups = unique_keys.size(0)

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
            logger.info(f"[PointpillarLite] dropping {dropped} pillars beyond max_pillars={self.max_pillars}")

        keep_point = (point_slot < self.max_points_per_pillar) & keep_pillar[sorted_group]

        # --- Feature augmentation: cluster-center + pillar-center offsets ---
        # Compute cluster mean (x, y, z) per pillar, using only the points that
        # will actually be kept (post-truncation), so it matches what the
        # network sees rather than points that get dropped for overflow.
        kept_pts = sorted_pts[keep_point]  # (K, 4)
        kept_group = sorted_group[keep_point]  # (K,)

        group_sum = torch.zeros(num_groups, 3, dtype=dtype, device=device)
        group_sum.index_add_(0, kept_group, kept_pts[:, :3])
        group_kept_count = torch.zeros(num_groups, dtype=dtype, device=device)
        group_kept_count.index_add_(0, kept_group, torch.ones_like(kept_group, dtype=dtype))
        group_mean = group_sum / group_kept_count.clamp(min=1).unsqueeze(-1)  # (num_groups, 3)

        cluster_offset = kept_pts[:, :3] - group_mean[kept_group]  # (K, 3)

        # Pillar geometric center, from grid coords directly (no dependence on points).
        pillar_center_x = self.x_min + (uix.to(dtype) + 0.5) * self.vx
        pillar_center_y = self.y_min + (uiy.to(dtype) + 0.5) * self.vy
        center_offset_x = kept_pts[:, 0] - pillar_center_x[kept_group]
        center_offset_y = kept_pts[:, 1] - pillar_center_y[kept_group]

        augmented = torch.cat(
            [
                kept_pts,  # x, y, z, intensity
                cluster_offset,  # x_c, y_c, z_c
                center_offset_x.unsqueeze(-1),  # x_p
                center_offset_y.unsqueeze(-1),  # y_p
            ],
            dim=-1,
        )  # (K, NUM_FEATURES)

        sel_b = ub[sorted_group[keep_point]]
        sel_pid = local_pillar_id[sorted_group[keep_point]]
        sel_slot = point_slot[keep_point]
        pillars[sel_b, sel_pid, sel_slot] = augmented

        kb = ub[keep_pillar]
        klid = local_pillar_id[keep_pillar]
        pillar_coords[kb, klid, 0] = uix[keep_pillar]
        pillar_coords[kb, klid, 1] = uiy[keep_pillar]

        return {
            "pillars": pillars,
            "pillar_coords": pillar_coords,
        }


def _smoke_test():
    """
    Smoke test for PointpillarLite.
    """
    params = MmpercParams()
    voxelizer = PointpillarLite(params=params, max_points_per_pillar=5, max_pillars=10)

    # Random input: (B, N, 4)
    points = torch.rand((2, 20, 4)) * torch.tensor([50.0, 50.0, 5.0, 1.0])  # x,y,z,intensity
    output = voxelizer(points)
    print(f"pillars.shape: {output['pillars'].shape}, pillar_coords.shape: {output['pillar_coords'].shape}")


if __name__ == "__main__":
    _smoke_test()
