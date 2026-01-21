from typing import Dict, Tuple

import common.params as params
import torch
from torch import Tensor


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
        x_range: Tuple[float, float] = params.X_RANGE,
        y_range: Tuple[float, float] = params.Y_RANGE,
        z_range: Tuple[float, float] = params.Z_RANGE,
        voxel_size: Tuple[float, float, float] = params.VOXEL_SIZE,
        max_points_per_pillar: int = 20,
        max_pillars: int = 12000,
    ) -> None:
        # Spatial bounds
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

        # Voxel size
        self.vx, self.vy, self.vz = voxel_size

        # Pillar limits
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

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
        # Ensure only x, y, z, intensity are used
        points = points[..., :4]
        B, N, C = points.shape
        assert C == 4, f"Expected 4 channels (x,y,z,intensity), got {C}"
        device = points.device

        # ------------------------------------------------------------
        # 1. Flatten for filtering and voxel coordinate computation
        # ------------------------------------------------------------
        pts = points.reshape(B * N, C)

        valid_mask = (
            (pts[:, 0] >= self.x_min)
            & (pts[:, 0] < self.x_max)
            & (pts[:, 1] >= self.y_min)
            & (pts[:, 1] < self.y_max)
            & (pts[:, 2] >= self.z_min)
            & (pts[:, 2] < self.z_max)
        )

        # Zero out invalid points (keeps indexing simple)
        pts[~valid_mask] = 0.0

        # Batch index for each point
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        # ------------------------------------------------------------
        # 2. Compute voxel indices (ix, iy)
        # ------------------------------------------------------------
        ix = ((pts[:, 0] - self.x_min) / self.vx).long()
        iy = ((pts[:, 1] - self.y_min) / self.vy).long()

        coords = torch.stack([batch_idx, ix, iy], dim=1)  # (B*N, 3)

        # ------------------------------------------------------------
        # 3. Unique pillars (per batch)
        # ------------------------------------------------------------
        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
        # unique_coords: (P, 3) → [batch, ix, iy]

        # ------------------------------------------------------------
        # 4. Allocate pillar buffers
        # ------------------------------------------------------------
        pillars = torch.zeros(
            (B, self.max_pillars, self.max_points_per_pillar, C),
            dtype=torch.float32,
            device=device,
        )
        pillar_count = torch.zeros(B, self.max_pillars, dtype=torch.long, device=device)
        pillar_coords = torch.zeros(B, self.max_pillars, 2, dtype=torch.long, device=device)

        next_pillar_id = torch.zeros(B, dtype=torch.long, device=device)

        # ------------------------------------------------------------
        # 5. Fill pillars
        # ------------------------------------------------------------
        for gid, (b, x, y) in enumerate(unique_coords):
            b = int(b.item())
            pid = int(next_pillar_id[b].item())

            if pid >= self.max_pillars:
                continue  # skip overflow

            pillar_coords[b, pid] = torch.tensor([x, y], device=device)

            mask = inverse == gid
            pts_in_pillar = pts[mask]  # (K, 4)

            count = min(pts_in_pillar.size(0), self.max_points_per_pillar)
            pillars[b, pid, :count] = pts_in_pillar[:count]
            pillar_count[b, pid] = count

            next_pillar_id[b] += 1

        return {
            "pillars": pillars,  # (B, P, M, 4)
            "pillar_coords": pillar_coords,  # (B, P, 2)
            "pillar_count": pillar_count,  # (B, P)
        }
