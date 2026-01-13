import torch


class TorchPillarVoxelizer:
    def __init__(
        self,
        x_range=(-50, 50),
        y_range=(-50, 50),
        z_range=(-5, 3),
        voxel_size=(0.32, 0.32, 8.0),
        max_points_per_pillar=20,
        max_pillars=12000,
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

        self.vx, self.vy, self.vz = voxel_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        self.grid_x = int((self.x_max - self.x_min) / self.vx)
        self.grid_y = int((self.y_max - self.y_min) / self.vy)

    def __call__(self, points):
        """
        points: (B, N, C) tensor
        """

        B, N, C = points.shape
        device = points.device

        # ----------------------------------------------------
        # 1. Filter invalid points (flatten → mask → zero → reshape)
        # ----------------------------------------------------
        pts = points.reshape(B * N, C)

        mask = (
            (pts[:, 0] >= self.x_min)
            & (pts[:, 0] < self.x_max)
            & (pts[:, 1] >= self.y_min)
            & (pts[:, 1] < self.y_max)
            & (pts[:, 2] >= self.z_min)
            & (pts[:, 2] < self.z_max)
        )

        pts[~mask] = 0.0

        # batch index for each point
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        # ----------------------------------------------------
        # 3. Compute voxel indices
        # ----------------------------------------------------
        ix = ((pts[:, 0] - self.x_min) / self.vx).long()
        iy = ((pts[:, 1] - self.y_min) / self.vy).long()

        coords = torch.stack([batch_idx, ix, iy], dim=1)  # (B*N, 3)

        # ----------------------------------------------------
        # 4. Unique pillars per batch
        # ----------------------------------------------------
        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
        # unique_coords: (P, 3) → [batch, ix, iy]

        # ----------------------------------------------------
        # 5. Allocate output buffers
        # ----------------------------------------------------
        pillars = torch.zeros((B, self.max_pillars, self.max_points_per_pillar, C), dtype=torch.float32, device=device)
        pillar_count = torch.zeros(B, self.max_pillars, dtype=torch.long, device=device)
        pillar_coords = torch.zeros(B, self.max_pillars, 2, dtype=torch.long, device=device)

        # ----------------------------------------------------
        # 6. Fill pillars
        # ----------------------------------------------------
        for pid, (b, x, y) in enumerate(unique_coords):
            if pid >= self.max_pillars:
                break

            pillar_coords[b, pid] = torch.tensor([x, y], device=device)

            mask = inverse == pid
            pts_in_pillar = pts[mask]

            count = min(pts_in_pillar.size(0), self.max_points_per_pillar)
            pillars[b, pid, :count] = pts_in_pillar[:count]
            pillar_count[b, pid] = count

        return {
            "pillars": pillars,  # (B, P, M, C)
            "pillar_coords": pillar_coords,  # (B, P, 2)
            "pillar_count": pillar_count,  # (B, P)
        }
