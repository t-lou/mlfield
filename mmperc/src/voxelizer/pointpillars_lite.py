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
        points: (B, N, 4) tensor  # x, y, z, intensity
        """

        points = points[..., :4]  # ensure only first 4 channels are used
        B, N, C = points.shape
        device = points.device
        assert C == 4, points.shape

        # 1. Flatten for filtering
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

        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        # 2. Voxel indices
        ix = ((pts[:, 0] - self.x_min) / self.vx).long()
        iy = ((pts[:, 1] - self.y_min) / self.vy).long()

        coords = torch.stack([batch_idx, ix, iy], dim=1)  # (B*N, 3)

        # 3. Unique pillars
        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
        # unique_coords: (P, 3) → [batch, ix, iy]

        # 4. Allocate buffers
        pillars = torch.zeros(
            (B, self.max_pillars, self.max_points_per_pillar, C),
            dtype=torch.float32,
            device=device,
        )
        pillar_count = torch.zeros(B, self.max_pillars, dtype=torch.long, device=device)
        pillar_coords = torch.zeros(B, self.max_pillars, 2, dtype=torch.long, device=device)

        # per-batch pillar index
        next_pillar_id = torch.zeros(B, dtype=torch.long, device=device)

        # 5. Fill pillars
        for gid, (b, x, y) in enumerate(unique_coords):
            b = int(b.item())

            pid = int(next_pillar_id[b].item())
            if pid >= self.max_pillars:
                continue

            pillar_coords[b, pid] = torch.tensor([x, y], device=device)

            mask = (inverse == gid)
            pts_in_pillar = pts[mask]  # (K, 4)

            count = min(pts_in_pillar.size(0), self.max_points_per_pillar)
            pillars[b, pid, :count] = pts_in_pillar[:count]
            pillar_count[b, pid] = count

            next_pillar_id[b] += 1

        return {
            "pillars": pillars,          # (B, P, M, 4)
            "pillar_coords": pillar_coords,
            "pillar_count": pillar_count,
        }
