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
        points: torch tensor (N, 5) with columns [x, y, z, intensity, timestamp]
        """

        # filter by range
        mask = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] < self.y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] < self.z_max)
        )
        points = points[mask]

        # compute voxel indices
        ix = ((points[:, 0] - self.x_min) / self.vx).long()
        iy = ((points[:, 1] - self.y_min) / self.vy).long()

        coords = torch.stack([ix, iy], dim=1)

        # unique pillars
        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)

        # limit pillars
        if unique_coords.size(0) > self.max_pillars:
            unique_coords = unique_coords[: self.max_pillars]

        # allocate pillar buffer
        device = points.device
        pillars = torch.zeros((self.max_pillars, self.max_points_per_pillar, 5), dtype=torch.float32, device=device)
        pillar_count = torch.zeros(self.max_pillars, dtype=torch.long, device=device)

        # fill pillars
        for i in range(points.size(0)):
            pid = inverse[i]
            if pid >= self.max_pillars:
                continue
            cnt = pillar_count[pid]
            if cnt < self.max_points_per_pillar:
                pillars[pid, cnt] = points[i]
                pillar_count[pid] += 1

        return {"pillars": pillars, "pillar_coords": unique_coords, "pillar_count": pillar_count}
