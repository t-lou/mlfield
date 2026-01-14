from scatter.scatter import scatter_to_bev
from torch import nn


class PointPillarsBEV(nn.Module):
    def __init__(self, voxelizer, pfn, backbone, bev_h, bev_w):
        super().__init__()
        self.voxelizer = voxelizer
        self.pfn = pfn
        self.backbone = backbone
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, points):
        """
        points: (B, N, 5)
        """
        vox = self.voxelizer(points)
        pillars = vox["pillars"]  # (B, P, M, C_in)
        pillar_coords = vox["pillar_coords"]  # (B, P, 2)

        pillar_feats = self.pfn(pillars)  # (B, P, C_out)

        bev = scatter_to_bev(
            pillar_feats,
            pillar_coords,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
        )  # (B, C_out, H, W)

        x = self.backbone(bev)  # TinyBEVBackbone
        return x
