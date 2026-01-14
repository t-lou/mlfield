import common.params as params
from backbone.tiny_bev_backbone import TinyBEVBackbone
from encoder.simple_pfn import SimplePFN
from scatter.scatter import scatter_to_bev
from torch import Tensor, nn
from voxelizer.pointpillar_lite import PointpillarLite


class PointPillarBEV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.voxelizer = PointpillarLite()
        self.pfn = SimplePFN(in_channels=4, out_channels=64)
        self.backbone = TinyBEVBackbone(out_channels=params.BEV_CHANNELS)
        self.bev_h = params.BEV_H
        self.bev_w = params.BEV_W

    def forward(self, points) -> Tensor:
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
