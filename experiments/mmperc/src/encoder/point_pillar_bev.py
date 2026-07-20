import logging

from torch import Tensor, nn

import common.params as params
from backbone.tiny_bev_backbone import TinyBEVBackbone
from encoder.simple_pfn import SimplePFN
from scatter.scatter import scatter_to_bev
from voxelizer.pointpillar_lite import PointpillarLite


class PointPillarBEV(nn.Module):
    """
    Full lidar → BEV encoder:
        - voxelization (PointPillars-style)
        - PFN feature extraction
        - scatter to BEV grid
        - lightweight BEV backbone

    Output:
        (B, params.BEV_CHANNELS, BEV_H/2, BEV_W/2)
    """

    def __init__(self) -> None:
        super().__init__()

        # Raw point cloud → pillars
        self.voxelizer = PointpillarLite()

        # Pillar Feature Network (per-pillar feature extraction)
        self.pfn = SimplePFN(in_channels=4, out_channels=64)

        # BEV backbone (expands 64 → params.BEV_CHANNELS)
        self.backbone = TinyBEVBackbone(out_channels=params.BEV_CHANNELS)

        # Precomputed BEV grid resolution
        self.bev_h = params.BEV_H
        self.bev_w = params.BEV_W

    def forward(self, points: Tensor) -> Tensor:
        """
        Args:
            points: (B, N, 5)
                Raw lidar points: x, y, z, intensity, timestamp

        Returns:
            BEV feature map: (B, params.BEV_CHANNELS, H/2, W/2)
        """

        # 1. Voxelization
        vox = self.voxelizer(points)
        pillars = vox["pillars"]  # (B, P, M, C_in)
        pillar_coords = vox["pillar_coords"]  # (B, P, 2)
        logging.debug(f"pillars.shape: {pillars.shape}, pillar_coords.shape: {pillar_coords.shape}")

        # 2. PFN → per-pillar features
        pillar_feats = self.pfn(pillars)  # (B, P, 64)
        logging.debug(f"pillar_feats.shape: {pillar_feats.shape}")

        # 3. Scatter to BEV grid
        bev = scatter_to_bev(
            pillar_feats,
            pillar_coords,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
        )  # (B, 64, H, W)
        logging.debug(f"bev.shape: {bev.shape}")

        # 4. BEV backbone, downsampling H/2, W/2
        bev_backbone = self.backbone(bev)
        logging.debug(f"bev_backbone.shape: {bev_backbone.shape}")

        return bev_backbone
