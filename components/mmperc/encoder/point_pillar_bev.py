from torch import Tensor, nn

from components.definitions.mmperc import MmpercParams
from components.mmperc.backbone.tiny_bev_backbone import TinyBEVBackbone
from components.mmperc.encoder.simple_pfn import SimplePFN
from components.mmperc.scatter.scatter import scatter_to_bev
from components.mmperc.voxelizer.pointpillar_lite import PointpillarLite
from components.utils.logger import logger


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

    def __init__(self, params: MmpercParams) -> None:
        super().__init__()

        # Raw point cloud → pillars
        self.voxelizer = PointpillarLite()

        # Pillar Feature Network (per-pillar feature extraction)
        self.pfn = SimplePFN(in_channels=4, out_channels=64)

        # BEV backbone (expands 64 → params.BEV_CHANNELS)
        self.backbone = TinyBEVBackbone(out_channels=params.bev_params.bev_channels)

        # Precomputed BEV grid resolution
        self.bev_h = params.bev_params.bev_h
        self.bev_w = params.bev_params.bev_w

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
        logger.debug(f"pillars.shape: {pillars.shape}, pillar_coords.shape: {pillar_coords.shape}")

        # 2. PFN → per-pillar features
        pillar_feats = self.pfn(pillars)  # (B, P, 64)
        logger.debug(f"pillar_feats.shape: {pillar_feats.shape}")

        # 3. Scatter to BEV grid
        bev = scatter_to_bev(
            pillar_feats,
            pillar_coords,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
        )  # (B, 64, H, W)
        logger.debug(f"bev.shape: {bev.shape}")

        # 4. BEV backbone, downsampling H/2, W/2
        bev_backbone = self.backbone(bev)
        logger.debug(f"bev_backbone.shape: {bev_backbone.shape}")

        return bev_backbone
