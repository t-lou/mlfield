import torch.nn as nn
from encoder.tiny_camera_encoder import TinyCameraEncoder
from encoder_3d.point_pillars_bev import PointPillarsBEV
from fusion.futr_fusion import FuTrFusionBlock
from head.drivable_head import DrivableAreaHead
from torch import Tensor


class SimpleModel(nn.Module):
    """
    Full multimodal model:
    - Lidar → BEV feature map
    - Camera → token embeddings
    - Cross-attention fusion
    - Drivable area prediction head
    """

    def __init__(self, bev_channels: int = 128) -> None:
        """
        Args:
            bev_channels:  Channel dimension of BEV features (default 128)
        """
        super().__init__()

        # 3D lidar → BEV feature map
        self.lidar_encoder = PointPillarsBEV()

        # RGB → camera tokens (B, N_cam, bev_channels)
        self.cam_encoder = TinyCameraEncoder(out_channels=bev_channels)

        # BEV–camera fusion block
        self.fusion = FuTrFusionBlock(bev_channels=bev_channels)

        # Final segmentation head
        self.head = DrivableAreaHead(in_channels=bev_channels)

    def forward(self, points: Tensor, images: Tensor) -> Tensor:
        """
        Args:
            points: (B, N, 4) lidar point cloud
            images: (B, 3, H, W) RGB camera images

        Returns:
            mask: (B, 1, H_bev, W_bev) drivable area prediction
        """

        # Lidar → BEV feature map
        bev: Tensor = self.lidar_encoder(points)  # (B, 128, H_bev, W_bev)

        # Camera → tokens
        cam_tokens: Tensor = self.cam_encoder(images)  # (B, N_cam, 128)

        # Cross-attention fusion
        bev_fused: Tensor = self.fusion(bev, cam_tokens)  # (B, 128, H_bev, W_bev)

        # Segmentation head
        mask: Tensor = self.head(bev_fused)

        return mask
