import common.params as params
import torch
import torch.nn as nn
from common.utils import rescale_image
from encoder.point_pillar_bev import PointPillarBEV
from encoder.tiny_camera_encoder import TinyCameraEncoder
from fusion.futr_fusion import FuTrFusionBlock
from head.semantics_head import FullResSemHead
from torch import Tensor


class SimpleModel(nn.Module):
    """
    Multimodal BEV detection model:
    - Lidar → BEV feature map
    - Camera → token embeddings
    - Cross-attention fusion
    - BEV detection heads (heatmap + regression)
    """

    def __init__(self, bev_channels: int = params.BEV_CHANNELS) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # 1. Lidar encoder → BEV feature map
        # ---------------------------------------------------------
        self.lidar_encoder = PointPillarBEV()  # (B, C, H, W)

        # ---------------------------------------------------------
        # 2. Camera encoder → token embeddings
        # ---------------------------------------------------------
        self.cam_encoder = TinyCameraEncoder()  # (B, N_cam, C)

        # ---------------------------------------------------------
        # 3. Fusion block (BEV <-> camera tokens)
        # ---------------------------------------------------------
        self.fusion = FuTrFusionBlock()

        # ---------------------------------------------------------
        # 4. Detection heads
        # ---------------------------------------------------------

        # Heatmap head (CenterNet-style)
        # Predicts object centers: (B, 1, H, W)
        self.bbox_heatmap_head = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, 1, kernel_size=1),
        )

        # Regression head
        # Predicts: dx, dy, log(w), log(l), sin(yaw), cos(yaw)
        # Shape: (B, 6, H, W)
        self.bbox_reg_head = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, 6, kernel_size=1),
        )

        # Semantic segmentation head
        self.sem_head = FullResSemHead(in_channels=self.cam_encoder.out_channels, num_classes=params.NUM_SEM_CLASSES)

    def forward(self, points: Tensor, images: Tensor) -> dict:
        """
        Args:
            points: (B, N, 4) lidar point cloud
            images: (B, 3, H, W) RGB camera images

        Returns:
            dict with:
                "heatmap": (B, 1, H_bev, W_bev)
                "reg":     (B, 6, H_bev, W_bev)
        """

        # ---------------------------------------------------------
        # 1. Lidar → BEV feature map
        # ---------------------------------------------------------
        lidar_token: Tensor = self.lidar_encoder(points)  # (B, C, H, W)

        # ---------------------------------------------------------
        # 2. Camera → tokens
        # ---------------------------------------------------------
        images = rescale_image(images)
        camera_tokens, cam_feat = self.cam_encoder(images)

        # ---------------------------------------------------------
        # 3. BEV–camera fusion
        # ---------------------------------------------------------
        bev_fused: Tensor = self.fusion(lidar_token, camera_tokens)  # (B, C, H, W)

        # ---------------------------------------------------------
        # 4. Detection heads
        # ---------------------------------------------------------

        # Heatmap prediction (sigmoid → probability)
        bbox_heatmap = torch.sigmoid(self.heatmap_head(bev_fused))

        # Regression prediction (raw values)
        bbox_reg = self.reg_head(bev_fused)

        # Semantic segmentation prediction
        sem_logits = self.sem_head(cam_feat)

        # ---------------------------------------------------------
        # 5. Return multi-task outputs
        # ---------------------------------------------------------
        return {
            "bbox_heatmap": bbox_heatmap,
            "bbox_reg": bbox_reg,
            "sem_logits": sem_logits,
        }
