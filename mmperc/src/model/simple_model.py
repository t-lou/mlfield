import torch
import torch.nn as nn
from torch import Tensor

import common.params as params
from encoder.point_pillar_bev import PointPillarBEV
from encoder.tiny_camera_encoder import TinyCameraEncoder
from fusion.futr_fusion import FuTrFusionBlock
from head.bbox2d_head import BBox2dHead
from head.semantics_head import FullResSemHead


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

        # BBox head (CenterNet-style)
        # Predicts object centers: (B, 1, H, W)
        # and
        # Regression head
        # Predicts: dx, dy, log(w), log(l), sin(yaw), cos(yaw)
        # Shape: (B, 6, H, W)
        self.bbox_head = BBox2dHead(bev_channels)

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
        camera_tokens, cam_feat = self.cam_encoder(images)

        # ---------------------------------------------------------
        # 3. BEV–camera fusion
        # ---------------------------------------------------------
        bev_fused: Tensor = self.fusion(lidar_token, camera_tokens)  # (B, C, H, W)

        # ---------------------------------------------------------
        # 4. Detection heads
        # ---------------------------------------------------------

        # BBox2d
        bbox_features = self.bbox_head(bev_fused)
        # Heatmap prediction (sigmoid → probability)
        bbox_heatmap = torch.sigmoid(bbox_features["heatmap"])
        # Regression prediction (raw values)
        bbox_reg = bbox_features["reg"]

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
