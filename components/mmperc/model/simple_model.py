import torch.nn as nn
from torch import Tensor

from components.definitions.mmperc import MmpercParams
from components.mmperc.encoder.point_pillar_bev import PointPillarBEV
from components.mmperc.encoder.tiny_camera_encoder import TinyCameraEncoder
from components.mmperc.fusion.futr_fusion import FuTrFusionBlock
from components.mmperc.head.bbox2d_head import BBox2dHead
from components.mmperc.head.semantics_head import FullResSemHead


class SimpleModel(nn.Module):
    """
    Multimodal BEV detection model:
    - Lidar → BEV feature map
    - Camera → token embeddings
    - Cross-attention fusion
    - BEV detection heads (heatmap + regression)
    """

    def __init__(self, params: MmpercParams) -> None:
        super().__init__()

        assert params.use_lidar or params.use_camera, "Hey, both lidar and camera off."

        # ---------------------------------------------------------
        # 1. Lidar encoder → BEV feature map
        # ---------------------------------------------------------
        if params.use_lidar:
            self.lidar_encoder = PointPillarBEV(params=params)  # (B, C, H, W)

        # ---------------------------------------------------------
        # 2. Camera encoder → token embeddings
        # ---------------------------------------------------------
        if params.use_camera:
            self.cam_encoder = TinyCameraEncoder(params=params)  # (B, N_cam, C)

        # ---------------------------------------------------------
        # 3. Fusion block (BEV <-> camera tokens)
        # ---------------------------------------------------------
        self._use_fusion = params.use_lidar and params.use_camera
        if self._use_fusion:
            self.fusion = FuTrFusionBlock(params=params)

        # ---------------------------------------------------------
        # 4. Detection heads
        # ---------------------------------------------------------

        # BBox head (CenterNet-style)
        # Predicts object centers: (B, 1, H, W)
        # and
        # Regression head
        # Predicts: dx, dy, log(w), log(l), sin(yaw), cos(yaw)
        # Shape: (B, 6, H, W)

        if params.pred_bbox:
            self.bbox_head = BBox2dHead(params.bev_params.bev_channels)

        # Semantic segmentation head
        if params.pred_semantics:
            self.sem_head = FullResSemHead(
                in_channels=self.cam_encoder.out_channels, num_classes=params.num_sem_classes
            )

        self._params = params

    def forward(self, points: Tensor, images: Tensor) -> dict:
        """
        Forward pass of the multi-task BEV + camera fusion model.

        Args:
            points:
                LiDAR point cloud tensor of shape (B, N, 4),
                where N is the number of points and each point is (x, y, z, intensity).

            images:
                RGB camera images of shape (B, 3, H_img, W_img).

        Returns:
            dict with the following keys:

                "bbox_heatmap":
                    Tensor of shape (B, 1, H_bev, W_bev)
                    CenterNet-style heatmap predicting object centers in BEV.

                "bbox_reg":
                    Tensor of shape (B, 6, H_bev, W_bev)
                    Regression outputs for each BEV cell:
                        [dx, dy, log(w), log(l), sin(yaw), cos(yaw)].

                "sem_logits":
                    Tensor of shape (B, C_sem, H_img', W_img')
                    Per-pixel semantic segmentation logits from the camera branch.
                    (H_img', W_img' depend on the camera encoder/decoder resolution.)

        Notes:
            - LiDAR is encoded into a BEV feature map.
            - Camera images are encoded into tokens + feature maps.
            - Fusion combines BEV and camera tokens.
            - Detection heads operate on fused BEV features.
            - Semantic head operates on camera feature maps.
        """
        lidar_token = None
        camera_tokens = None
        cam_feat = None

        # ---------------------------------------------------------
        # 1. Lidar → BEV feature map
        # ---------------------------------------------------------
        if self._params.use_lidar:
            lidar_token: Tensor = self.lidar_encoder(points)  # (B, C, H, W)

        # ---------------------------------------------------------
        # 2. Camera → tokens
        # ---------------------------------------------------------
        if self._params.use_camera:
            camera_tokens, cam_feat = self.cam_encoder(images)

        # ---------------------------------------------------------
        # 3. BEV–camera fusion
        # ---------------------------------------------------------
        if self._use_fusion:
            bev_fused: Tensor = self.fusion(lidar_token, camera_tokens)  # (B, C, H, W)
        elif self._params.use_lidar:
            bev_fused = lidar_token
        else:
            bev_fused = camera_tokens

        # Prepare output
        outputs = {}

        # ---------------------------------------------------------
        # 4. Detection heads
        # ---------------------------------------------------------

        # BBox2d predictions
        if hasattr(self, "bbox_head"):
            bbox_features = self.bbox_head(bev_fused)
            bbox_heatmap, bbox_reg = bbox_features["heatmap"], bbox_features["reg"]
            outputs["bbox_heatmap"] = bbox_heatmap
            outputs["bbox_reg"] = bbox_reg

        # Semantic segmentation prediction
        if hasattr(self, "sem_head"):
            sem_feature = self.sem_head(cam_feat)
            outputs["sem_logits"] = sem_feature

        return outputs
