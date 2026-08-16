from torch import nn


class JEPAEncoder(nn.Module):
    """
    Multimodal JEPA encoder:
        - LiDAR (multi-sensor)
        - Cameras (multi-camera)
        - CAN (optional)
    Produces:
        - shared latent representation for SSL
        - optional BEV projection for fine-tuning
    """

    def __init__(
        self,
        lidar_encoder: nn.Module,
        camera_encoder: nn.Module,
        fusion_module: nn.Module,
        bev_projector: nn.Module = None,
    ):
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.camera_encoder = camera_encoder
        self.fusion_module = fusion_module
        self.bev_projector = bev_projector  # used only in fine-tuning

    def forward_ssl(self, lidar_points_list, camera_images_list, cam_meta_list, can_bus=None):
        """
        SSL forward pass (no heads, no BEV projection).
        Uses ALL sensors.

        Args:
            lidar_points_list: list of LiDAR point clouds (one per sensor)
            camera_images_list: list of images (one per camera)
            cam_meta_list: list of calibration dicts (intrinsics/extrinsics)
            can_bus: optional CAN history

        Returns:
            latent: JEPA shared latent representation
        """

        # 1. LiDAR → tokens
        lidar_tokens, lidar_bev = self.lidar_encoder(lidar_points_list)

        # 2. Cameras → spherical tokens (your TinyCameraEncoder)
        cam_tokens = self.camera_encoder(camera_images_list, cam_meta_list)

        # 3. Fusion transformer (JEPA-style)
        latent = self.fusion_module(lidar_tokens, cam_tokens, can_bus)

        return latent

    def forward_finetune(self, latent, mode="front"):
        """
        Fine-tuning forward pass:
            - BEV projection
            - detection/segmentation/e2e heads

        Args:
            latent: JEPA latent from SSL
            mode: "front" or "all" (for datasets with full labels)

        Returns:
            bev_features: BEV tensor for heads
        """

        if self.bev_projector is None:
            raise ValueError("BEV projector required for fine-tuning")

        bev_features = self.bev_projector(latent, mode=mode)
        return bev_features
