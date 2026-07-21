from dataclasses import dataclass, field

from components.definitions.bev import BevParams
from components.definitions.train import TrainConfig


@dataclass
class MmpercParams:
    # -----------------------------
    # Lidar region of interest (meters)
    # -----------------------------
    bev_params: BevParams = field(default_factory=BevParams)

    # -----------------------------
    # Maximum number of lidar points and GT boxes per frame (for padding)
    # -----------------------------
    num_lidar_points: int = 12000
    num_gt_boxes: int = 200

    # -----------------------------
    # Number of semantic classes
    # -----------------------------
    num_sem_classes: int = 38

    # -----------------------------
    # Downsampling factor for camera images
    # -----------------------------
    image_scale: float = 0.25

    use_lidar: bool = True
    use_camera: bool = True
    pred_bbox: bool = True
    pred_semantics: bool = True

    train_config: TrainConfig = field(default_factory=TrainConfig)
