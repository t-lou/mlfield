from dataclasses import dataclass, field

from components.definitions.bev_params import BevParams
from components.definitions.train_config import TrainConfig


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
    # (38 A2D2 classes + 1 for "invalid/unfilled")
    # -----------------------------
    num_sem_classes: int = 39

    # -----------------------------
    # Loss weights for multitask optimization
    # -----------------------------
    # Global multipliers for bbox detection in multitask optimization.
    weight_loss_hm: float = 1.0
    weight_loss_reg: float = 1.0
    # Global multiplier for semantic loss in multitask optimization.
    weight_sem_loss: float = 2.0

    # -----------------------------
    # Semantic loss controls
    # -----------------------------
    # Downscale invalid-class contribution in weighted CE to avoid dominance.
    sem_invalid_ce_weight: float = 0.25
    # Enable per-batch class-balanced CE weights from semantic GT histograms.
    sem_use_class_balanced_ce: bool = True
    # Clamp range for auto-computed CE class weights.
    sem_ce_weight_min: float = 0.2
    sem_ce_weight_max: float = 5.0
    # Auxiliary binary invalid-vs-valid loss (from invalid class logit).
    sem_invalid_aux_weight: float = 0.5
    sem_invalid_bce_pos_weight: float = 2.0

    # -----------------------------
    # Downsampling factor for camera images
    # -----------------------------
    image_scale: float = 0.25

    use_lidar: bool = True
    use_camera: bool = True
    pred_bbox: bool = True
    pred_semantics: bool = True

    train_config: TrainConfig = field(default_factory=TrainConfig)

    path_data: str = "/repo/data/..."
    path_calibration: str = "/repo/data/cams_lidars.json"
    camera_name: str = "front_center"
    lidar_name: str = "front_center"
    use_camera_calibration: bool = True
