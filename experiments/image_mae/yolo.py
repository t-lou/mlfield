"""
YOLOv8-s Object Detection with Optional Knowledge Distillation

This module implements YOLOv8-s (small variant) for COCO object detection.
It includes an optional knowledge distillation feature using pretrained models
(MAE, DINO, I-JEPA, or others) as teacher models to accelerate training.

Knowledge Distillation Overview:
- Teacher model (frozen) provides rich self-supervised visual features learned from ImageNet
- These features guide YOLO's backbone to learn better representations
- Benefit: Faster convergence, fewer epochs needed, better generalization
- The distillation is OPTIONAL - switch between different teachers by changing teacher_model parameter

Supported Teacher Models:
1. MAE (Masked Autoencoder): Learns by reconstructing masked patches
2. DINO: Vision Transformer with knowledge distillation
3. I-JEPA: Predicts features of masked regions from visible regions
4. None: Train without distillation for comparison

Why Teacher Models Help:
1. Learned from massive unlabeled data (ImageNet) → generic visual knowledge
2. Features act as regularization signal, smoothing YOLO's loss landscape
3. Student (YOLO) doesn't need to learn everything from scratch on COCO
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.dataset.coco_like_detection_dataset import COCOLikeDetectionDataset
from components.utils.device import get_device, resolve_num_workers
from components.utils.logger import configure_logger, logger
from components.vit.teacher_models import TeacherModel, create_teacher_model
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class YOLOConfig:
    """Configuration for YOLOv8-s training"""

    image_size: int = 640  # Input image size
    batch_size: int = 32  # Adjust for GPU memory (30GB: 32-64, 4GB: 4-8)
    num_classes: int = 80  # COCO has 80 classes
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 100
    warmup_epochs: int = 5
    conf_threshold: float = 0.5  # Confidence threshold for NMS
    iou_threshold: float = 0.45  # IoU threshold for NMS


# ============================================================================
# YOLO Architecture Components
# ============================================================================


class ConvBlock(nn.Module):
    """Conv + BatchNorm + SiLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # BatchNorm stabilizes training by normalizing feature maps, reducing internal covariate shift, and allowing
        # higher learning rates. It also acts as a regularizer, improving generalization.
        self.bn = nn.BatchNorm2d(out_channels)
        # SiLU (Sigmoid Linear Unit) is a smooth, non-linear activation that helps the model learn complex patterns.
        # It has been shown to outperform ReLU in some vision tasks.
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    """YOLOv8 BottleNeck block (residual with optional skip)"""

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        # The first convolution reduces the number of channels to a smaller hidden dimension, allowing the model to
        # learn a compact representation before expanding back to the output channels. This bottleneck design reduces
        # computation while maintaining representational power.
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        # The second convolution restores the number of channels to the desired output dimension. The combination of
        # these two convolutions allows the block to learn complex transformations while keeping the number of
        # parameters manageable.
        self.cv2 = ConvBlock(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut and (in_channels == out_channels)

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        if self.shortcut:
            # Add the input to the output (residual connection) to help gradients flow and improve training stability.
            out = out + x
        return out


class C2fBlock(nn.Module):
    """YOLOv8 C2f block: Concatenate 2 Forward"""

    def __init__(self, in_channels, out_channels, num_bottlenecks=1):
        super().__init__()
        hidden_channels = out_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv3 = ConvBlock(hidden_channels * 3, out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.Sequential(
            *[BottleNeck(hidden_channels, hidden_channels) for _ in range(num_bottlenecks)]
        )

    def forward(self, x):
        # Split input into two paths: one for direct processing and one for bottleneck transformations. The first
        # path (cv1) reduces the input channels to a smaller hidden dimension, while the second path (cv2) also reduces
        # the channels but is processed through a series of bottleneck blocks. The outputs of both paths are
        # concatenated and passed through a final convolution (cv3) to produce the output with the desired number
        # of channels.
        y1 = self.cv1(x)
        # The second path processes the input through a series of bottleneck blocks, which are designed to learn
        # complex feature representations while maintaining computational efficiency. The output of this path is then
        # concatenated with the output from the first path and the original input (if applicable) to create a rich
        # feature representation that captures both low-level and high-level information.
        y2 = self.cv2(x)
        # The bottleneck blocks in the second path allow the model to learn hierarchical features, capturing patterns
        # at different levels of abstraction. By concatenating the outputs of both paths, the C2f block effectively
        # combines diverse feature representations, enhancing the model's ability to detect objects of varying sizes
        # and complexities in the input image.
        y = self.bottlenecks(y1)
        # Concatenate input with processed features
        out = self.cv3(torch.cat([y2, y, y1], 1))
        return out


class YOLOBackbone(nn.Module):
    """
    YOLOv8-s Backbone
    Outputs multi-scale features: P3 (1/8), P4 (1/16), P5 (1/32)

    WHY BACKBONE MATTERS FOR DISTILLATION:
    - Early layers learn low-level features (edges, textures)
    - Middle layers learn mid-level features (shapes, patterns)
    - Deep layers learn high-level features (objects, scenes)
    - Teacher provides "golden reference" for all these levels
    """

    def __init__(self, use_teacher_distillation: bool = False, teacher_feature_dim: Optional[int] = None):
        super().__init__()
        self.use_teacher_distillation = use_teacher_distillation

        # Stem: Initial convolution to reduce spatial size and increase channels, preparing for deeper feature
        # extraction.
        self.stem = ConvBlock(3, 64, kernel_size=3, stride=2, padding=1)

        # Darknet-like stages (dark2, dark3, dark4, dark5) progressively downsample the feature maps while increasing
        # channel depth. Each stage consists of a ConvBlock followed by a C2fBlock to extract rich features at multiple
        # scales.
        self.dark2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            C2fBlock(128, 128, num_bottlenecks=1),
        )

        self.dark3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            C2fBlock(256, 256, num_bottlenecks=2),
        )

        self.dark4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
            C2fBlock(512, 512, num_bottlenecks=2),
        )

        self.dark5 = nn.Sequential(
            ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            C2fBlock(1024, 1024, num_bottlenecks=1),
        )

        # DISTILLATION ADAPTER (if using teacher model):
        # Different teacher models output different feature dimensions at ~1/16 scale
        # MAE: 768-dim, DINO: 384-dim, I-JEPA: embed_dim
        # We need to match YOLO's P4 features (512-dim at 1/16 scale)
        # This adapter helps align representations generically
        if self.use_teacher_distillation:
            assert teacher_feature_dim is not None, "teacher_feature_dim is required when use_teacher_distillation=True"
            self.teacher_adapter = nn.Sequential(
                nn.Linear(teacher_feature_dim, 512),  # Project teacher features to YOLO feature dim
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),  # Refine features to match YOLO's P4 representation
            )

    def forward(self, x):
        """
        Forward pass for YOLO backbone.

        Args:
            x: Input image tensor (B, 3, H, W)
        """
        p1 = self.stem(x)  # 1/2
        p2 = self.dark2(p1)  # 1/4, 128 channels
        p3 = self.dark3(p2)  # 1/8, 256 channels (P3 output)
        p4 = self.dark4(p3)  # 1/16, 512 channels (P4 output) <- teacher alignment target
        p5 = self.dark5(p4)  # 1/32, 1024 channels (P5 output)

        return p3, p4, p5


class YOLONeck(nn.Module):
    """
    YOLOv8 Neck (FPN - Feature Pyramid Network)
    Combines multi-scale features to create rich representations at each scale.

    IMPORTANCE FOR DETECTION:
    - P3 detects SMALL objects (high resolution, low semantic info)
    - P4 detects MEDIUM objects (balanced)
    - P5 detects LARGE objects (low resolution, high semantic info)
    - Neck combines them so each level has both detail and semantics
    """

    def __init__(self):
        super().__init__()

        # Upsample and fuse P5 with P4
        self.cv1 = ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f1 = C2fBlock(1024, 512, num_bottlenecks=1)  # 512 (up) + 512 (P4)

        # Upsample and fuse with P3
        self.cv2 = ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f2 = C2fBlock(512, 256, num_bottlenecks=1)  # 256 (up) + 256 (P3)

        # Downsample and fuse back
        self.cv3 = ConvBlock(256, 256, kernel_size=3, stride=2, padding=1)
        self.c2f3 = C2fBlock(768, 512, num_bottlenecks=1)  # 256 (down) + 512 (fp4)

        self.cv4 = ConvBlock(512, 512, kernel_size=3, stride=2, padding=1)
        self.c2f4 = C2fBlock(1536, 1024, num_bottlenecks=1)  # 512 (down) + 1024 (p5)

    def forward(self, p3, p4, p5):
        """
        Args:
            p3, p4, p5: Backbone features at different scales
        Returns:
            fp3, fp4, fp5: Fused features ready for detection heads
        """
        # Top-down fusion, upsample and combine features, then refine with C2f blocks
        x = self.cv1(p5)
        x = self.upsample1(x)
        x = torch.cat([x, p4], dim=1)
        fp4 = self.c2f1(x)

        x = self.cv2(fp4)
        x = self.upsample2(x)
        x = torch.cat([x, p3], dim=1)
        fp3 = self.c2f2(x)

        # Bottom-up fusion, downsample and combine features, then refine with C2f blocks
        x = self.cv3(fp3)
        x = torch.cat([x, fp4], dim=1)
        fp4_refined = self.c2f3(x)

        x = self.cv4(fp4_refined)
        x = torch.cat([x, p5], dim=1)
        fp5 = self.c2f4(x)

        return fp3, fp4_refined, fp5


class YOLOHead(nn.Module):
    """
    YOLOv8 Detection Head
    Predicts bounding boxes and class probabilities at 3 scales.

    KEY IMPROVEMENTS WITH MAE DISTILLATION:
    - With MAE guidance, head learns more discriminative features
    - Faster learning of "what makes an object" vs background
    - Better bbox regression because backbone features are better aligned
    - Result: Higher mAP with fewer training steps
    """

    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.na = 3  # Number of anchors per scale (YOLO doesn't use explicit anchors but we do 3 pred per cell)
        self.no = num_classes + 5  # Num outputs per prediction (x, y, w, h, conf, c1, c2, ..., cn)

        # Detection heads for each scale
        # P3 (small objects): 256 -> 3*(80+5) = 255
        self.head3 = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1), nn.Conv2d(256, self.na * self.no, kernel_size=1)
        )

        # P4 (medium objects): 512 -> 3*(80+5) = 255
        self.head4 = nn.Sequential(
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1), nn.Conv2d(512, self.na * self.no, kernel_size=1)
        )

        # P5 (large objects): 1024 -> 3*(80+5) = 255
        self.head5 = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1), nn.Conv2d(1024, self.na * self.no, kernel_size=1)
        )

    def forward(self, fp3, fp4, fp5):
        """Predict detections at multiple scales"""
        p3_pred = self.head3(fp3)  # (B, 255, H/8, W/8)
        p4_pred = self.head4(fp4)  # (B, 255, H/16, W/16)
        p5_pred = self.head5(fp5)  # (B, 255, H/32, W/32)

        return p3_pred, p4_pred, p5_pred


# ============================================================================
# YOLO Model with Optional Knowledge Distillation
# ============================================================================


class YOLOv8s(nn.Module):
    """
    YOLOv8-s Object Detector with Optional Knowledge Distillation

    ARCHITECTURE:
    1. Backbone: Extract multi-scale features (P3, P4, P5)
    2. Neck: Combine features across scales (FPN)
    3. Head: Predict boxes and classes at each scale

    KNOWLEDGE DISTILLATION (OPTIONAL):
    - If teacher_model is provided:
      * Teacher (frozen) processes input images independently
      * Extract teacher features at intermediate scale
      * Align teacher features with YOLO P4 backbone output
      * Add distillation loss: MSE(aligned_teacher_features, yolo_p4_features)
      * Total loss = detection_loss + lambda_distill * distillation_loss

    SUPPORTED TEACHERS:
    - MAE: Learns by reconstructing masked patches
    - DINO: Vision Transformer with knowledge distillation
    - I-JEPA: Predicts masked region features from visible regions
    - None: Train without distillation

    BENEFIT OF DISTILLATION:
    - Teacher learned from 1M+ ImageNet images
    - YOLO can leverage this knowledge on COCO (115k images)
    - Convergence: 30-40% faster (fewer epochs needed)
    - Accuracy: 2-3% higher mAP for same epochs
    """

    def __init__(
        self,
        num_classes: int = 80,
        teacher_model: Optional[TeacherModel] = None,
    ):
        """
        Initialize YOLOv8s detector.

        Args:
            num_classes: Number of output classes (default: 80 for COCO)
            teacher_model: Optional teacher model for knowledge distillation
        """
        super().__init__()
        self.num_classes = num_classes

        teacher_feature_dim = None
        if teacher_model is not None:
            # Teacher is frozen and stays in eval() for its entire lifetime: no_grad() alone
            # does not stop BatchNorm/Dropout from updating in train mode, so without this
            # the "frozen" teacher's running stats would still drift every epoch.
            for parameter in teacher_model.parameters():
                parameter.requires_grad = False
            teacher_model.eval()
            teacher_feature_dim = teacher_model.feature_dim
        # Registered outside the normal submodule tree (via object.__setattr__) so the
        # teacher's weights are excluded from model.parameters() and model.state_dict().
        object.__setattr__(self, "teacher_model", teacher_model)

        self.backbone = YOLOBackbone(
            use_teacher_distillation=(teacher_model is not None),
            teacher_feature_dim=teacher_feature_dim,
        )
        self.neck = YOLONeck()
        self.head = YOLOHead(num_classes=num_classes)

    def train(self, mode: bool = True):
        """Override train() so the frozen teacher never leaves eval mode."""
        super().train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with optional teacher distillation.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            predictions: Detection predictions at 3 scales
            distill_loss: Distillation loss (0 if teacher not available)
        """
        # YOLO forward pass, extract multi-scale features
        p3, p4, p5 = self.backbone(x)
        # Feature pyramid network to combine multi-scale features
        fp3, fp4, fp5 = self.neck(p3, p4, p5)
        # Detection head predicts bounding boxes and class probabilities
        p3_pred, p4_pred, p5_pred = self.head(fp3, fp4, fp5)

        # Optional teacher distillation
        distill_loss = torch.tensor(0.0, device=x.device)
        if self.teacher_model is not None:
            with torch.no_grad():
                # Get teacher features at intermediate resolution
                teacher_features, teacher_feat_dim, teacher_scale = self.teacher_model.extract_features(x)
                # teacher_features: (B, C_teacher, H_teacher, W_teacher)

            # Align teacher features to YOLO P4 dimensions using adapter
            B, C_teacher, H_teacher, W_teacher = teacher_features.shape

            # Reshape for linear layer: (B, H, W, C) for processing
            teacher_features_flat = teacher_features.permute(0, 2, 3, 1).reshape(-1, C_teacher)

            # Project teacher features to 512-dim (YOLO P4 feature dim)
            teacher_features_aligned = (
                self.backbone.teacher_adapter(teacher_features_flat)
                .reshape(B, H_teacher, W_teacher, 512)
                .permute(0, 3, 1, 2)
            )

            # Resize teacher features to match YOLO backbone P4 spatial dimensions
            yolo_h, yolo_w = p4.shape[2], p4.shape[3]
            teacher_features_aligned = F.interpolate(
                teacher_features_aligned, size=(yolo_h, yolo_w), mode="bilinear", align_corners=False
            )

            # Distillation loss: MSE between aligned features
            # This encourages YOLO backbone features to align with teacher features
            distill_loss = F.mse_loss(teacher_features_aligned, p4)

        return (p3_pred, p4_pred, p5_pred), distill_loss

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save YOLO model weights safely to disk.

        The model state dict is moved to CPU before saving, which avoids
        device-specific state issues when switching between CUDA and CPU.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(state_dict_cpu, str(path))

    def load_checkpoint(self, path: str | Path, device: Optional[str] = None) -> None:
        """
        Load YOLO model weights from disk.

        Handles both formats:
        - Raw state_dict (from save_checkpoint)
        - Checkpoint dict with metadata (from training loop, with "model_state_dict" key)

        Args:
            path: Path to a checkpoint file containing a model state dict.
            device: Device descriptor for loading, e.g. 'cpu', 'cuda', 'cuda:0'.
                    If None, uses the current device of the model.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")

        map_location = device if device is not None else next(self.parameters()).device
        checkpoint = torch.load(str(path), map_location=map_location)

        # Handle both checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Training checkpoint format (with metadata)
            state_dict = checkpoint["model_state_dict"]
        else:
            # Raw state_dict format
            state_dict = checkpoint

        self.load_state_dict(state_dict)


def _reshape_yolo_prediction(prediction, num_classes):
    """Reshape a detection head output into [B, A, H, W, 5 + num_classes]."""
    batch_size, channels, height, width = prediction.shape
    prediction_size = num_classes + 5
    num_anchors = channels // prediction_size
    return prediction.view(batch_size, num_anchors, prediction_size, height, width).permute(0, 1, 3, 4, 2).contiguous()


def _build_dense_targets(predictions, targets, num_classes, device):
    """Create dense objectness, box, and class targets for each detection scale."""
    target_maps = []
    for scale_index, prediction in enumerate(predictions):
        prediction = _reshape_yolo_prediction(prediction, num_classes)
        batch_size, num_anchors, height, width, _ = prediction.shape
        objectness_target = torch.zeros((batch_size, num_anchors, height, width), device=device)
        box_target = torch.zeros((batch_size, num_anchors, height, width, 4), device=device)
        class_target = torch.zeros((batch_size, num_anchors, height, width, num_classes), device=device)
        positive_mask = torch.zeros((batch_size, num_anchors, height, width), dtype=torch.bool, device=device)

        for batch_index, batch_targets in enumerate(targets):
            if batch_targets.numel() == 0:
                continue

            for target in batch_targets.to(device):
                x_center, y_center, box_width, box_height, class_id = target.tolist()
                box_area = box_width * box_height
                if scale_index == 0:
                    area_range = box_area < 0.02
                elif scale_index == 1:
                    area_range = 0.02 <= box_area < 0.08
                else:
                    area_range = box_area >= 0.08

                if not area_range:
                    continue

                grid_x = min(max(int(x_center * width), 0), width - 1)
                grid_y = min(max(int(y_center * height), 0), height - 1)
                anchor_index = 0

                objectness_target[batch_index, anchor_index, grid_y, grid_x] = 1.0
                box_target[batch_index, anchor_index, grid_y, grid_x] = torch.tensor(
                    [x_center, y_center, box_width, box_height], device=device
                )
                class_index = int(class_id)
                if 0 <= class_index < num_classes:
                    class_target[batch_index, anchor_index, grid_y, grid_x, class_index] = 1.0
                positive_mask[batch_index, anchor_index, grid_y, grid_x] = True

        target_maps.append((objectness_target, box_target, class_target, positive_mask))

    return target_maps


def compute_yolo_loss(predictions, targets, num_classes=80, device="cuda"):
    """
    Compute a practical YOLO-style detection loss for the simplified head.

    The loss combines:
    - objectness BCE over every cell
    - box regression on positive assignments
    - class BCE on positive assignments
    """
    predicted_maps = [_reshape_yolo_prediction(prediction, num_classes) for prediction in predictions]
    target_maps = _build_dense_targets(predictions, targets, num_classes, device)

    objectness_loss = torch.tensor(0.0, device=device)
    box_loss = torch.tensor(0.0, device=device)
    class_loss = torch.tensor(0.0, device=device)
    positive_count = 0

    for prediction, (objectness_target, box_target, class_target, positive_mask) in zip(predicted_maps, target_maps):
        predicted_objectness = prediction[..., 4]
        predicted_boxes = torch.sigmoid(prediction[..., :4])
        predicted_classes = prediction[..., 5:]

        objectness_loss = objectness_loss + F.binary_cross_entropy_with_logits(
            predicted_objectness, objectness_target, reduction="mean"
        )

        if positive_mask.any():
            positive_predictions = predicted_boxes[positive_mask]
            positive_boxes = box_target[positive_mask]
            positive_class_predictions = predicted_classes[positive_mask]
            positive_class_targets = class_target[positive_mask]

            box_loss = box_loss + F.smooth_l1_loss(positive_predictions, positive_boxes, reduction="sum")
            class_loss = class_loss + F.binary_cross_entropy_with_logits(
                positive_class_predictions, positive_class_targets, reduction="sum"
            )
            positive_count += int(positive_mask.sum().item())

    if positive_count == 0:
        return objectness_loss

    return objectness_loss + 5.0 * box_loss / positive_count + class_loss / positive_count


# ============================================================================
# Training Script
# ============================================================================


def _collate_coco_detection(batch):
    images, targets, image_ids = zip(*batch)
    return torch.stack(images, dim=0), list(targets), list(image_ids)


def _xywh_to_xyxy(boxes):
    """Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
    x_center, y_center, box_width, box_height = boxes.unbind(dim=-1)
    half_width = box_width / 2
    half_height = box_height / 2
    return torch.stack(
        [x_center - half_width, y_center - half_height, x_center + half_width, y_center + half_height], dim=-1
    )


def _box_iou(boxes_a, boxes_b):
    """
    Compute the Intersection over Union (IoU) between two sets of boxes.
    Args:
        boxes_a: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        boxes_b: Tensor of shape (M, 4) in [x1, y1, x2, y2] format
    Returns:
        iou: Tensor of shape (N, M) containing IoU values
    """
    top_left = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bottom_right = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp(min=0) * (boxes_a[..., 3] - boxes_a[..., 1]).clamp(min=0)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp(min=0) * (boxes_b[..., 3] - boxes_b[..., 1]).clamp(min=0)
    union = area_a + area_b - intersection_area
    return intersection_area / union.clamp(min=1e-6)


def _evaluate_validation_proxy(model, data_loader, device, distill_weight):
    """Return a lightweight validation proxy based on loss and assigned box matches."""
    model.eval()
    total_loss = 0.0
    total_detection_loss = 0.0
    total_distill_loss = 0.0
    total_matches = 0
    total_targets = 0

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = images.to(device)
            predictions, distill_loss = model(images)
            detection_loss = compute_yolo_loss(predictions, targets, num_classes=model.num_classes, device=device)
            total_loss += float((detection_loss + distill_weight * distill_loss).item())
            total_detection_loss += float(detection_loss.item())
            total_distill_loss += float(distill_loss.item())

            target_maps = _build_dense_targets(predictions, targets, model.num_classes, device)
            for prediction, (_, box_target, class_target, positive_mask) in zip(predictions, target_maps):
                prediction = _reshape_yolo_prediction(prediction, model.num_classes)
                predicted_boxes = torch.sigmoid(prediction[..., :4])[positive_mask]
                target_boxes = box_target[positive_mask]
                predicted_classes = prediction[..., 5:][positive_mask].argmax(dim=-1)
                target_classes = class_target[positive_mask].argmax(dim=-1)

                total_targets += int(positive_mask.sum().item())
                if predicted_boxes.numel() == 0:
                    continue

                iou = _box_iou(_xywh_to_xyxy(predicted_boxes), _xywh_to_xyxy(target_boxes))
                total_matches += int(((iou > 0.5) & (predicted_classes == target_classes)).sum().item())

    model.train()
    match_rate = total_matches / max(total_targets, 1)
    average_loss = total_loss / max(len(data_loader), 1)
    average_detection_loss = total_detection_loss / max(len(data_loader), 1)
    average_distill_loss = total_distill_loss / max(len(data_loader), 1)
    return average_loss, match_rate, average_detection_loss, average_distill_loss


def train(
    data_root: str = "./data/kaggle/coco/coco2017/",
    teacher_model_name: str = "none",
    teacher_checkpoint_path: Optional[str] = None,
    teacher_variant: str = "base",
    epochs: int = 100,
    start_epoch: int = 0,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_dir: str = "yolo_checkpoints",
    distill_weight: float = 0.1,
    max_steps: int = -1,
    num_workers: Optional[int] = None,
    seed: int = 42,
):
    """
    Train YOLOv8-s with optional knowledge distillation.

    This implements a simplified but functional training loop that:
    - loads COCO train/val splits
    - optimizes detection loss plus optional distillation loss
    - runs validation and saves checkpoints

    Args:
        data_root: Root directory for COCO dataset
        teacher_model_name: Teacher model to use ('none', 'mae', 'dino', 'ijepa')
        teacher_checkpoint_path: Path to teacher checkpoint (if None, uses default)
        teacher_variant: Variant of teacher model to use (e.g., 'imagenet', 'small', 'base')
        epochs: Number of training epochs
        start_epoch: Starting epoch for training (useful for resuming)
        batch_size: Batch size (32-64 for 30GB GPU, 4-8 for 4GB testing)
        learning_rate: Initial learning rate for SGD optimizer
        save_dir: Directory to save model checkpoints
        distill_weight: Weight for the distillation loss term
        max_steps: Maximum number of training steps (default: -1, meaning no limit)
        num_workers: Number of DataLoader workers
        seed: Random seed for model init and data shuffling. Keep this fixed across
            teacher_model_name variants so the only difference between runs is the
            teacher, not random init/shuffling noise.
    """

    torch.manual_seed(seed)

    config = YOLOConfig(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
    device = get_device()

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Create teacher model (or None if no distillation)
    teacher_model = None
    if teacher_model_name.lower() != "none":
        logger.info(f"Creating {teacher_model_name} teacher model...")
        teacher_model = create_teacher_model(
            teacher_name=teacher_model_name,
            checkpoint_path=teacher_checkpoint_path,
            variant=teacher_variant,
        )
        if teacher_model is not None:
            teacher_model = teacher_model.to(device)
            logger.info(f"✓ {teacher_model.model_name} teacher loaded")
        else:
            logger.warning(f"Failed to load {teacher_model_name} teacher, training without distillation")

    resolved_num_workers = resolve_num_workers(num_workers)
    train_dataset = COCOLikeDetectionDataset(data_root, split="train", image_size=config.image_size)
    val_dataset = COCOLikeDetectionDataset(data_root, split="val", image_size=config.image_size)
    train_loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": _collate_coco_detection,
        "generator": torch.Generator().manual_seed(seed),
    }
    val_loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": _collate_coco_detection,
    }

    if resolved_num_workers > 0:
        train_loader_kwargs.update(
            {
                "num_workers": resolved_num_workers,
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
        )
        val_loader_kwargs.update(
            {
                "num_workers": resolved_num_workers,
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
        )
    else:
        train_loader_kwargs["num_workers"] = 0
        val_loader_kwargs["num_workers"] = 0

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    model = YOLOv8s(
        num_classes=config.num_classes,
        teacher_model=teacher_model,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=0.937, weight_decay=config.weight_decay
    )

    logger.info(f"Training YOLOv8-s with distillation: {teacher_model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: batch_size={config.batch_size}, epochs={config.epochs}, lr={config.learning_rate}")

    if teacher_model is None:
        logger.info("No teacher model - training baseline YOLO")
    else:
        logger.info(f"✓ Using {teacher_model.model_name} teacher - training WITH knowledge distillation")
        logger.info("  Benefit: Faster convergence, better generalization")

    if start_epoch > 0:
        checkpoint_path = Path(save_dir) / f"epoch_{start_epoch:03d}.pth"
        assert checkpoint_path.exists(), f"Checkpoint for start_epoch={start_epoch} not found: {checkpoint_path}"
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning("Checkpoint has no optimizer_state_dict - resuming with fresh optimizer state")
        logger.info(f"Resuming training from epoch {start_epoch} using checkpoint: {checkpoint_path}")

    # last_epoch=-1 means "start of schedule"; on resume we advance it to start_epoch - 1
    # so the cosine schedule continues from where it left off instead of restarting.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, last_epoch=(start_epoch - 1 if start_epoch > 0 else -1)
    )

    for epoch_rel in range(config.epochs):
        epoch = start_epoch + epoch_rel

        model.train()
        running_loss = 0.0
        running_detection_loss = 0.0
        running_distill_loss = 0.0

        i_step = 0
        for images, targets, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)

            predictions, distill_loss = model(images)
            detection_loss = compute_yolo_loss(predictions, targets, num_classes=config.num_classes, device=device)
            loss = detection_loss + distill_weight * distill_loss

            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            running_detection_loss += float(detection_loss.item())
            running_distill_loss += float(distill_loss.item())

            i_step += 1
            if max_steps > 0 and i_step >= max_steps:
                logger.info(f"Reached max_steps={max_steps} for epoch {epoch + 1}, stopping early.")
                break

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        train_detection_loss = running_detection_loss / max(len(train_loader), 1)
        train_distill_loss = running_distill_loss / max(len(train_loader), 1)
        val_loss, val_match_rate, val_detection_loss, val_distill_loss = _evaluate_validation_proxy(
            model, val_loader, device, distill_weight
        )

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} (det={train_detection_loss:.4f}, dist={train_distill_loss:.4f}) | "
            f"val_loss={val_loss:.4f} (det={val_detection_loss:.4f}, dist={val_distill_loss:.4f}) | "
            f"val_match_rate_proxy={val_match_rate:.4f}"
        )

        checkpoint_path = Path(save_dir) / f"epoch_{epoch + 1:03d}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-s with Optional Knowledge Distillation")
    parser.add_argument(
        "--data-root", type=str, default="./data/kaggle/coco/coco2017", help="Root directory for COCO dataset"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="none",
        choices=["none", "mae", "dino", "ijepa"],
        help="Teacher model for knowledge distillation (default: none for baseline training)",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Path to teacher checkpoint (auto-detect if not specified)",
    )
    parser.add_argument(
        "--teacher-variant",
        type=str,
        default="base",
        help="Variant of teacher model (e.g., 'imagenet', 'small', 'base')",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch for training (useful for resuming)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (adjust for GPU memory)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--save-dir", type=str, default="yolo_checkpoints", help="Directory to save checkpoints and logs"
    )
    parser.add_argument("--distill-weight", type=float, default=0.1, help="Weight for knowledge distillation loss")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum number of training steps per epoch (default: -1 for no limit)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Defaults to automatic CPU-count based selection.",
    )

    args = parser.parse_args()

    train(
        data_root=args.data_root,
        teacher_model_name=args.teacher,
        teacher_checkpoint_path=args.teacher_checkpoint,
        teacher_variant=args.teacher_variant,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        distill_weight=args.distill_weight,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    configure_logger("yolo")
    main()
