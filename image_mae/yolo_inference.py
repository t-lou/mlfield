"""
YOLOv8-s Inference Script

Usage:
    python -m image_mae.yolo_inference -i input.jpg -o output.png
    python -m image_mae.yolo_inference -i input.jpg -o output.png --checkpoint yolo_checkpoint/final.pth
    python -m image_mae.yolo_inference -i input.jpg -o output.png --conf-threshold 0.4 --iou-threshold 0.5
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from logger import create_logger
from PIL import Image
from yolo import YOLOv8s

logger = create_logger("yolo_inference", level="INFO")


class YOLOInference:
    """YOLO inference wrapper with NMS post-processing"""

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        num_classes: int = 80,
    ):
        """
        Initialize YOLO inference engine.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            num_classes: Number of classes
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

        # Load model
        self.model = YOLOv8s(num_classes=num_classes, use_mae_distillation=False)
        self.model.to(self.device)

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            logger.warning("⚠️  No checkpoint provided. Using untrained model (weights initialized randomly).")

        self.model.eval()

        # COCO class names (80 classes)
        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.model.load_checkpoint(ckpt_path, device=self.device)

    def preprocess(self, image_path: str, image_size: int = 640) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Load and preprocess image for inference.

        Returns:
            image_tensor: Preprocessed image tensor (1, 3, H, W)
            original_size: (width, height) of original image
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        original_size = img.size  # (width, height)

        # Resize to model input size
        img = img.resize((image_size, image_size), Image.BILINEAR)

        # Convert to tensor and normalize (ImageNet normalization)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, original_size

    def postprocess(self, predictions: Tuple, original_size: Tuple[int, int], image_size: int = 640) -> List[dict]:
        """
        Post-process model predictions: decode, NMS, scale to original size.

        Returns:
            List of detections, each with:
            {
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': (x_min, y_min, x_max, y_max) in original image coordinates
            }
        """
        p3_pred, p4_pred, p5_pred = predictions
        detections = []

        scales = [8, 16, 32]  # Stride for each prediction level
        pred_list = [p3_pred, p4_pred, p5_pred]

        orig_w, orig_h = original_size
        scale_x = orig_w / image_size
        scale_y = orig_h / image_size

        for scale_idx, (pred, stride) in enumerate(zip(pred_list, scales)):
            batch_size, channels, grid_h, grid_w = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)

            # pred[..., 0:4] = [x_center, y_center, w, h] (normalized)
            # pred[..., 4] = objectness (logit)
            # pred[..., 5:] = class logits

            for b in range(batch_size):
                for i in range(grid_h):
                    for j in range(grid_w):
                        for a in range(3):
                            objectness_logit = pred[b, i, j, a, 4]
                            objectness = torch.sigmoid(objectness_logit).item()

                            logger.debug(
                                f"Debug: scale_idx={scale_idx}, b={b}, i={i}, j={j}, a={a}, objectness={objectness:.4f}"
                            )

                            # Decode boxes in the same normalized format used during training.
                            x_center = torch.sigmoid(pred[b, i, j, a, 0]).item()
                            y_center = torch.sigmoid(pred[b, i, j, a, 1]).item()
                            box_w = torch.sigmoid(pred[b, i, j, a, 2]).item()
                            box_h = torch.sigmoid(pred[b, i, j, a, 3]).item()

                            # Convert to pixel coordinates (at model resolution)
                            x_center_px = x_center * image_size
                            y_center_px = y_center * image_size
                            w_px = box_w * image_size
                            h_px = box_h * image_size

                            # Convert to (x_min, y_min, x_max, y_max)
                            x_min = x_center_px - w_px / 2
                            y_min = y_center_px - h_px / 2
                            x_max = x_center_px + w_px / 2
                            y_max = y_center_px + h_px / 2

                            # Clamp to image bounds
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(image_size, x_max)
                            y_max = min(image_size, y_max)

                            # Scale to original image size
                            x_min = int(x_min * scale_x)
                            y_min = int(y_min * scale_y)
                            x_max = int(x_max * scale_x)
                            y_max = int(y_max * scale_y)

                            # Get class scores
                            class_logits = pred[b, i, j, a, 5:]
                            class_scores = torch.softmax(class_logits, dim=0)
                            class_id = class_scores.argmax().item()
                            class_conf = class_scores[class_id].item()

                            # Combined confidence
                            final_conf = objectness * class_conf
                            if final_conf < self.conf_threshold:
                                continue

                            detections.append(
                                {
                                    "class_id": class_id,
                                    "class_name": self.class_names[class_id],
                                    "confidence": final_conf,
                                    "objectness": objectness,
                                    "class_conf": class_conf,
                                    "bbox": (x_min, y_min, x_max, y_max),
                                }
                            )

        logger.info(f"✓ Post-processing complete. {len(detections)} detections before NMS.")

        # Apply NMS
        detections = self._apply_nms(detections)
        logger.info(f"✓ Post-processing complete. {len(detections)} detections after NMS.")

        return detections

    def _apply_nms(self, detections: List[dict], iou_threshold: float = None) -> List[dict]:
        """Apply Non-Maximum Suppression"""
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)

            # Remove detections with high IoU
            detections = [
                d
                for d in detections
                if d["class_id"] != current["class_id"] or self._compute_iou(current["bbox"], d["bbox"]) < iou_threshold
            ]

        return keep

    @staticmethod
    def _compute_iou(box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two boxes (x_min, y_min, x_max, y_max)"""
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    @torch.no_grad()
    def infer(self, image_path: str, image_size: int = 640) -> List[dict]:
        """
        Run inference on image.

        Returns:
            List of detections with bbox, class_id, class_name, and confidence
        """
        img_tensor, original_size = self.preprocess(image_path, image_size)
        predictions, _ = self.model(img_tensor)
        detections = self.postprocess(predictions, original_size, image_size)
        return detections

    @staticmethod
    def draw_detections(image_path: str, detections: List[dict], output_path: str):
        """
        Draw bounding boxes and labels on image and save.

        Args:
            image_path: Input image path
            detections: List of detections
            output_path: Output image path
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Color palette for different classes
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

        for det in detections:
            x_min, y_min, x_max, y_max = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            class_id = det["class_id"]

            # Draw bounding box
            color = tuple(map(int, colors[class_id]))
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y_min = max(y_min - 5, label_size[1] + 5)
            cv2.rectangle(
                img,
                (x_min, label_y_min - label_size[1] - 5),
                (x_min + label_size[0] + 5, label_y_min + 5),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x_min + 2, label_y_min - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Save output image
        cv2.imwrite(output_path, img)
        logger.info(f"✓ Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-s Inference")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input image path")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output image path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detect if not specified)",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--image-size", type=int, default=640, help="Model input size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    assert args.checkpoint is not None
    checkpoint_path = args.checkpoint

    # Initialize inference engine
    logger.info("Initializing YOLO inference engine...")
    inferencer = YOLOInference(
        checkpoint_path=checkpoint_path,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    # Run inference
    logger.info(f"Running inference on: {args.input}")
    detections = inferencer.infer(args.input, image_size=args.image_size)

    logger.info(f"Detected {len(detections)} objects:")
    for det in detections:
        logger.info(f"  - {det['class_name']}: {det['confidence']:.2%} bbox={det['bbox']}")

    # Draw and save results
    logger.info(f"Drawing detections and saving to: {args.output}")
    YOLOInference.draw_detections(args.input, detections, args.output)


if __name__ == "__main__":
    main()
