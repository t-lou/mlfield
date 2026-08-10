import os
from enum import Enum

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms


class Mode(Enum):
    YOLO = "yolo"
    DETR = "detr"


class COCOLikeDetectionDataset(Dataset):
    """
    COCO dataset wrapper for object detection.
    Handles loading images and annotations from COCO format.
    """

    def __init__(self, coco_root: str, split: str = "train", image_size: int = 640, mode: Mode = Mode.YOLO):
        """
        Args:
            coco_root: Root directory containing train2017, val2017, annotations
            split: "train" or "val"
            image_size: Target image size for model input
        """
        self.image_size = image_size
        self.split = split
        self.mode = mode

        image_dir = os.path.join(coco_root, f"{split}2017")
        ann_file = os.path.join(coco_root, "annotations", f"instances_{split}2017.json")

        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        # COCO category IDs are sparse/non-contiguous, so map them to [0, num_classes-1].
        self.category_ids = sorted(self.coco.getCatIds())
        self.category_id_to_index = {category_id: index for index, category_id in enumerate(self.category_ids)}

        # Standard COCO normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]

        # Load image
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize image to the target square size used by the model.
        if self.mode == Mode.YOLO:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if self.mode == Mode.YOLO:
            targets = self._format_yolo(anns, orig_w, orig_h)
        else:
            targets = self._format_detr(anns, img_id)

        img = self.transform(img)
        return img, targets, img_id

    def _format_yolo(self, anns, orig_w, orig_h):
        """Return YOLO-style normalized center-based boxes."""

        # Convert annotations to model format: [x_center, y_center, w, h, class_id]
        # The targets are normalized to the resized image size so they stay aligned
        # with the tensor fed to the detector.
        targets = []
        for ann in anns:
            if ann["area"] < 1:  # Skip very small objects
                continue

            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / orig_w
            y_center = (y + h / 2) / orig_h
            w_norm = w / orig_w
            h_norm = h / orig_h

            # Scale coordinates to the resized square image used for training.
            # The image is resized to (image_size, image_size), so boxes are first
            # mapped into that resized coordinate system and then normalized by it.
            scale_x = self.image_size / max(orig_w, 1)
            scale_y = self.image_size / max(orig_h, 1)
            x_center = ((x + w / 2) * scale_x) / self.image_size
            y_center = ((y + h / 2) * scale_y) / self.image_size
            w_norm = (w * scale_x) / self.image_size
            h_norm = (h * scale_y) / self.image_size

            class_id = self.category_id_to_index.get(ann["category_id"])
            if class_id is None:
                continue

            targets.append([x_center, y_center, w_norm, h_norm, class_id])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))

        return targets

    def _format_detr(self, anns, img_id):
        """Return DETR-style absolute XYWH + metadata."""
        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, w, h])
            labels.append(self.category_id_to_index[ann["category_id"]])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
