import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from components.dataset.coco_like_detection_dataset import COCOLikeDetectionDataset, Mode
from components.utils.device import get_device, resolve_num_workers
from components.utils.logger import configure_logger, logger
from components.vit.teacher_models import create_teacher_model
from components.vit.vit_encoder import VitEncoder


class DetrEncoder(nn.Module):
    def __init__(self, embed_dim=384, num_layers=6, num_heads=6):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DetrDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_queries=100, num_layers=6, num_heads=6):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, memory):
        B = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        x = queries
        for layer in self.layers:
            x = layer(x, memory)
        return x


class DetrHead(nn.Module):
    def __init__(self, num_classes, embed_dim=384):
        super().__init__()
        self.class_head = nn.Linear(embed_dim, num_classes + 1)  # +1 for "no object"
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid(),  # DeTr outputs normalized boxes
        )

    def forward(self, x):
        return {
            "pred_logits": self.class_head(x),
            "pred_boxes": self.bbox_head(x),
        }


def box_cxcywh_to_xyxy(boxes):
    """
    Convert normalized center-based boxes (cx, cy, w, h)
    into corner format (x_min, y_min, x_max, y_max), for IoU/GIoU.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized IoU between two sets of boxes.
    boxes1: (N, 4)
    boxes2: (M, 4)
    """
    # Intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Union
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=1e-6)

    # Smallest enclosing box
    x1_c = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    y1_c = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    x2_c = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    y2_c = torch.max(boxes1[:, None, 3], boxes2[:, 3])

    area_c = (x2_c - x1_c).clamp(min=0) * (y2_c - y1_c).clamp(min=0)

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


def generalized_box_iou_loss(pred_boxes, tgt_boxes):
    """
    GIoU loss = 1 - GIoU
    """
    giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes))
    return 1.0 - giou.diag().mean()


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost=1, bbox_cost=5, giou_cost=2):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    def forward(self, pred_logits, pred_boxes, targets):
        # Convert to CPU numpy for scipy
        bs, num_queries = pred_logits.shape[:2]
        indices = []

        for b in range(bs):
            tgt_boxes = targets[b]["boxes"]
            tgt_labels = targets[b]["labels"]

            if tgt_boxes.numel() == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue

            out_prob = pred_logits[b].softmax(-1)
            out_bbox = pred_boxes[b]

            # classification cost
            class_cost = -out_prob[:, tgt_labels]

            # bbox L1 cost
            bbox_cost = torch.cdist(out_bbox, tgt_boxes, p=1)

            # giou cost
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_boxes),
            )

            cost = self.class_cost * class_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost

            i, j = linear_sum_assignment(cost.detach().cpu())
            indices.append((torch.as_tensor(i), torch.as_tensor(j)))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, ce_weight=1.0, bbox_weight=5.0, giou_weight=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.giou_loss = generalized_box_iou_loss
        self.ce_weight = ce_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        indices = self.matcher(pred_logits, pred_boxes, targets)

        bs, num_queries = pred_logits.shape[:2]
        loss_ce = pred_logits.new_zeros(())
        loss_bbox = pred_logits.new_zeros(())
        loss_giou = pred_logits.new_zeros(())

        for b, (idx_pred, idx_tgt) in enumerate(indices):
            tgt = targets[b]

            target_classes = pred_logits.new_full((num_queries,), self.num_classes, dtype=torch.long)
            if idx_pred.numel() > 0:
                target_classes[idx_pred] = tgt["labels"][idx_tgt].to(pred_logits.device)
            loss_ce += self.ce_loss(pred_logits[b], target_classes)

            if idx_pred.numel() > 0:
                loss_bbox += self.l1_loss(pred_boxes[b][idx_pred], tgt["boxes"][idx_tgt])
                loss_giou += self.giou_loss(pred_boxes[b][idx_pred], tgt["boxes"][idx_tgt])

        loss_ce = loss_ce / bs
        loss_bbox = loss_bbox / bs
        loss_giou = loss_giou / bs
        loss = self.ce_weight * loss_ce + self.bbox_weight * loss_bbox + self.giou_weight * loss_giou

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss": loss,
        }


class DeTr(nn.Module):
    def __init__(
        self,
        backbone: VitEncoder,
        encoder: DetrEncoder,
        decoder: DetrDecoder,
        head: DetrHead,
        teacher_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def forward(self, imgs):
        feats = self.backbone.forward_detr(imgs)  # (B, C, H, W)
        B, C, H, W = feats.shape
        feats_seq = feats.flatten(2).transpose(1, 2)  # (B, HW, C)

        memory = self.encoder(feats_seq)
        hs = self.decoder(memory)
        out = self.head(hs)

        distill_loss = feats.new_zeros(())
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_feats = self.teacher_model.forward_detr(imgs)
            distill_loss = nn.functional.mse_loss(feats, teacher_feats)

        return out, distill_loss


def _collate_coco_detection(batch):
    images, targets, image_ids = zip(*batch)
    return torch.stack(images, dim=0), list(targets), list(image_ids)


def _abs_xywh_to_norm_cxcywh(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert absolute pixel (x, y, w, h) boxes (top-left) to normalized (cx, cy, w, h) in [0, 1].

    COCOLikeDetectionDataset's DETR mode returns absolute pixel-space boxes scaled to
    image_size (when image_size is set); DetrHead predicts normalized center-based boxes,
    so targets need this conversion before matching/loss. reshape(-1, 4) also normalizes
    the shape of an empty (0,) boxes tensor (no annotations) to (0, 4).
    """
    boxes = boxes.reshape(-1, 4)
    x, y, w, h = boxes.unbind(-1)
    cx = (x + w / 2) / image_size
    cy = (y + h / 2) / image_size
    w_n = w / image_size
    h_n = h / image_size
    return torch.stack([cx, cy, w_n, h_n], dim=-1)


@torch.no_grad()
def _evaluate_validation_proxy(model, val_loader, criterion, device, distill_weight, image_size):
    model.eval()
    running_loss = 0.0
    running_detection_loss = 0.0
    running_distill_loss = 0.0
    running_matched = 0
    running_total_preds = 0

    for images, targets, _ in val_loader:
        images = images.to(device)
        targets = [
            {
                "boxes": _abs_xywh_to_norm_cxcywh(t["boxes"].to(device), image_size),
                "labels": t["labels"].to(device),
            }
            for t in targets
        ]

        predictions, distill_loss = model(images)
        detection_loss = criterion(predictions, targets)["loss"]
        loss = detection_loss + distill_weight * distill_loss

        running_loss += float(loss.item())
        running_detection_loss += float(detection_loss.item())
        running_distill_loss += float(distill_loss.item())

        pred_labels = predictions["pred_logits"].argmax(-1)
        num_classes = predictions["pred_logits"].shape[-1] - 1
        running_matched += int((pred_labels != num_classes).sum().item())
        running_total_preds += pred_labels.numel()

    n_batches = max(len(val_loader), 1)
    val_loss = running_loss / n_batches
    val_detection_loss = running_detection_loss / n_batches
    val_distill_loss = running_distill_loss / n_batches
    val_match_rate = running_matched / max(running_total_preds, 1)

    model.train()
    return val_loss, val_match_rate, val_detection_loss, val_distill_loss


def train(
    data_root: str = "./data/kaggle/coco/coco2017/",
    teacher_model_name: str = "none",
    teacher_checkpoint_path: Optional[str] = None,
    teacher_variant: str = "base",
    epochs: int = 100,
    start_epoch: int = 0,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_dir: str = "detr_checkpoints",
    distill_weight: float = 0.1,
    num_workers: Optional[int] = None,
    seed: int = 42,
):
    """
    Train DeTr with optional knowledge distillation (not implemented)

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

    image_size = 640
    num_classes = 80
    weight_decay = 1e-4
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
    train_dataset = COCOLikeDetectionDataset(data_root, split="train", image_size=image_size, mode=Mode.DETR)
    val_dataset = COCOLikeDetectionDataset(data_root, split="val", image_size=image_size, mode=Mode.DETR)
    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": _collate_coco_detection,
        "generator": torch.Generator().manual_seed(seed),
    }
    val_loader_kwargs = {
        "batch_size": batch_size,
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

    model = DeTr(
        backbone=VitEncoder(add_cls_token=False),
        encoder=DetrEncoder(),
        decoder=DetrDecoder(),
        head=DetrHead(num_classes=num_classes),
        teacher_model=teacher_model,
    ).to(device)

    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.937, weight_decay=weight_decay)

    if teacher_model is None:
        logger.info("No teacher model - training baseline DeTr")
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
        optimizer, T_max=epochs, last_epoch=(start_epoch - 1 if start_epoch > 0 else -1)
    )

    for epoch_rel in range(epochs):
        epoch = start_epoch + epoch_rel

        model.train()
        running_loss = 0.0
        running_detection_loss = 0.0
        running_distill_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = [
                {
                    "boxes": _abs_xywh_to_norm_cxcywh(t["boxes"].to(device), image_size),
                    "labels": t["labels"].to(device),
                }
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            predictions, distill_loss = model(images)
            detection_loss = criterion(predictions, targets)["loss"]
            loss = detection_loss + distill_weight * distill_loss

            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            running_detection_loss += float(detection_loss.item())
            running_distill_loss += float(distill_loss.item())

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        train_detection_loss = running_detection_loss / max(len(train_loader), 1)
        train_distill_loss = running_distill_loss / max(len(train_loader), 1)
        val_loss, val_match_rate, val_detection_loss, val_distill_loss = _evaluate_validation_proxy(
            model, val_loader, criterion, device, distill_weight, image_size
        )

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
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
    parser = argparse.ArgumentParser(description="DeTr with Optional Knowledge Distillation")
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
        "--save-dir", type=str, default="detr_checkpoints", help="Directory to save checkpoints and logs"
    )
    parser.add_argument("--distill-weight", type=float, default=0.1, help="Weight for knowledge distillation loss")
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
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    configure_logger("detr")
    main()
