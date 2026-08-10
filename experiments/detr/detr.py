import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


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
    def __init__(self, embed_dim=384, num_classes=91):
        super().__init__()
        self.class_head = nn.Linear(embed_dim, num_classes + 1)  # +1 for "no object"
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid(),  # DETR outputs normalized boxes
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
    giou = generalized_box_iou(pred_boxes, tgt_boxes)
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

            i, j = linear_sum_assignment(cost.cpu())
            indices.append((torch.as_tensor(i), torch.as_tensor(j)))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.giou_loss = generalized_box_iou_loss

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        indices = self.matcher(pred_logits, pred_boxes, targets)

        loss_ce = 0
        loss_bbox = 0
        loss_giou = 0

        for b, (idx_pred, idx_tgt) in enumerate(indices):
            tgt = targets[b]

            loss_ce += self.ce_loss(pred_logits[b][idx_pred], tgt["labels"][idx_tgt])
            loss_bbox += self.l1_loss(pred_boxes[b][idx_pred], tgt["boxes"][idx_tgt])
            loss_giou += self.giou_loss(pred_boxes[b][idx_pred], tgt["boxes"][idx_tgt])

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }


class DETR(nn.Module):
    def __init__(self, backbone, encoder, decoder, head):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

    def forward(self, imgs):
        feats = self.backbone.forward_detr(imgs)  # (B, C, H, W)
        B, C, H, W = feats.shape
        feats = feats.flatten(2).transpose(1, 2)  # (B, HW, C)

        memory = self.encoder(feats)
        hs = self.decoder(memory)
        out = self.head(hs)
        return out
