import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =========================
# 1. Shared backbone
# =========================


class SimpleBackbone(nn.Module):
    """
    Minimal CNN backbone.
    Replace with ResNet, Swin, ViT, etc. as needed.
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, feat_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(feat_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x  # [B, feat_dim, H, W]


# =========================
# 2. Task-specific heads
# =========================


class SegmentationHead(nn.Module):
    """
    Example segmentation head.
    Assumes feature map spatial size ~= input size (or small).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_out = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, feats):
        x = F.relu(self.bn1(self.conv1(feats)))
        x = self.conv_out(x)  # [B, num_classes, H, W]
        return x


class DepthHead(nn.Module):
    """
    Example regression head (e.g., depth).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_out = nn.Conv2d(in_channels, 1, 1)

    def forward(self, feats):
        x = F.relu(self.bn1(self.conv1(feats)))
        x = self.conv_out(x)  # [B, 1, H, W]
        return x


# =========================
# 3. Full multi-task model
# =========================


class MultiTaskModel(nn.Module):
    """
    Shared backbone + two task heads.
    Example tasks: segmentation and depth.
    """

    def __init__(self, backbone: nn.Module, head_a: nn.Module, head_b: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head_a = head_a  # e.g., segmentation
        self.head_b = head_b  # e.g., depth

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        out_a = self.head_a(feats)
        out_b = self.head_b(feats)
        return {
            "task_a": out_a,
            "task_b": out_b,
        }


# =========================
# 4. Uncertainty-weighted multi-task loss
# =========================


class MTLLoss(nn.Module):
    """
    Kendall et al. uncertainty-based weighting for 2 tasks.
    L = sum_i ( 1/(2*sigma_i^2) * L_i + log sigma_i )
    We parameterize log_sigma_i to keep sigma positive.
    """

    def __init__(self, loss_fn_a, loss_fn_b, init_log_sigma_a: float = 0.0, init_log_sigma_b: float = 0.0):
        super().__init__()
        self.loss_fn_a = loss_fn_a
        self.loss_fn_b = loss_fn_b

        # log sigma are learnable scalars
        self.log_sigma_a = nn.Parameter(torch.tensor(init_log_sigma_a))
        self.log_sigma_b = nn.Parameter(torch.tensor(init_log_sigma_b))

    def forward(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute per-task base losses
        loss_a_raw = self.loss_fn_a(preds["task_a"], targets["task_a"])
        loss_b_raw = self.loss_fn_b(preds["task_b"], targets["task_b"])

        # Compute weighted losses
        sigma_a = torch.exp(self.log_sigma_a)
        sigma_b = torch.exp(self.log_sigma_b)

        loss_a = (1.0 / (2.0 * sigma_a**2)) * loss_a_raw + self.log_sigma_a
        loss_b = (1.0 / (2.0 * sigma_b**2)) * loss_b_raw + self.log_sigma_b

        total_loss = loss_a + loss_b

        # For logging
        stats = {
            "loss_a_raw": float(loss_a_raw.detach().cpu().item()),
            "loss_b_raw": float(loss_b_raw.detach().cpu().item()),
            "sigma_a": float(sigma_a.detach().cpu().item()),
            "sigma_b": float(sigma_b.detach().cpu().item()),
            "loss_total": float(total_loss.detach().cpu().item()),
        }
        return total_loss, stats


# =========================
# 5. Example dataset stub
# =========================


class DummyMTLDataset(Dataset):
    """
    Minimal example dataset.
    Replace with real dataset (e.g., NYUv2, Cityscapes, etc.).
    """

    def __init__(self, n_samples: int = 1000, img_size: int = 64, num_classes: int = 5):
        super().__init__()
        self.n = n_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Dummy RGB image
        x = torch.randn(3, self.img_size, self.img_size)

        # Dummy segmentation target (H, W), values in [0, num_classes-1]
        seg = torch.randint(0, self.num_classes, (self.img_size, self.img_size), dtype=torch.long)

        # Dummy depth target (H, W)
        depth = torch.rand(self.img_size, self.img_size)

        targets = {
            "task_a": seg,
            "task_b": depth,
        }
        return x, targets


# =========================
# 6. Gradient logging utilities
# =========================


def compute_grad_norms(model: nn.Module, task_params: Dict[str, nn.Parameter]) -> Dict[str, float]:
    """
    Example gradient norm logging for each task parameter set.
    Here we just log global grad norm for the whole model,
    but you can maintain separate param groups if you want per-task.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    return {"grad_norm_total": total_norm}


# =========================
# 7. Training loop
# =========================


def train_one_epoch(
    model: nn.Module,
    mtl_loss_fn: MTLLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    mtl_loss_fn.train()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()

        # Forward
        preds = model(images)

        # Multi-task loss (uncertainty-weighted)
        loss, stats = mtl_loss_fn(preds, targets)

        # Backward
        loss.backward()

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Gradient logging (global)
        grad_stats = compute_grad_norms(model, {})

        if batch_idx % 20 == 0:
            print(
                f"[Batch {batch_idx}] "
                f"total={stats['loss_total']:.4f} "
                f"seg={stats['loss_a_raw']:.4f} "
                f"depth={stats['loss_b_raw']:.4f} "
                f"sigma_a={stats['sigma_a']:.3f} "
                f"sigma_b={stats['sigma_b']:.3f} "
                f"grad_norm={grad_stats['grad_norm_total']:.3f}"
            )


# =========================
# 8. Putting it all together
# =========================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    img_size = 64

    # Dataset & dataloader
    dataset = DummyMTLDataset(n_samples=200, img_size=img_size, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # Model
    backbone = SimpleBackbone(in_channels=3, feat_dim=128)
    head_seg = SegmentationHead(in_channels=128, num_classes=num_classes)
    head_depth = DepthHead(in_channels=128)

    model = MultiTaskModel(backbone, head_seg, head_depth).to(device)

    # Losses for each task
    # For segmentation: CrossEntropyLoss expects [B, C, H, W] and [B, H, W]
    loss_fn_seg = nn.CrossEntropyLoss()
    # For depth: L1 loss (you can use scale-invariant, etc.)
    loss_fn_depth = nn.L1Loss()

    mtl_loss_fn = MTLLoss(loss_fn_seg, loss_fn_depth).to(device)

    # Optimizer (includes sigma parameters via mtl_loss_fn)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(mtl_loss_fn.parameters()), lr=1e-3)

    # Train a few epochs
    for epoch in range(3):
        print(f"\n=== Epoch {epoch} ===")
        train_one_epoch(model, mtl_loss_fn, dataloader, optimizer, device)


if __name__ == "__main__":
    main()
