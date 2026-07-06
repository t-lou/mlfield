import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from components.utils.device import get_device
from components.utils.logger import configure_logger, logger
from components.vit import VitEncoder

# Try to reuse the MAE dataset.
DEFAULG_DATA_ROOT_DIR = "./data/kaggle/imagenet/"


# Global and local crop transformations for DINO
GLOBAL_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

LOCAL_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(96, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.Resize(96),
        transforms.ToTensor(),
    ]
)


def multicrop_augment(img):
    """Generate multiple crops of the input image for DINO training."""
    crops = []
    # Generate 2 global crops and 8 local crops
    for _ in range(2):
        crops.append(GLOBAL_TRANSFORM(img))
    for _ in range(8):
        crops.append(LOCAL_TRANSFORM(img))
    return crops  # list of 10 tensors


def dino_collate_fn(batch):
    """Collate a batch of multi-crop samples into a list of crop tensors."""
    if not batch:
        return []
    num_crops = len(batch[0])
    if any(len(sample) != num_crops for sample in batch):
        raise ValueError("Inconsistent number of crops in batch")
    return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(num_crops)]


class Imagenet256Dataset(torch.utils.data.Dataset):
    """Dataset for ImageNet 256x256 images with multi-crop augmentation."""

    def __init__(self, root_dir: str, transform: callable):
        """Initialize the dataset with the root directory and transformation."""
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = list(Path(root_dir).rglob("*.jpg")) + list(Path(root_dir).rglob("*.png"))
        logger.info(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get an item from the dataset at the specified index."""
        img_path = self.image_paths[idx]
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            crops = self.transform(image)
        return crops


class ViTBackbone(nn.Module):
    """Vision Transformer backbone for DINO, returning the CLS token embedding."""

    def __init__(self):
        super().__init__()
        # Create a ViT Encoder
        self.vit = VitEncoder(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            qkv_bias=True,
        )
        # assume vit.forward(x) returns CLS + patch tokens

    def forward(self, x):
        feats = self.vit.forward(x)  # [B, D]
        return feats  # CLS token embedding


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, nlayers=3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(nlayers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        """Forward pass through the DINO head, returning normalized logits."""
        x = self.mlp(x)  # features
        x = self.last_layer(x)  # logits
        x = nn.functional.normalize(x, dim=-1)  # L2‑normalize final output
        return x


class DINOModel(nn.Module):
    """DINO model combining the ViT backbone and the DINO head, returning the projected features."""

    def __init__(self, out_dim=65536):
        super().__init__()
        self.backbone = ViTBackbone()
        dim = dim = self.backbone.vit.embed_dim
        self.head = DINOHead(dim, out_dim=out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


class DINOLoss(nn.Module):
    """DINO loss function for self-supervised learning, comparing student and teacher outputs across multiple crops."""

    def __init__(self, out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        """Initialize the DINO loss with temperature parameters and center momentum.

        Args:
            out_dim: Output dimension of the DINO head.
            teacher_temp: Temperature for the teacher outputs.
            student_temp: Temperature for the student outputs.
            center_momentum: Momentum for updating the center of teacher outputs.
        """
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs, global_crop_indices=[0, 1]):
        """Compute the DINO loss between student and teacher outputs.

        Args:
            student_outputs: List of tensors from the student model for each crop.
            teacher_outputs: List of tensors from the teacher model for each global crop.
            global_crop_indices: Indices of the global crops in the student outputs.
        """
        student_out = [s / self.student_temp for s in student_outputs]
        teacher_out = [(t - self.center) / self.teacher_temp for t in teacher_outputs]

        student_probs = [F.log_softmax(s, dim=-1) for s in student_out]
        teacher_probs = [F.softmax(t, dim=-1) for t in teacher_out]

        total_loss = 0
        n_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            # Get the corresponding global crop index for the teacher output
            t_view = global_crop_indices[t_idx]

            # Compute the cross-entropy loss between the teacher and student outputs for all crops except the
            # corresponding global crop
            for s_idx, s_prob in enumerate(student_probs):
                if s_idx == t_view:
                    # Skip the student output corresponding to the same global crop as the teacher output
                    continue

                # Compute the cross-entropy loss between the teacher and student outputs
                total_loss += torch.sum(-t_prob * s_prob, dim=-1).mean()
                n_terms += 1

        total_loss /= n_terms

        # Update the center of the teacher outputs using momentum
        batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)
        # Update the center with momentum to stabilize training
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total_loss


class DINOSession(nn.Module):
    def __init__(
        self,
        out_dim=65536,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
        device=None,
    ):
        """Initialize a DINO training session with student and teacher models, loss function, and parameters."""
        super().__init__()
        self.device = device if device is not None else get_device()
        self.loss_fn = DINOLoss(
            out_dim=out_dim, teacher_temp=teacher_temp, student_temp=student_temp, center_momentum=center_momentum
        ).to(self.device)

        # Create student and teacher models
        self.student = DINOModel(out_dim=out_dim).to(self.device)
        self.teacher = DINOModel(out_dim=out_dim).to(self.device)

        # Initialize teacher with student weights and freeze teacher parameters
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, student_inputs, teacher_inputs):
        """Compute the DINO loss for a batch of student and teacher inputs."""
        # Compute student and teacher outputs
        student_outputs = [self.student(x) for x in student_inputs]
        teacher_outputs = [self.teacher(x) for x in teacher_inputs]
        # Compute the DINO loss
        loss = self.loss_fn(student_outputs, teacher_outputs)
        return loss

    def update_teacher(self, momentum=0.996):
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                # Update teacher parameters with momentum, using the student parameters
                pt.mul_(momentum).add_(ps, alpha=1 - momentum)

    def save(self, path_ckpt: Path):
        torch.save(
            {
                "student": {k: v.cpu() for k, v in self.student.state_dict().items()},
                "teacher": {k: v.cpu() for k, v in self.teacher.state_dict().items()},
                "loss_fn": {k: v.cpu() for k, v in self.loss_fn.state_dict().items()},
            },
            str(path_ckpt),
        )

    def load(self, path_ckpt: Path):
        ckpt = torch.load(str(path_ckpt), map_location=self.device)

        self.student.load_state_dict(ckpt["student"])
        self.teacher.load_state_dict(ckpt["teacher"])
        self.loss_fn.load_state_dict(ckpt["loss_fn"])


def train(num_epochs=100, bs=8, data_root=DEFAULG_DATA_ROOT_DIR, start_epoch=-1):
    """Train the DINO model on ImageNet 256x256 dataset with multi-crop augmentation."""
    if not Path(data_root).exists():
        raise ValueError(f"Data root directory {data_root} does not exist.")

    dir_ckpt = Path("./dino_checkpoints")
    if not dir_ckpt.exists():
        dir_ckpt.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dataset = Imagenet256Dataset(root_dir=data_root, transform=multicrop_augment)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        collate_fn=dino_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    dino_session = DINOSession(device=device)

    if start_epoch > 0 and (dir_ckpt / f"epoch_{start_epoch:03d}.pth").exists():
        dino_session.load(dir_ckpt / f"epoch_{start_epoch:03d}.pth")
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        logger.info(f"Starting training from scratch at epoch {start_epoch}")
        start_epoch = 0

    student = dino_session.student
    teacher = dino_session.teacher
    criterion = dino_session.loss_fn
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.04)

    for epoch in range(start_epoch, num_epochs):
        for imgs in loader:
            imgs = [img.to(device, non_blocking=True) for img in imgs]

            # Student sees all crops, teacher only sees global crops.
            student_outputs = [student(crop) for crop in imgs]
            with torch.no_grad():
                teacher_outputs = [teacher(imgs[i]) for i in range(2)]

            loss = criterion(student_outputs, teacher_outputs, global_crop_indices=[0, 1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dino_session.update_teacher(momentum=0.996)

        dino_session.save(dir_ckpt / f"epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    configure_logger("dino")

    argument_parser = argparse.ArgumentParser(description="DINO Training")
    argument_parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    argument_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    argument_parser.add_argument("--start-epoch", type=int, default=-1, help="Starting epoch for training")
    args = argument_parser.parse_args()
    train(args.num_epochs, args.batch_size, start_epoch=args.start_epoch)
