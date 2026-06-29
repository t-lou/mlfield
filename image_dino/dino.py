import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

DEFAULT_VIT_NAME = "vit_small_patch16_224"


global_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
)

local_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(96, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
)


def multicrop_augment(img):
    crops = []
    for _ in range(2):
        crops.append(global_transform(img))
    for _ in range(8):
        crops.append(local_transform(img))
    return crops  # list of 10 tensors


class Imagenet256Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = list(Path(root_dir).rglob("*.jpg")) + list(Path(root_dir).rglob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # transform must return a LIST of crops
        crops = self.transform(image)  # list of tensors
        return torch.stack(crops)  # shape [Nc, C, H, W]


class ViTBackbone(nn.Module):
    def __init__(self, model_name=DEFAULT_VIT_NAME):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=False)
        # assume vit.forward_features(x) returns CLS + patch tokens

    def forward(self, x):
        feats = self.vit.forward_features(x)  # [B, D]
        return feats  # CLS token embedding


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, nlayers=3):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(nlayers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False  # fixed norm

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOModel(nn.Module):
    def __init__(self, vit_name=DEFAULT_VIT_NAME, out_dim=65536):
        super().__init__()
        self.backbone = ViTBackbone(vit_name)
        dim = self.backbone.vit.num_features
        self.head = DINOHead(dim, out_dim=out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


class DINOLoss(nn.Module):
    def __init__(self, out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs, global_crop_indices=[0, 1]):
        student_out = [s / self.student_temp for s in student_outputs]
        teacher_out = [(t - self.center) / self.teacher_temp for t in teacher_outputs]

        student_probs = [F.log_softmax(s, dim=-1) for s in student_out]
        teacher_probs = [F.softmax(t, dim=-1) for t in teacher_out]

        total_loss = 0
        n_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            t_view = global_crop_indices[t_idx]

            for s_idx, s_prob in enumerate(student_probs):
                if s_idx == t_view:
                    continue

                total_loss += torch.sum(-t_prob * s_prob, dim=-1).mean()
                n_terms += 1

        total_loss /= n_terms

        batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total_loss


class DINOSession(nn.Module):
    def __init__(
        self, vit_name=DEFAULT_VIT_NAME, out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__()
        self.model = DINOModel(vit_name=vit_name, out_dim=out_dim)
        self.loss_fn = DINOLoss(
            out_dim=out_dim, teacher_temp=teacher_temp, student_temp=student_temp, center_momentum=center_momentum
        )

        self.student = DINOModel()
        self.teacher = DINOModel()

        # teacher starts as copy of student, no grad
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, student_inputs, teacher_inputs):
        student_outputs = [self.student(x) for x in student_inputs]
        teacher_outputs = [self.teacher(x) for x in teacher_inputs]
        loss = self.loss_fn(student_outputs, teacher_outputs)
        return loss

    def update_teacher(self, momentum=0.996):
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.data = momentum * pt.data + (1 - momentum) * ps.data


def train(num_epochs=100, bs=64, vit_name=DEFAULT_VIT_NAME):
    dataset = Imagenet256Dataset(root_dir="../data/kaggle/imagenet", transform=multicrop_augment)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)

    dino_session = DINOSession(vit_name=vit_name)
    student = dino_session.student
    teacher = dino_session.teacher
    criterion = dino_session.loss_fn.cuda()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.04)

    for _ in range(num_epochs):
        for imgs in loader:
            # imgs: list of crops per sample → we need to stack per crop
            # Suppose dataset returns list-of-crops already:
            # imgs is [B, 10, C, H, W]
            B, Nc, C, H, W = imgs.shape
            imgs = imgs.cuda()

            # split crops
            crops = [imgs[:, i] for i in range(Nc)]
            global_crops = crops[:2]
            local_crops = crops[2:]
            # local_crops used implicitly, as DINO loss is computed between student and teacher outputs for all crops,
            # but teacher only sees global crops
            _ = local_crops

            # student on all crops
            student_outputs = []
            for crop in crops:
                student_outputs.append(student(crop))

            # teacher on global crops only
            with torch.no_grad():
                teacher_outputs = []
                for crop in global_crops:
                    teacher_outputs.append(teacher(crop))

            loss = criterion(student_outputs, teacher_outputs, global_crop_indices=[0, 1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dino_session.update_teacher(momentum=0.996)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="DINO Training")
    argument_parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    argument_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    argument_parser.add_argument(
        "--vit-name", type=str, default=DEFAULT_VIT_NAME, help="ViT model name for the backbone"
    )
    args = argument_parser.parse_args()
    train(args.num_epochs, args.batch_size, args.vit_name)
