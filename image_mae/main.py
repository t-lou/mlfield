import argparse
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class MAEVariantConfig:
    dataset_size: int
    image_size: int
    patch_size: int
    batch_size: int
    encoder_dim: int
    encoder_depth: int
    encoder_heads: int
    decoder_dim: int
    decoder_depth: int
    decoder_heads: int
    mask_ratio: float
    learning_rate: float


VARIANT_CONFIG = {
    # Fits typical 4GB GPUs for debugging (with AMP).
    "cifar10": MAEVariantConfig(
        dataset_size=50_000,
        image_size=32,
        patch_size=4,
        batch_size=64,
        encoder_dim=384,
        encoder_depth=8,
        encoder_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        decoder_heads=6,
        mask_ratio=0.75,
        learning_rate=1e-3,
    ),
    # MAE-Base style scale for larger GPUs (32-40GB+ suggested).
    "imagenet": MAEVariantConfig(
        dataset_size=1_281_167,
        image_size=224,
        patch_size=16,
        batch_size=32,
        # batch_size=128,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
        learning_rate=1.5e-4,
    ),
    "coco": MAEVariantConfig(
        dataset_size=1_281_167,
        image_size=224,
        patch_size=16,
        batch_size=32,
        # batch_size=128,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
        learning_rate=1.5e-4,
    ),
}


class DummyImageDataset(Dataset):
    def __init__(self, variant: str = "cifar10"):
        cfg = VARIANT_CONFIG[variant]
        self.variant = variant
        self.size = cfg.dataset_size
        self.image_size = cfg.image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random image tensor keeps this script self-contained for pipeline checks.
        return torch.randn(3, self.image_size, self.image_size)


def make_dummy_dataloader(batch_size: int, variant: str):
    ds = DummyImageDataset(variant=variant)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


class ImageOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, *_ = self.dataset[idx]
        return img


def make_cifar10_dataloader(root: str, batch_size: int, num_workers: int = 4):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        ImageOnlyDataset(dataset),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_imagenet_dataloader(root: str, batch_size: int, num_workers: int = 8, train: bool = True):
    # ImageNet must be downloaded and prepared in advance.
    # Expected layout:
    #   <root>/train/<class_name>/*.JPEG
    #   <root>/val/<class_name>/*.JPEG
    # torchvision does not download ImageNet automatically.
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    split = "train" if train else "val"
    dataset = datasets.ImageFolder(
        root=os.path.join(root, split),
        transform=transform,
    )

    return DataLoader(
        ImageOnlyDataset(dataset),
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_coco_dataloader(root: str, batch_size: int, num_workers: int = 8, train: bool = True):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    split = "train" if train else "val"
    image_root = os.path.join(root, f"{split}2017")
    ann_file = os.path.join(root, "annotations", f"instances_{split}2017.json")

    dataset = datasets.CocoDetection(
        root=image_root,
        annFile=ann_file,
        transform=transform,
    )

    return DataLoader(
        ImageOnlyDataset(dataset),
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def build_2d_sincos_position_embedding(grid_size: int, embed_dim: int, add_cls_token: bool = True):
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos position embeddings")

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    pos_h = grid[0].reshape(-1)
    pos_w = grid[1].reshape(-1)

    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000**omega)

    out_h = torch.outer(pos_h, omega)
    out_w = torch.outer(pos_w, omega)
    pos_embed = torch.cat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)

    if add_cls_token:
        cls_pos = torch.zeros(1, embed_dim, dtype=pos_embed.dtype)
        pos_embed = torch.cat([cls_pos, pos_embed], dim=0)

    return pos_embed.unsqueeze(0)


class MAE(nn.Module):
    def __init__(self, cfg: MAEVariantConfig, in_chans: int = 3):
        super().__init__()
        self.cfg = cfg
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=in_chans,
            embed_dim=cfg.encoder_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = cfg.patch_size * cfg.patch_size * in_chans

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_dim))
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.encoder_dim), requires_grad=False)

        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(dim=cfg.encoder_dim, num_heads=cfg.encoder_heads) for _ in range(cfg.encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(cfg.encoder_dim)

        self.decoder_embed = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.decoder_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(dim=cfg.decoder_dim, num_heads=cfg.decoder_heads) for _ in range(cfg.decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        with torch.no_grad():
            self.pos_embed_enc.copy_(
                build_2d_sincos_position_embedding(self.patch_embed.grid_size, self.cfg.encoder_dim)
            )
            self.pos_embed_dec.copy_(
                build_2d_sincos_position_embedding(self.patch_embed.grid_size, self.cfg.decoder_dim)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.cfg.patch_size
        n = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, n, p, n, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(shape=(imgs.shape[0], n * n, p * p * self.in_chans))

    def random_masking(self, x, mask_ratio: float):
        n, l, d = x.shape  # noqa: E741
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed_enc[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.cfg.mask_ratio)

        cls_token = self.cls_token + self.pos_embed_enc[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.pos_embed_dec

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return self.decoder_pred(x[:, 1:, :])

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, target

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss, target = self.forward_loss(imgs, pred, mask)
        return loss, pred, target, mask


def build_model_and_loader(variant: str, data_root: str = "./data"):
    cfg = VARIANT_CONFIG[variant]
    model = MAE(cfg)

    if variant == "cifar10":
        loader = make_cifar10_dataloader(root=data_root, batch_size=cfg.batch_size)
    elif variant == "imagenet":
        loader = make_imagenet_dataloader(root=data_root, batch_size=cfg.batch_size)
    else:
        loader = make_coco_dataloader(root=data_root, batch_size=cfg.batch_size)

    return cfg, model, loader


def train_dummy(variant: str = "cifar10", steps: int = 1, data_root: str = "./data"):
    cfg, model, loader = build_model_and_loader(variant, data_root=data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)

    use_amp = device.type == "cuda"

    if use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    model.train()
    for step, imgs in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, pred, target, mask = model(imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, pred, target, mask = model(imgs)
            loss.backward()
            optimizer.step()

        masked_pct = float(mask.mean() * 100.0)
        print(
            f"variant={variant} step={step} loss={float(loss):.6f} "
            f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)} masked={masked_pct:.1f}%"
        )

        if step >= steps:
            break


def main():
    parser = argparse.ArgumentParser(description="MAE debug trainer with CIFAR-10 and ImageNet presets")
    parser.add_argument("--variant", type=str, default="cifar10", choices=["cifar10", "imagenet", "coco"])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for CIFAR-10 or ImageNet. CIFAR-10 downloads here. ImageNet must be prepared locally.",
    )
    args = parser.parse_args()

    if args.variant not in VARIANT_CONFIG:
        supported = ", ".join(sorted(VARIANT_CONFIG))
        raise ValueError(f"Unknown variant '{args.variant}'. Supported: {supported}")

    train_dummy(variant=args.variant, steps=args.steps, data_root=args.data_root)


if __name__ == "__main__":
    main()
