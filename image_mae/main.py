import argparse
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DEFAULT_KAGGLE_DATASETS = {
    # This is not the official ILSVRC release. It is a Kaggle-hosted ImageNet-style layout.
    "imagenet": "ifigotin/imagenetmini-1000",
    "coco": "awsaf49/coco-2017-dataset",
}


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
        batch_size=512,
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


def _find_dataset_root_by_markers(base_dir: Path, required_dirs: tuple[str, ...]) -> Path | None:
    # Prefer shallow paths first because Kaggle archives often unpack into one top-level directory.
    candidates = [base_dir]
    candidates.extend(p for p in base_dir.rglob("*") if p.is_dir())

    for path in candidates:
        if all((path / marker).exists() for marker in required_dirs):
            return path
    return None


def _download_kaggle_dataset(dataset_slug: str, dest_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Kaggle support requires the 'kaggle' package. Install with: pip install kaggle") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local auth setup
        raise RuntimeError(
            "Kaggle authentication failed. Place credentials at ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)

    # Some Kaggle datasets contain zip files inside the first extracted directory.
    for zip_path in dest_dir.rglob("*.zip"):
        extract_dir = zip_path.parent
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def resolve_variant_data_root(
    variant: str,
    data_root: str,
    use_kaggle: bool,
    kaggle_imagenet_dataset: str,
    kaggle_coco_dataset: str,
) -> str:
    root = Path(data_root)
    if variant == "cifar10":
        return str(root)

    if variant == "imagenet":
        prepared_root = _find_dataset_root_by_markers(root, ("train", "val"))
        if prepared_root is not None:
            return str(prepared_root)

        if not use_kaggle:
            raise FileNotFoundError(
                f"Could not find ImageNet layout under {root}. Expected directories: train/ and val/."
            )

        cache_dir = root / "kaggle" / "imagenet"
        prepared_root = _find_dataset_root_by_markers(cache_dir, ("train", "val"))
        if prepared_root is None:
            _download_kaggle_dataset(kaggle_imagenet_dataset, cache_dir)
            prepared_root = _find_dataset_root_by_markers(cache_dir, ("train", "val"))

        if prepared_root is None:
            raise FileNotFoundError("Downloaded ImageNet dataset does not expose train/ and val/ directories.")
        return str(prepared_root)

    # coco
    prepared_root = _find_dataset_root_by_markers(root, ("train2017", "val2017", "annotations"))
    if prepared_root is not None:
        return str(prepared_root)

    if not use_kaggle:
        raise FileNotFoundError(
            f"Could not find COCO layout under {root}. Expected train2017/, val2017/, annotations/."
        )

    cache_dir = root / "kaggle" / "coco"
    prepared_root = _find_dataset_root_by_markers(cache_dir, ("train2017", "val2017", "annotations"))
    if prepared_root is None:
        _download_kaggle_dataset(kaggle_coco_dataset, cache_dir)
        prepared_root = _find_dataset_root_by_markers(cache_dir, ("train2017", "val2017", "annotations"))

    if prepared_root is None:
        raise FileNotFoundError("Downloaded COCO dataset does not expose train2017/, val2017/, annotations/.")
    return str(prepared_root)


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
    def __init__(self, variant, in_chans: int = 3):
        super().__init__()

        self.cfg = VARIANT_CONFIG[variant]
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
            in_chans=in_chans,
            embed_dim=self.cfg.encoder_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = self.cfg.patch_size * self.cfg.patch_size * in_chans

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.encoder_dim))
        self.pos_embed_enc = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.cfg.encoder_dim), requires_grad=False
        )

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=self.cfg.encoder_dim, num_heads=self.cfg.encoder_heads)
                for _ in range(self.cfg.encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(self.cfg.encoder_dim)

        self.decoder_embed = nn.Linear(self.cfg.encoder_dim, self.cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.cfg.decoder_dim))
        self.pos_embed_dec = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.cfg.decoder_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=self.cfg.decoder_dim, num_heads=self.cfg.decoder_heads)
                for _ in range(self.cfg.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.cfg.decoder_dim)
        self.decoder_pred = nn.Linear(self.cfg.decoder_dim, self.patch_dim)

        self._init_weights()  # auto load?

        self.path_final_ckpt = Path(f"mae_checkpoints/{variant}/final.pth")
        if not self.path_final_ckpt.parent.exists():
            self.path_final_ckpt.parent.mkdir(parents=True, exist_ok=True)

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

    def unpatchify(self, x):
        p = self.cfg.patch_size
        n = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], n, n, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], self.in_chans, n * p, n * p))

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

    def save_checkpoint(self, path=None):
        path_ckpt = self.path_final_ckpt if path is None else path
        torch.save(self.state_dict(), path_ckpt)

    def load_checkpoint(self, path=None, device=None):
        path_ckpt = self.path_final_ckpt if path is None else path
        if not path_ckpt.exists():
            print(f"{path_ckpt} not found, cannot load")
            return
        state_dict = torch.load(path_ckpt, map_location=device)
        self.load_state_dict(state_dict)


def mae_visualize(model, imgs, save_path):
    model.eval()
    with torch.no_grad():
        # ---- 1. Encode ----
        latent, mask, ids_restore = model.forward_encoder(imgs)

        # ---- 2. Decode ----
        pred = model.forward_decoder(latent, ids_restore)  # (B, N, patch_dim)

        # ---- 3. Unpatchify ----
        rec_imgs = model.unpatchify(pred)  # (B, 3, H, W)

        # ---- 4. Build masked image ----
        B, C, H, W = imgs.shape
        patch = model.cfg.patch_size

        # mask: (B, N), 1 = masked
        mask = mask.unsqueeze(-1).repeat(1, 1, patch * patch * C)
        mask = model.unpatchify(mask)  # (B, 3, H, W)
        masked_imgs = imgs * (1 - mask)  # zero out masked patches

    # ---- 5. Plot first 6 ----
    num_show = min(6, imgs.shape[0])
    fig, axes = plt.subplots(3, num_show, figsize=(3 * num_show, 9))

    for i in range(num_show):
        # original
        axes[0, i].imshow(imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # masked
        axes[1, i].imshow(masked_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[1, i].set_title("Masked")
        axes[1, i].axis("off")

        # reconstructed
        axes[2, i].imshow(rec_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[2, i].set_title("Reconstructed")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def build_model_and_loader(
    variant: str,
    data_root: str = "./data",
    use_kaggle: bool = True,
    kaggle_imagenet_dataset: str = DEFAULT_KAGGLE_DATASETS["imagenet"],
    kaggle_coco_dataset: str = DEFAULT_KAGGLE_DATASETS["coco"],
):
    model = MAE(variant)
    model.load_checkpoint()

    cfg = VARIANT_CONFIG[variant]
    resolved_root = resolve_variant_data_root(
        variant=variant,
        data_root=data_root,
        use_kaggle=use_kaggle,
        kaggle_imagenet_dataset=kaggle_imagenet_dataset,
        kaggle_coco_dataset=kaggle_coco_dataset,
    )

    if variant == "cifar10":
        loader = make_cifar10_dataloader(root=resolved_root, batch_size=cfg.batch_size)
    elif variant == "imagenet":
        loader = make_imagenet_dataloader(root=resolved_root, batch_size=cfg.batch_size)
    else:
        loader = make_coco_dataloader(root=resolved_root, batch_size=cfg.batch_size)

    return cfg, model, loader


def train(
    variant: str = "cifar10",
    steps: int = -1,
    data_root: str = "./data",
    epoch: int = 0,
    use_kaggle: bool = True,
    kaggle_imagenet_dataset: str = DEFAULT_KAGGLE_DATASETS["imagenet"],
    kaggle_coco_dataset: str = DEFAULT_KAGGLE_DATASETS["coco"],
):
    cfg, model, loader = build_model_and_loader(
        variant,
        data_root=data_root,
        use_kaggle=use_kaggle,
        kaggle_imagenet_dataset=kaggle_imagenet_dataset,
        kaggle_coco_dataset=kaggle_coco_dataset,
    )
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

        if steps >= 0 and step >= steps:
            break
        elif (step + 1) % 10_000 == 0:
            print(
                f"variant={variant} step={step} loss={float(loss):.6f} "
                f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)}"
            )

            model.save_checkpoint()

    print(
        f"variant={variant} step={step} loss={float(loss):.6f} "
        f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)}"
    )
    model.save_checkpoint()

    path_vis = Path(f"mae_visualizations/{variant}")
    path_vis.mkdir(parents=True, exist_ok=True)
    mae_visualize(model, imgs, save_path=(path_vis / f"step_{epoch}.png"))


def main():
    parser = argparse.ArgumentParser(description="MAE debug trainer with CIFAR-10 and ImageNet presets")
    parser.add_argument("--variant", type=str, default="cifar10", choices=["cifar10", "imagenet", "coco"])
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for CIFAR-10 or ImageNet. CIFAR-10 downloads here. ImageNet must be prepared locally.",
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="Disable Kaggle auto-download for ImageNet/COCO. Requires datasets to already exist in --data-root.",
    )
    parser.add_argument(
        "--kaggle-imagenet-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASETS["imagenet"],
        help="Kaggle dataset slug for ImageNet-style data (owner/dataset).",
    )
    parser.add_argument(
        "--kaggle-coco-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASETS["coco"],
        help="Kaggle dataset slug for COCO-style data (owner/dataset).",
    )
    args = parser.parse_args()

    if args.variant not in VARIANT_CONFIG:
        supported = ", ".join(sorted(VARIANT_CONFIG))
        raise ValueError(f"Unknown variant '{args.variant}'. Supported: {supported}")

    for epoch in range(args.epochs):
        train(
            variant=args.variant,
            steps=args.steps,
            data_root=args.data_root,
            epoch=epoch,
            use_kaggle=not args.no_kaggle,
            kaggle_imagenet_dataset=args.kaggle_imagenet_dataset,
            kaggle_coco_dataset=args.kaggle_coco_dataset,
        )


if __name__ == "__main__":
    main()
