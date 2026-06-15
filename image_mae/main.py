import torch
from torch.utils.data import DataLoader, Dataset


class DummyImageDataset(Dataset):
    def __init__(self, size=1000, image_size=32):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        return img


def make_dummy_dataloader(batch_size=32, image_size=32):
    ds = DummyImageDataset(image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


class DummyPatchifier:
    def __init__(self, patch_size=4):
        self.patch_size = patch_size

    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        N = (H // self.patch_size) * (W // self.patch_size)
        return torch.randn(B, N, self.patch_size * self.patch_size * C)

    def unpatchify(self, patches, H, W):
        B = patches.shape[0]
        return torch.randn(B, 3, H, W)


class DummyMasker:
    def __init__(self, mask_ratio=0.75):
        self.mask_ratio = mask_ratio

    def __call__(self, patches):
        B, N, D = patches.shape
        num_keep = int(N * (1 - self.mask_ratio))
        visible = patches[:, :num_keep]
        mask_idx = torch.arange(num_keep, N)
        return visible, mask_idx


class DummyEncoder(torch.nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.proj = torch.nn.Linear(48, embed_dim)  # dummy

    def forward(self, visible_patches):
        return self.proj(visible_patches)


class DummyDecoder(torch.nn.Module):
    def __init__(self, embed_dim=128, patch_dim=48):
        super().__init__()
        self.fc = torch.nn.Linear(embed_dim, patch_dim)

    def forward(self, encoded, mask_idx, total_patches):
        B = encoded.shape[0]
        recon = torch.randn(B, total_patches, encoded.shape[-1])
        return self.fc(recon)


class DummyLoss(torch.nn.Module):
    def forward(self, pred, target):
        return ((pred - target) ** 2).mean()


class DummyMAE(torch.nn.Module):
    def __init__(self, patch_size=4, embed_dim=128):
        super().__init__()
        self.patchifier = DummyPatchifier(patch_size)
        self.masker = DummyMasker()
        self.encoder = DummyEncoder(embed_dim)
        self.decoder = DummyDecoder(embed_dim)

    def forward(self, imgs):
        B, C, H, W = imgs.shape

        patches = self.patchifier.patchify(imgs)
        visible, mask_idx = self.masker(patches)

        encoded = self.encoder(visible)
        recon = self.decoder(encoded, mask_idx, patches.shape[1])

        return recon, patches


def train_dummy():
    loader = make_dummy_dataloader(batch_size=8, image_size=32)
    model = DummyMAE()
    loss_fn = DummyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for imgs in loader:
        recon, target = model(imgs)
        loss = loss_fn(recon, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print("loss:", float(loss))
        break  # one batch for debugging


if __name__ == "main":
    train_dummy()
