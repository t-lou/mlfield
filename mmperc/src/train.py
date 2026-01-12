import torch.nn as nn
import torch.optim as optim
from backbone.tiny_bev_backbone import TinyBEVBackbone
from device_utils import get_best_device
from fusion.futr_fusion import FuTrFusionBlock
from tasks.drivable_head import DrivableAreaHead
from torch.utils.data import DataLoader
from voxelizer.pointpillars_lite import TorchPillarVoxelizer


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()

    for batch in dataloader:
        points = batch["points"].to(device)  # (N, 5)
        cam_tokens = batch["cam_tokens"].to(device)  # (B, N_cam, C)
        target_mask = batch["drivable_mask"].to(device)  # (B, 1, H, W)

        optimizer.zero_grad()

        pred_mask = model(points, cam_tokens)

        loss = criterion(pred_mask, target_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


class DrivableModel(nn.Module):
    """
    Full pipeline:
    points → voxelizer → PFN (TODO) → BEV backbone → fusion → drivable head
    """

    def __init__(self):
        super().__init__()
        self.voxelizer = TorchPillarVoxelizer()
        # TODO: add PFN here
        self.backbone = TinyBEVBackbone(in_channels=64)
        self.fusion = FuTrFusionBlock(bev_channels=128)
        self.head = DrivableAreaHead(in_channels=128)

    def forward(self, points, cam_tokens):
        vox = self.voxelizer(points)["pillars"]  # placeholder
        # TODO: PFN → bev_features
        bev = self.backbone(vox)
        fused = self.fusion(bev, cam_tokens)
        mask = self.head(fused)
        return mask


def main():
    device = get_best_device()

    # TODO: replace with your dataset
    dataset = ...
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = DrivableModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}: loss={loss:.4f}")


if __name__ == "__main__":
    main()
