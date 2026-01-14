import torch.nn as nn
import torch.optim as optim
from backbone.tiny_bev_backbone import TinyBEVBackbone
from common.device import get_best_device
from datasets.a2d2_dataset import A2D2Dataset
from encoder_3d.point_pillars_bev import PointPillarsBEV
from fusion.futr_fusion import FuTrFusionBlock
from head.drivable_head import DrivableAreaHead
from torch.utils.data import DataLoader
from voxel_encoder.simple_pfn import SimplePFN
from voxelizer.pointpillars_lite import TorchPillarVoxelizer


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()

    for batch in dataloader:
        points = batch["points"].to(device)  # (N, 5)
        cam_tokens = batch["cam_tokens"].to(device)  # (B, N_cam, C)
        target_mask = batch["semantics"].to(device)  # (B, 1, H, W)

        optimizer.zero_grad()

        pred_mask = model(points, cam_tokens)

        loss = criterion(pred_mask, target_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


class FullModel(nn.Module):
    """
    Full pipeline:
    points → voxelizer → PFN (TODO) → BEV backbone → fusion → drivable head
    """

    def __init__(self):
        super().__init__()

        # Lidar encoder
        voxelizer = TorchPillarVoxelizer()
        pfn = SimplePFN(in_channels=5, out_channels=64)
        backbone = TinyBEVBackbone(in_channels=64, mid_channels=64, out_channels=128)
        bev_w = int((voxelizer.x_max - voxelizer.x_min) / voxelizer.vx)
        bev_h = int((voxelizer.y_max - voxelizer.y_min) / voxelizer.vy)
        self.lidar_encoder = PointPillarsBEV(voxelizer, pfn, backbone, bev_h, bev_w)

        # TODO: add PFN here
        self.fusion = FuTrFusionBlock(bev_channels=128)
        self.head = DrivableAreaHead(in_channels=128)

    def forward(self, points, cam_tokens):
        vox = self.lidar_encoder(points)["pillars"]  # placeholder
        # TODO: PFN → bev_features
        bev = self.backbone(vox)
        fused = self.fusion(bev, cam_tokens)
        mask = self.head(fused)
        return mask


def main():
    device = get_best_device()

    # TODO: replace with your dataset
    path_dataset = "../data/a2d2-preview/camera_lidar_semantic_bboxes"
    dataset = A2D2Dataset(root=path_dataset, use_cam_tokens=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = FullModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}: loss={loss:.4f}")


if __name__ == "__main__":
    main()
