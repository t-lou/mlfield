import torch.nn as nn
import torch.optim as optim
from common.device import get_best_device
from datasets.a2d2_dataset import A2D2Dataset
from model.simple_model import SimpleModel
from torch.utils.data import DataLoader


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


def main():
    device = get_best_device()

    # TODO: replace with your dataset
    path_dataset = "../data/a2d2-preview/camera_lidar_semantic_bboxes"
    dataset = A2D2Dataset(root=path_dataset, use_cam_tokens=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}: loss={loss:.4f}")


if __name__ == "__main__":
    main()
