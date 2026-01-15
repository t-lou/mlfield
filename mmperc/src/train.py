import torch.optim as optim
from common.device import get_best_device
from datasets.a2d2_dataset import A2D2Dataset
from model.simple_model import SimpleModel
from pipeline.train_bbox2d import train_one_epoch
from torch.utils.data import DataLoader


def main():
    device = get_best_device()

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
