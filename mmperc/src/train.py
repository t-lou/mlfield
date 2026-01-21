import torch.optim as optim
from common.device import get_best_device
from datasets.a2d2_dataset import A2D2Dataset
from model.simple_model import SimpleModel
from pipeline.train_bbox2d import train_model
from torch.utils.data import DataLoader


def main():
    device = get_best_device()

    path_dataset = "/workspace/mmperc/data/a2d2"
    dataset = A2D2Dataset(root=path_dataset)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, dataloader, optimizer, device, num_epochs=5, ckpt_dir="checkpoints")


if __name__ == "__main__":
    main()
