import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

transform = T.Compose(
    [T.RandomResizedCrop(32), T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.ToTensor()]
)

dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, labels = next(iter(loader))

grid = images.permute(0, 2, 3, 1)
plt.figure(figsize=(10, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(grid[i])
    plt.axis("off")

# plt.show()
plt.savefig("image.png")
