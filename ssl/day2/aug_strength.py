import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

# Define transforms
weak = T.Compose([T.RandomResizedCrop(32, scale=(0.8, 1.0)), T.RandomHorizontalFlip(), T.ToTensor()])

strong = T.Compose(
    [
        T.RandomResizedCrop(32, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor(),
    ]
)

dataset = CIFAR10(root="../data", train=True, download=True)
img, _ = dataset[0]

x_w = weak(img)
x_s = strong(img)

pixel_dist = torch.norm(x_w - x_s).item()
print("Pixel distance:", pixel_dist)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

with torch.no_grad():
    f_w = model(x_w.unsqueeze(0))
    f_s = model(x_s.unsqueeze(0))

feat_dist = F.cosine_similarity(f_w, f_s).item()
print("Feature cosine similarity:", feat_dist)
