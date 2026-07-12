import torch
from torchvision import transforms

from components.utils.logger import logger
from components.vit.dino_defs import DINOConfig


class DINOTransform:
    def __init__(self, config: DINOConfig):

        logger.info(
            f"Initialize DINO transform, {config.num_teachers} teachers of size {config.teacher_base_res} "
            f"and {config.num_students} students of size {config.student_base_res}."
        )

        self.num_teachers = config.num_teachers
        self.num_students = config.num_students

        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(config.teacher_base_res, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(config.student_base_res, scale=(0.05, 0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, img):
        crops = []
        for _ in range(self.num_teachers):
            crops.append(self.global_transform(img))
        for _ in range(self.num_students):
            crops.append(self.local_transform(img))
        return crops


def dino_collate_fn(batch):
    """Collate a batch of multi-crop samples into a list of crop tensors."""
    if not batch:
        return []
    num_crops = len(batch[0])
    if any(len(sample) != num_crops for sample in batch):
        raise ValueError("Inconsistent number of crops in batch")
    return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(num_crops)]
