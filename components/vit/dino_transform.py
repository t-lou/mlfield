import torch
from torchvision import transforms

from components.utils.logger import logger
from components.vit.dino_defs import DINOConfig


def _blur(res: int, p: float):
    """Gaussian blur with sigma in [0.1, 2.0] (DINO recipe), applied with probability p."""
    kernel = 2 * max(1, int(0.05 * res)) + 1
    return transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel, sigma=(0.1, 2.0))], p=p)


def _plain_transform(res: int, scale: tuple[float, float]):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(res, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    )


def _dino_transform(res: int, scale: tuple[float, float], blur_p: float, solarize_p: float):
    ops = [
        transforms.RandomResizedCrop(res, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        _blur(res, blur_p),
    ]
    if solarize_p > 0:
        ops.append(transforms.RandomSolarize(threshold=128, p=solarize_p))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class DINOTransform:
    def __init__(self, config: DINOConfig):
        logger.info(
            f"Initialize DINO transform, {config.num_teachers} teachers of size {config.teacher_base_res} "
            f"and {config.num_students} students of size {config.student_base_res}."
        )

        self.num_teachers = config.num_teachers
        self.num_students = config.num_students

        if config.dino_augmentations:
            # DINO recipe: the first global crop is always blurred, further global crops are rarely
            # blurred but sometimes solarized; local crops are blurred half of the time.
            res, local_res = config.teacher_base_res, config.student_base_res
            self.global_transforms = [
                _dino_transform(res, (0.4, 1.0), blur_p=1.0, solarize_p=0.0),
                _dino_transform(res, (0.4, 1.0), blur_p=0.1, solarize_p=0.2),
            ]
            self.local_transform = _dino_transform(local_res, (0.05, 0.4), blur_p=0.5, solarize_p=0.0)
        else:
            plain_global = _plain_transform(config.teacher_base_res, (0.4, 1.0))
            self.global_transforms = [plain_global, plain_global]
            self.local_transform = _plain_transform(config.student_base_res, (0.05, 0.4))

        # kept for backward compatibility (the first global transform)
        self.global_transform = self.global_transforms[0]

    def __call__(self, img):
        crops = []
        for i in range(self.num_teachers):
            crops.append(self.global_transforms[min(i, len(self.global_transforms) - 1)](img))
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
