from dataclasses import dataclass

import torch


@dataclass
class TorchPointCloud:
    points: torch.Tensor  # shape (N, 5): [x, y, z, intensity, timestamp]

    @classmethod
    def from_xyzit(cls, arr, device="cpu"):
        """
        arr: numpy or torch array of shape (N, 5)
        """
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr, dtype=torch.float32)
        return cls(points=arr.to(device))
