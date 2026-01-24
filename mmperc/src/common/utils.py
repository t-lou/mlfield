import common.params as params
import torch.nn.functional as F


def rescale_image(x, scale_factor: float = params.IMAGE_SCALE, is_label: bool = False):
    if scale_factor == 1.0:
        return x

    # -----------------------------
    # Normalize shapes
    # -----------------------------
    if is_label:
        # Labels: (H, W) or (B, H, W)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        else:
            raise ValueError(f"Label tensor must be 2D or 3D, got {x.shape}")

        x = x.float()
        mode = "nearest"

    else:
        # Images: (H, W, C) → (1, C, H, W)
        if x.ndim == 3 and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1).unsqueeze(0)

        # Images: (C, H, W) → (1, C, H, W)
        elif x.ndim == 3:
            x = x.unsqueeze(0)

        # Images: (B, C, H, W)
        elif x.ndim != 4:
            raise ValueError(f"Image tensor must be 3D or 4D, got {x.shape}")

        mode = "bilinear"

    # -----------------------------
    # Compute new size
    # -----------------------------
    H, W = x.shape[-2], x.shape[-1]
    H_new = int(H * scale_factor)
    W_new = int(W * scale_factor)

    # -----------------------------
    # Interpolate
    # -----------------------------
    x_rescaled = F.interpolate(
        x,
        size=(H_new, W_new),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # -----------------------------
    # Restore shapes
    # -----------------------------
    if is_label:
        x_rescaled = x_rescaled.squeeze(1).long()
        return x_rescaled[0] if x_rescaled.shape[0] == 1 else x_rescaled

    else:
        # (1, C, H, W) → (C, H, W)
        if x_rescaled.shape[0] == 1:
            return x_rescaled[0]
        return x_rescaled
