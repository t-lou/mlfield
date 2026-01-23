import common.params as params
import torch.nn.functional as F


def rescale_image(x, scale_factor: float = params.IMAGE_SCALE, is_label: bool = False):
    """
    Safely rescale either an RGB image or a semantic label map.

    Args:
        x: Tensor of shape (B, C, H, W) for images
           or (B, H, W) for labels.
        scale_factor: float scaling factor.
        is_label: if True, uses nearest-neighbor interpolation
                  and preserves integer class IDs.

    Returns:
        Rescaled tensor with correct interpolation mode.
    """

    if scale_factor == 1.0:
        return x

    # Handle labels: (B, H, W) → (B, 1, H, W)
    if is_label:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.float()  # interpolate requires float
        mode = "nearest"
    else:
        # Images must be 4D
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor of shape (B, C, H, W), got {x.shape}")
        mode = "bilinear"

    B, C, H, W = x.shape
    H_new = int(H * scale_factor)
    W_new = int(W * scale_factor)

    x_rescaled = F.interpolate(
        x,
        size=(H_new, W_new),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Convert labels back to (B, H, W) and long dtype
    if is_label:
        return x_rescaled.squeeze(1).long()

    return x_rescaled
