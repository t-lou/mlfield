import common.params as params
import torch.nn.functional as F


def rescale_image(x, scale_factor: float = params.IMAGE_SCALE):
    """
    Rescale input image tensor by a given scale factor.

    Args:
        x: Input image tensor of shape (B, C, H, W)

    Returns:
        Rescaled image tensor of shape (B, C, H_new, W_new)
    """
    if scale_factor == 1.0:
        return x

    B, C, H, W = x.shape
    H_new = int(H * scale_factor)
    W_new = int(W * scale_factor)

    x_rescaled = F.interpolate(x, size=(H_new, W_new), mode='bilinear', align_corners=False)
    return x_rescaled