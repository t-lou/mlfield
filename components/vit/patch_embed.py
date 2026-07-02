import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings using a convolutional projection.

    Divides input image into non-overlapping patches and linearly embeds them.
    This is the standard approach in Vision Transformers (ViT).

    Implementation: Patches are extracted using Conv2d with stride=patch_size,
    then flattened and projected to embed_dim.

    Example:
        For 224x224 image with patch_size=16 and embed_dim=768:
        - Output shape: (batch, 196, 768) where 196 = (224/16)^2

    Improvement: Consider adding:
        - Learnable patch projection instead of just convolution
        - Overlapping patches for smoother transitions
        - Adaptive patch sizing based on image content
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        """
        Args:
            img_size: Input image size (assumes square images)
            patch_size: Size of each patch
            in_chans: Number of input channels (3 for RGB)
            embed_dim: Output embedding dimension

        Raises:
            ValueError: If img_size is not divisible by patch_size
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches and project to embedding space.

        Args:
            x: Input images of shape (batch, in_chans, img_size, img_size)

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        # Use convolution to extract patches and project to embedding dimension
        x = self.proj(x)  # (batch, embed_dim, grid_size, grid_size)
        # Flatten spatial dimensions and transpose to (batch, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x
