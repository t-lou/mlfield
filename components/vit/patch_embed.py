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

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        """
        Args:
            patch_size: Size of each patch
            in_chans: Number of input channels (3 for RGB)
            embed_dim: Output embedding dimension

        Raises:
            ValueError: If img_size is not divisible by patch_size
        """
        super().__init__()

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


def _smoke_test():
    """Smoke test for the PatchEmbed module."""
    batch_size = 2
    in_chans = 3
    img_size = 224
    patch_size = 16
    embed_dim = 768

    # Create a random input tensor simulating images
    x = torch.randn(batch_size, in_chans, img_size, img_size)

    # Initialize PatchEmbed
    patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

    # Forward pass
    _ = patch_embed(x)


if __name__ == "__main__":
    _smoke_test()
