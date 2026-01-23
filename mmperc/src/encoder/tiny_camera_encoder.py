import common.params as params
import torch.nn as nn
from torch import Tensor


class TinyCameraEncoder(nn.Module):
    """
    Minimal camera encoder.

    Converts an RGB image:
        (B, 3, H, W)

    Into a sequence of camera tokens:
        (B, N_cam, C)
    and a feature map:
        (B, C, H', W')

    where:
        - N_cam = H' * W' after downsampling
        - C = out_channels (default 128)
    """

    def __init__(self) -> None:
        super().__init__()

        # A small CNN backbone that downsamples the image 3 times.
        # Each stride=2 halves spatial resolution.
        # Final output: (B, out_channels, H', W')
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, params.BEV_CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(params.BEV_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # LayerNorm applied after flattening into tokens
        self.norm = nn.LayerNorm(params.BEV_CHANNELS)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: RGB image tensor of shape (B, 3, H, W)

        Returns:
            tokens: (B, N_cam, out_channels)
                    where N_cam = H' * W' after downsampling
            feat:   (B, out_channels, H', W')
        """
        # Extract CNN features
        feat: Tensor = self.conv(x)  # (B, C, H', W')
        B, C, H2, W2 = feat.shape

        # Flatten spatial dimensions → sequence of tokens
        # (B, C, H', W') → (B, H'*W', C)
        tokens: Tensor = feat.flatten(2).transpose(1, 2)

        # Normalize token embeddings
        tokens = self.norm(tokens)

        return tokens, feat
