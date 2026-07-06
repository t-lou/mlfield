import torch
from torch import nn

from .mlp import MLP


class TransformerBlock(nn.Module):
    """
    Vision Transformer (ViT) encoder/decoder block with self-attention and MLP.

    Standard transformer architecture with pre-normalization:
    Implements: x -> LayerNorm -> MultiheadAttention -> x + attn_out
               x -> LayerNorm -> MLP -> x + mlp_out

    This design improves training stability and convergence compared to post-normalization.

    Improvement: Consider adding:
        - Flash attention for faster computation
        - Sparse attention patterns for longer sequences
        - Rotary position embeddings for better length extrapolation
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            mlp_ratio: Expansion ratio for MLP hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of same shape
        """
        # Pre-normalization for stability
        attn_in = self.norm1(x)
        # Self-attention
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        # Residual connection, in order to preserve information and improve gradient flow
        x = x + attn_out
        # Pre-normalization for MLP, again for stability
        x = x + self.mlp(self.norm2(x))
        return x


def _smoke_test():
    """Smoke test for the TransformerBlock module."""
    batch_size = 2
    seq_len = 4
    dim = 8
    num_heads = 2
    mlp_ratio = 2.0
    dropout = 0.1

    # Create a random input tensor simulating a batch of sequences
    x = torch.randn(batch_size, seq_len, dim)

    # Initialize TransformerBlock
    transformer_block = TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)

    # Forward pass
    output = transformer_block(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"


if __name__ == "__main__":
    _smoke_test()
