import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) with GeLU-like activation.

    Standard MLP used in transformer blocks: Linear -> ReLU6 -> Dropout -> Linear -> Dropout
    Expands features by mlp_ratio and then projects back to original dimension.

    Architecture:
        Input -> FC(dim -> hidden_dim) -> ReLU6 -> Dropout -> FC(hidden_dim -> dim) -> Dropout

    Improvement: Consider adding:
        - GELU activation instead of ReLU6 for better performance
        - Optional layer normalization
        - Depthwise separable convolutions for efficiency
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        Args:
            dim: Input and output feature dimension
            mlp_ratio: Expansion ratio for hidden layer (hidden_dim = dim * mlp_ratio)
            dropout: Dropout probability
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, dim)

        Returns:
            Output tensor of same shape as input
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _smoke_test():
    """Smoke test for the MLP module."""
    batch_size = 2
    seq_len = 4
    dim = 8
    mlp_ratio = 2.0
    dropout = 0.1

    mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
    input_tensor = torch.randn(batch_size, seq_len, dim)
    output_tensor = mlp(input_tensor)

    assert output_tensor.shape == input_tensor.shape, "Output shape mismatch"


if __name__ == "__main__":
    _smoke_test()
