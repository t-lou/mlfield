"""Multi-Layer Perceptron implementations for Vision Transformers.

Provides flexible MLP variants used in transformer blocks with
support for different activation functions and normalization strategies.
"""

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable activation.

    Standard MLP used in transformer blocks with improved defaults:
    - GELU activation (better than ReLU6 for vision tasks)
    - Flexible architecture supporting different activation choices
    - Efficient implementation with minimal overhead

    Architecture:
        Input -> Linear(dim -> hidden_dim) -> Activation -> Dropout ->
        Linear(hidden_dim -> dim) -> Dropout -> Output

    Design choices:
    - GELU over ReLU6: Better gradient flow and smoother outputs
    - Two separate dropout layers: One after activation, one after projection
    - No normalization: Let LayerNorm in parent block handle it
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, act_layer: str = "GELU") -> None:
        """Initialize MLP.

        Args:
            dim: Input and output feature dimension
            mlp_ratio: Expansion ratio for hidden layer (hidden_dim = dim * mlp_ratio)
            dropout: Dropout probability
            act_layer: Activation function name ("GELU", "SiLU", "Mish", "ReLU", "ReLU6")
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = self._get_activation(act_layer)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

        self.mlp_ratio = mlp_ratio
        self.act_layer = act_layer

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation name

        Returns:
            Activation module

        Raises:
            ValueError: If activation name is not supported
        """
        activations = {
            "GELU": nn.GELU(),
            "gelu": nn.GELU(),
            "SiLU": nn.SiLU(),
            "silu": nn.SiLU(),
            "Mish": nn.Mish(),
            "mish": nn.Mish(),
            "ReLU": nn.ReLU(),
            "relu": nn.ReLU(),
            "ReLU6": nn.ReLU6(),
            "relu6": nn.ReLU6(),
        }

        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}. Supported: {list(activations.keys())}")

        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

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
    """Smoke test for MLP implementations."""
    batch_size = 2
    seq_len = 4
    dim = 8
    mlp_ratio = 2.0
    dropout = 0.1

    print("Testing MLP with different activations...")

    for act in ["GELU", "SiLU", "ReLU"]:
        mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout, act_layer=act)
        input_tensor = torch.randn(batch_size, seq_len, dim)
        output_tensor = mlp(input_tensor)

        assert output_tensor.shape == input_tensor.shape, f"Output shape mismatch for {act}"
        print(f"  ✓ {act}: {input_tensor.shape} -> {output_tensor.shape}")

    print("✓ All MLP tests passed!")


if __name__ == "__main__":
    _smoke_test()
