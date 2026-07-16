import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Stochastic depth: randomly drops entire residual paths.
    Equivalent to timm.models.layers.DropPath.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # shape: (batch, 1, 1) so each sample drops independently
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()  # binarize
        return x / keep_prob * random_tensor


class VitBlock(nn.Module):
    """Vision Transformer Block.

    Key differences from standard transformer blocks:
    - Pre-LayerNorm: LayerNorm is applied before attention and MLP.
    - Fused QKV projection: A single linear layer computes Q, K, V.
    - Stochastic depth: DropPath is applied to both attention and MLP outputs.
    - GELU activation in MLP.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        qkv_bias=False,  # DINO uses qkv_bias = False
    ):
        """Initialize a Vision Transformer block.

        Args:
            dim: Embedding dimension (D)
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
            drop_path: Stochastic depth rate
            qkv_bias: Whether to include bias in QKV projection
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # standard ViT scaling

        # --- LayerNorm before attention ---
        self.norm1 = nn.LayerNorm(dim)

        # qkv projection (DINO uses bias=False)
        # produces [q, k, v] in one fused linear layer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # output projection after attention
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # dropout on attention output

        # stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- MLP ---
        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),  # dropout inside MLP
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),  # dropout on MLP output
        )

        # attention dropout
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, padding_mask: torch.Tensor | None = None):
        B, N, D = x.shape

        # --- Attention ---
        x_norm = self.norm1(x)

        # fused qkv projection
        qkv = self.qkv(x_norm)  # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape into heads
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if padding_mask is not None:
            padding_mask = padding_mask.to(torch.bool)
            # mask out padded key positions for every query
            key_mask = padding_mask[:, None, None, :]
            attn = attn.masked_fill(key_mask, torch.finfo(attn.dtype).min)
            # set padded query rows to zero so they do not contribute after residual
            query_mask = padding_mask[:, None, :, None]
            attn = attn.masked_fill(query_mask, 0.0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # attention dropout

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)

        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        out = self.proj(out)
        out = self.proj_drop(out)

        # residual + stochastic depth
        x = x + self.drop_path(out)

        # --- MLP ---
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)

        if padding_mask is not None:
            mlp_out = mlp_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # residual + stochastic depth
        x = x + self.drop_path(mlp_out)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x


def _smoke_test():
    """Run a simple smoke test to verify the VitBlock implementation."""
    batch_size = 2
    num_patches = 16
    embed_dim = 64
    num_heads = 4

    # Create a random input tensor simulating patch embeddings
    x = torch.rand(batch_size, num_patches, embed_dim)

    # Initialize a VitBlock
    block = VitBlock(dim=embed_dim, num_heads=num_heads)

    # Forward pass
    out = block(x)

    # Check output shape
    assert out.shape == (batch_size, num_patches, embed_dim), (
        f"Expected shape {(batch_size, num_patches, embed_dim)}, but got {out.shape}"
    )


if __name__ == "__main__":
    _smoke_test()
