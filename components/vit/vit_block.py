import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    """Fallback RMSNorm for PyTorch versions without ``nn.RMSNorm``."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


_NORM_LAYERS = {"LayerNorm": nn.LayerNorm, "RMSNorm": getattr(nn, "RMSNorm", _RMSNorm)}
_ACT_LAYERS = {"GELU": nn.GELU, "SiLU": nn.SiLU, "ReLU": nn.ReLU}


def get_norm_layer(name: str):
    """Return a ``dim -> nn.Module`` factory for the given normalization name."""
    if name not in _NORM_LAYERS:
        raise ValueError(f"Unsupported norm_layer {name!r}, expected one of {tuple(_NORM_LAYERS)}")
    return _NORM_LAYERS[name]


def get_act_layer(name: str):
    """Return the activation class for the given activation name."""
    if name not in _ACT_LAYERS:
        raise ValueError(f"Unsupported act_layer {name!r}, expected one of {tuple(_ACT_LAYERS)}")
    return _ACT_LAYERS[name]


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
        # Sample in fp32: rounding `keep_prob + rand` in bf16/fp16 skews the drop rate
        keep = torch.rand(shape, device=x.device) < keep_prob
        return x * (keep / keep_prob).to(x.dtype)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"


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
        qkv_bias=False,  # reference DINO / DINOv2 / MAE use qkv_bias=True; set per recipe
        norm_layer="LayerNorm",
        act_layer="GELU",
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
            norm_layer: Normalization layer name ("LayerNorm" or "RMSNorm")
            act_layer: MLP activation name ("GELU", "SiLU" or "ReLU")
        """
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        norm_cls = get_norm_layer(norm_layer)
        act_cls = get_act_layer(act_layer)

        # --- LayerNorm before attention ---
        self.norm1 = norm_cls(dim)

        # qkv projection
        # produces [q, k, v] in one fused linear layer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # output projection after attention
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # dropout on attention output

        # stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- MLP ---
        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_cls(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_cls(),
            nn.Dropout(proj_drop),  # dropout inside MLP
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),  # dropout on MLP output
        )

        # attention dropout
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, padding_mask: torch.Tensor | None = None):
        """
        Args:
            x: Tokens (B, N, D)
            padding_mask: Optional bool mask (B, N), True marks padded tokens. Padded tokens are
                ignored as attention keys and zeroed in the returned tensor.
        """
        B, N, D = x.shape

        # --- Attention ---
        x_norm = self.norm1(x)

        # fused qkv projection -> (3, B, heads, N, head_dim) as strided views (no copies)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if padding_mask is not None:
            # SDPA attends where the mask is True, so invert the padding mask.
            attn_mask = ~padding_mask.to(torch.bool)[:, None, None, :]
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)

        # residual + stochastic depth
        x = x + self.drop_path(out)

        # --- MLP ---
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)

        # residual + stochastic depth
        x = x + self.drop_path(mlp_out)

        # Padded tokens never influence valid ones (masked as keys, rows are independent
        # elsewhere), so a single clear at the end keeps them exactly zero.
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

    # Padded tokens must not change the result for valid tokens and come back as zeros
    padding_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    padding_mask[1, 10:] = True
    block.eval()
    masked = block(x, padding_mask=padding_mask)
    short = block(x[1:2, :10])
    assert torch.allclose(masked[1, :10], short[0], atol=1e-5), "Padding leaked into valid tokens"
    assert masked[1, 10:].abs().max() == 0, "Padded tokens must be zero"


if __name__ == "__main__":
    _smoke_test()
