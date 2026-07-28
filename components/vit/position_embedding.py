import math

import torch
import torch.nn.functional as F


def build_2d_sincos_position_embedding(grid_size: int, embed_dim: int, add_cls_token: bool = True) -> torch.Tensor:
    """
    Create 2D sinusoidal position embeddings for Vision Transformer.

    Sinusoidal embeddings have several advantages:
    - No learnable parameters (consistent across datasets)
    - Can extrapolate to longer sequences
    - Each dimension encodes different frequencies

    The embedding combines separate sinusoidal patterns for height and width:
    pos_embed = [sin(w_h*h), cos(w_h*h), sin(w_w*w), cos(w_w*w)]
    where w are frequency weights following transformer conventions.

    Args:
        grid_size: Grid dimension (grid_size x grid_size patches)
        embed_dim: Embedding dimension (must be divisible by 4)
        add_cls_token: If True, prepend zeros for CLS token

    Returns:
        Position embeddings of shape (1, num_patches+cls, embed_dim)

    Raises:
        ValueError: If embed_dim is not divisible by 4

    Example:
        >>> pos_emb = build_2d_sincos_position_embedding(14, 768)
        >>> pos_emb.shape
        torch.Size([1, 197, 768])  # 196 patches + 1 CLS token

    Improvement: Consider adding:
        - Learnable position biases for fine-tuning
        - RoPE (Rotary Position Embeddings) for better extrapolation
        - Interpolation strategy for different resolutions
    """
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos position embeddings")

    pos_embed = positional_encoding_2d(grid_size, grid_size, embed_dim, device=torch.device("cpu"), dtype=torch.float32)

    if add_cls_token:
        # Prepend a zero vector for the CLS token position embedding
        cls_pos = torch.zeros(1, embed_dim, dtype=pos_embed.dtype)
        pos_embed = torch.cat([cls_pos, pos_embed], dim=0)

    return pos_embed.unsqueeze(0)


def positional_encoding_1d(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generate 1D positional encoding for a sequence of length `length` with embedding dimension `dim`.
    """
    half = dim // 2
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    omega = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / max(half, 1))
    angles = pos * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim].to(dtype=dtype)


def positional_encoding_2d(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generate 2D positional encoding for a grid of size (h, w) with embedding dimension dim.
    """
    half = dim // 2
    dim_h = half
    dim_w = dim - half

    emb_h = positional_encoding_1d(h, dim_h, device=device, dtype=dtype)[:, None, :]
    emb_w = positional_encoding_1d(w, dim_w, device=device, dtype=dtype)[None, :, :]

    emb_h = emb_h.expand(h, w, dim_h)
    emb_w = emb_w.expand(h, w, dim_w)
    emb = torch.cat([emb_h, emb_w], dim=-1)
    return emb.view(h * w, dim)


class PosEmbdCache(torch.nn.Module):
    """Small helper that lazily builds and caches positional encodings by shape/device/dtype."""

    def __init__(self):
        super().__init__()
        self._cache_1d = {}
        self._cache_2d = {}

    def get_1d(self, length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (length, dim, device, dtype)
        if key not in self._cache_1d:
            self._cache_1d[key] = positional_encoding_1d(length, dim, device, dtype)
        return self._cache_1d[key]

    def get_2d(self, h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, dim, device, dtype)
        if key not in self._cache_2d:
            self._cache_2d[key] = positional_encoding_2d(h, w, dim, device, dtype)
        return self._cache_2d[key]


def _smoke_test():
    """Smoke test for the build_2d_sincos_position_embedding function."""
    grid_size = 14
    embed_dim = 768
    pos_emb = build_2d_sincos_position_embedding(grid_size, embed_dim)
    assert pos_emb.shape == (1, grid_size * grid_size + 1, embed_dim), "Position embedding shape mismatch"


if __name__ == "__main__":
    _smoke_test()
