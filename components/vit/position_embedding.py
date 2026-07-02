import torch


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

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    pos_h = grid[0].reshape(-1)
    pos_w = grid[1].reshape(-1)

    # Compute frequency weights for sinusoidal embeddings, following the transformer convention
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000**omega)  # Frequency scaling for sinusoidal embeddings

    out_h = torch.outer(pos_h, omega)
    out_w = torch.outer(pos_w, omega)
    pos_embed = torch.cat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)

    if add_cls_token:
        # Prepend a zero vector for the CLS token position embedding
        cls_pos = torch.zeros(1, embed_dim, dtype=pos_embed.dtype)
        pos_embed = torch.cat([cls_pos, pos_embed], dim=0)

    return pos_embed.unsqueeze(0)
