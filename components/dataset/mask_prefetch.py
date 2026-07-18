"""Mask prefetching utilities for DataLoader-based mask generation (for JEPA)."""

from typing import List, Tuple

import torch

# In I-JEPA training, the mask generation and processing can be a bottleneck. This module provides functions to
# prefetch masks on the CPU (or any device) in a vectorized manner, avoiding Python loops and .item() calls. The
# masks are generated in batches and can be used directly in the DataLoader's collate_fn.


def sample_rect_mask_cpu(
    grid_h: int,
    grid_w: int,
    scale_min: float,
    scale_max: float,
    ar_min: float,
    ar_max: float,
    num_masks: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample rectangular boolean masks on a patch grid (CPU-optimized version).

    Fully vectorized, no .item() calls. Can run on CPU or GPU.

    Args:
        grid_h: Height of patch grid
        grid_w: Width of patch grid
        scale_min: Minimum scale (fraction of total patches)
        scale_max: Maximum scale
        ar_min: Minimum aspect ratio
        ar_max: Maximum aspect ratio
        num_masks: Number of masks to sample
        device: Device to create tensors on

    Returns:
        Tensor of shape (num_masks, grid_h * grid_w) with bool dtype
    """
    total = grid_h * grid_w

    # Batch sample areas and aspects
    areas = torch.empty(num_masks, device=device).uniform_(scale_min, scale_max) * total
    aspects = torch.empty(num_masks, device=device).uniform_(ar_min, ar_max)

    # Compute heights and widths vectorized
    h_vals = torch.round((areas * aspects) ** 0.5).int()
    w_vals = torch.round((areas / aspects) ** 0.5).int()
    h_vals = torch.clamp(h_vals, min=1, max=grid_h)
    w_vals = torch.clamp(w_vals, min=1, max=grid_w)

    # Sample positions vectorized
    top_vals = torch.randint(0, max(1, grid_h - 1), (num_masks,), device=device)
    left_vals = torch.randint(0, max(1, grid_w - 1), (num_masks,), device=device)

    # Clamp heights/widths to fit within grid from sampled positions
    h_vals = torch.clamp(h_vals, max=grid_h - top_vals)
    w_vals = torch.clamp(w_vals, max=grid_w - left_vals)

    # Build all masks at once using broadcasting (NO PYTHON LOOP)
    y_grid = torch.arange(grid_h, device=device).view(-1, 1)  # (H, 1)
    x_grid = torch.arange(grid_w, device=device).view(1, -1)  # (1, W)

    # Reshape for broadcasting with batch dimension
    top_vals_expanded = top_vals.view(-1, 1, 1)  # (B, 1, 1)
    left_vals_expanded = left_vals.view(-1, 1, 1)  # (B, 1, 1)
    h_vals_expanded = h_vals.view(-1, 1, 1)  # (B, 1, 1)
    w_vals_expanded = w_vals.view(-1, 1, 1)  # (B, 1, 1)

    y_grid_expanded = y_grid.view(1, -1, 1)  # (1, H, 1)
    x_grid_expanded = x_grid.view(1, 1, -1)  # (1, 1, W)

    # Create masks via broadcasting (fully vectorized)
    in_y_range = (y_grid_expanded >= top_vals_expanded) & (
        y_grid_expanded < top_vals_expanded + h_vals_expanded
    )  # (B, H, 1)
    in_x_range = (x_grid_expanded >= left_vals_expanded) & (
        x_grid_expanded < left_vals_expanded + w_vals_expanded
    )  # (1, 1, W)

    masks = (in_y_range & in_x_range).view(num_masks, grid_h * grid_w)  # (B, H*W)

    return masks if num_masks > 1 else masks[0]


def generate_masks_for_batch(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    context_scale_min: float,
    context_scale_max: float,
    target_scale_min: float,
    target_scale_max: float,
    aspect_ratio_min: float,
    aspect_ratio_max: float,
    num_target_blocks: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Generate context and target masks for a batch (CPU-friendly, for DataLoader prefetch).

    Args:
        batch_size: Number of samples in batch
        grid_h, grid_w: Patch grid dimensions
        context_scale_min/max: Scale range for context mask
        target_scale_min/max: Scale range for target masks
        aspect_ratio_min/max: Aspect ratio range
        num_target_blocks: Number of target blocks to sample
        device: Device to create tensors on

    Returns:
        (context_masks, target_masks) where:
        - context_masks: (B, H*W) bool
        - target_masks: list[(B, H*W) bool]
    """
    # Batch sample all context masks at once
    context_masks_all = sample_rect_mask_cpu(
        grid_h,
        grid_w,
        context_scale_min,
        context_scale_max,
        aspect_ratio_min,
        aspect_ratio_max,
        num_masks=batch_size,
        device=torch.device("cpu"),
    ).to(device)  # (B, N)

    # Sample all target masks at once for each target block
    target_masks: List[torch.Tensor] = []
    for block_idx in range(num_target_blocks):
        target_masks_all = sample_rect_mask_cpu(
            grid_h,
            grid_w,
            target_scale_min,
            target_scale_max,
            aspect_ratio_min,
            aspect_ratio_max,
            num_masks=batch_size,
            device=torch.device("cpu"),
        ).to(device)  # (B, N)

        # Apply context mask constraint: target cannot overlap context
        available = ~context_masks_all  # (B, N)
        target_masks_all = target_masks_all & available

        # Handle empty target masks (vectorized, no Python loops)
        empty_mask = target_masks_all.sum(dim=1) == 0  # (B,) bool

        if empty_mask.any():
            empty_indices = torch.where(empty_mask)[0]
            for idx in empty_indices:
                avail_positions = torch.where(available[idx])[0]
                if avail_positions.numel() > 0:
                    random_offset = torch.randint(0, avail_positions.numel(), (1,), device=device)
                    target_masks_all[idx, avail_positions[random_offset]] = True
                else:
                    random_pos = torch.randint(0, target_masks_all.shape[1], (1,), device=device)
                    target_masks_all[idx, random_pos] = True

        target_masks.append(target_masks_all)

    return context_masks_all, target_masks


def collate_fn_with_masks(
    batch,
    config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Custom collate function that prefetches masks alongside images.

    Use this as the collate_fn for DataLoader when using mask prefetching.

    Args:
        batch: List of (image_tensor,) tuples from dataset
        config: IJEPAConfig object
        device: Device to generate masks on

    Returns:
        (images, context_masks, target_masks_list) ready for model forward()
    """
    # Stack images
    images = torch.stack([item[0] if isinstance(item, tuple) else item for item in batch])
    assert images.shape[1] == 3, "Images must have 3 channels (RGB)"
    assert images.shape[2] == config.image_size and images.shape[3] == config.image_size, (
        f"Images must be of size ({config.image_size}, {config.image_size}), got {images.shape[2:]}"
    )

    # Generate masks on CPU (or specified device)
    assert config.image_size % config.patch_size == 0, "Image size must be divisible by patch size"
    grid_h = config.image_size // config.patch_size
    grid_w = config.image_size // config.patch_size

    context_masks, target_masks = generate_masks_for_batch(
        batch_size=images.shape[0],
        grid_h=grid_h,
        grid_w=grid_w,
        context_scale_min=config.context_scale_min,
        context_scale_max=config.context_scale_max,
        target_scale_min=config.target_scale_min,
        target_scale_max=config.target_scale_max,
        aspect_ratio_min=config.aspect_ratio_min,
        aspect_ratio_max=config.aspect_ratio_max,
        num_target_blocks=config.num_target_blocks,
        device=device,
    )

    return images, context_masks, target_masks
