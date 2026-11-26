"""Multi-block mask sampling for I-JEPA (Assran et al., 2023).

Pure tensor functions without any model dependency, so they can be used from the model, from
DataLoader collate functions (mask prefetching) and from tests. Everything is vectorized and free
of host syncs.
"""

from __future__ import annotations

import torch


def sample_rect_masks(
    grid_h: int,
    grid_w: int,
    scale_min: float,
    scale_max: float,
    ar_min: float,
    ar_max: float,
    num_masks: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample rectangular boolean masks on a patch grid.

    Each rectangle gets an area in [scale_min, scale_max] of the grid and an aspect ratio in
    [ar_min, ar_max]; its position is uniform over all placements that keep it fully inside the
    grid, so the sampled size is never truncated.

    Returns:
        Tensor of shape (num_masks, grid_h * grid_w) with bool dtype
    """
    total = grid_h * grid_w
    areas = torch.empty(num_masks, device=device).uniform_(scale_min, scale_max, generator=generator) * total
    aspects = torch.empty(num_masks, device=device).uniform_(ar_min, ar_max, generator=generator)

    h = torch.round(torch.sqrt(areas * aspects)).long().clamp(1, grid_h)
    w = torch.round(torch.sqrt(areas / aspects)).long().clamp(1, grid_w)

    # uniform over the (grid - size + 1) valid placements
    top = (torch.rand(num_masks, device=device, generator=generator) * (grid_h - h + 1)).long()
    left = (torch.rand(num_masks, device=device, generator=generator) * (grid_w - w + 1)).long()
    top = torch.minimum(top, grid_h - h)
    left = torch.minimum(left, grid_w - w)

    ys = torch.arange(grid_h, device=device).view(1, -1, 1)
    xs = torch.arange(grid_w, device=device).view(1, 1, -1)
    top, left, h, w = (t.view(-1, 1, 1) for t in (top, left, h, w))
    masks = (ys >= top) & (ys < top + h) & (xs >= left) & (xs < left + w)  # (num_masks, H, W)
    return masks.reshape(num_masks, total)


def sample_context_and_target_masks(
    cfg,
    batch_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample per-image context and target masks as in the I-JEPA paper.

    ``num_target_blocks`` target blocks and one large context block are sampled independently.
    The context is the context block with all target blocks removed, so the context encoder never
    sees what it has to predict. Target blocks may overlap each other.

    Args:
        cfg: IJEPAConfig (uses image_size, patch_size, context_*, target_*, aspect_ratio_*).

    Returns:
        context_keep_mask: (B, N) bool
        target_masks: list of num_target_blocks tensors (B, N) bool
    """
    grid = cfg.image_size // cfg.patch_size
    num_blocks = cfg.num_target_blocks

    targets = sample_rect_masks(
        grid,
        grid,
        cfg.target_scale_min,
        cfg.target_scale_max,
        cfg.aspect_ratio_min,
        cfg.aspect_ratio_max,
        num_blocks * batch_size,
        device,
        generator,
    ).view(num_blocks, batch_size, -1)

    context_block = sample_rect_masks(
        grid,
        grid,
        cfg.context_scale_min,
        cfg.context_scale_max,
        cfg.context_aspect_ratio_min,
        cfg.context_aspect_ratio_max,
        batch_size,
        device,
        generator,
    )

    context = context_block & ~targets.any(dim=0)
    # Degenerate configs (targets covering the whole context block) would leave nothing to
    # encode; fall back to the plain context block for those samples.
    context = torch.where(context.any(dim=1, keepdim=True), context, context_block)
    return context, list(targets.unbind(0))
