"""
Compute per-layer attention entropy for DINO student & teacher across checkpoints.

Entropy is computed on the post-softmax attention matrix of every block:
    attn: [B, heads, N, N]  (rows sum to 1 over the N=1+H*W tokens)

For each row (query token) we compute Shannon entropy of the distribution
over keys, then normalize by log(N) so values live in [0, 1]:
    0.0 -> attention collapsed onto a single token (maximally "peaky")
    1.0 -> attention uniform over all tokens (maximally "diffuse")

Low, decreasing entropy over training is generally a good sign for DINO:
it means the model is learning to concentrate attention (e.g. on salient
objects) rather than spreading it uniformly. But entropy that collapses
too low too early, or collapses uniformly across ALL images, can also
indicate representational collapse -- that's why we track student vs
teacher separately, and CLS-row vs patch-row entropy separately.

Usage:
    python attention_entropy.py \
        --path-config configs/dino.yaml \
        --ckpt-dir checkpoints/ \
        --ckpt-glob "epoch_*.pt" \
        --images img1.jpg img2.jpg img3.jpg img4.jpg \
        --image-size 224 \
        --output-dir entropy_out/

Adjust the three blocks marked "# ADJUST" below to match your project's
actual import paths / attribute names before running.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from components.utils.config import load_yaml
from components.utils.device import get_device
from components.vit.dino_defs import DINOConfig
from components.vit.dino_inf import load_from_checkpoint
from components.vit.dino_vis import preprocess_image

# --------------------------------------------------------------------------
# Attention capture
# --------------------------------------------------------------------------


class LayerAttentionCatcher:
    """Hook every block's attn_drop and store its (pre-dropout-in-eval) output.

    In eval mode nn.Dropout is a no-op, so this tensor equals the raw
    post-softmax attention matrix, shape [B, heads, N, N].
    """

    def __init__(self):
        self.per_layer: Dict[int, torch.Tensor] = {}

    def make_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.per_layer[layer_idx] = output.detach()

        return _hook

    def reset(self):
        self.per_layer = {}


def register_entropy_hooks(model: nn.Module) -> LayerAttentionCatcher:
    """Attach hooks to every block's attn_drop.

    ADJUST: this path (model.backbone.vit.blocks[i].attn_drop) must match
    your actual module structure. Based on the VitEncoder/VitBlock code
    you shared, this should be correct as-is.
    """
    catcher = LayerAttentionCatcher()
    vit = model.backbone.vit  # ADJUST if your wrapping differs
    for i, block in enumerate(vit.blocks):
        block.attn_drop.register_forward_hook(catcher.make_hook(i))
    return catcher


# --------------------------------------------------------------------------
# Entropy computation
# --------------------------------------------------------------------------


def attention_entropy(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise Shannon entropy, normalized by log(N).

    Args:
        attn: [B, heads, N, N] post-softmax attention, rows sum to 1.

    Returns:
        norm_entropy: [B, heads, N] in [0, 1].
    """
    p = attn.clamp_min(eps)
    raw_entropy = -(p * p.log()).sum(dim=-1)  # [B, heads, N]
    max_entropy = math.log(attn.shape[-1])
    return raw_entropy / max_entropy


@dataclass
class EntropyRow:
    epoch: int
    network: str  # "student" or "teacher"
    image: str
    layer: int
    head: Optional[int]  # None => averaged over heads
    region: str  # "cls" or "patch" or "all"
    entropy: float


def summarize_attention(
    attn: torch.Tensor,
    epoch: int,
    network: str,
    image_name: str,
    layer: int,
) -> List[EntropyRow]:
    """Reduce one layer's attention tensor into a handful of entropy summaries."""
    norm_entropy = attention_entropy(attn)  # [B, heads, N]
    norm_entropy = norm_entropy[0]  # drop batch dim -> [heads, N]

    rows: List[EntropyRow] = []

    # CLS-row entropy (query 0 = CLS token attending to everything).
    # This is the row your attention-map visualizations are built from.
    cls_ent = norm_entropy[:, 0].mean().item()
    rows.append(EntropyRow(epoch, network, image_name, layer, None, "cls", cls_ent))

    # Patch-row entropy (queries 1..N, i.e. how peaky/diffuse patch tokens
    # are on average) -- a proxy for general representational sharpness.
    if norm_entropy.shape[1] > 1:
        patch_ent = norm_entropy[:, 1:].mean().item()
        rows.append(EntropyRow(epoch, network, image_name, layer, None, "patch", patch_ent))

    # All-row entropy (single overall number per layer).
    all_ent = norm_entropy.mean().item()
    rows.append(EntropyRow(epoch, network, image_name, layer, None, "all", all_ent))

    # Per-head CLS entropy -- useful to check if averaging is hiding a
    # sharp head among diffuse ones.
    for h in range(norm_entropy.shape[0]):
        rows.append(EntropyRow(epoch, network, image_name, layer, h, "cls_per_head", norm_entropy[h, 0].item()))

    return rows


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------


def parse_epoch_from_filename(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse epoch number from checkpoint filename: {path}")
    return int(match.group(1))


# --------------------------------------------------------------------------
# Main extraction loop
# --------------------------------------------------------------------------


@torch.no_grad()
def compute_entropy_for_checkpoint(
    config,
    ckpt_path: Path,
    image_paths: List[Path],
    image_size: int,
    device: torch.device,
) -> List[EntropyRow]:
    epoch = parse_epoch_from_filename(ckpt_path)

    rows: List[EntropyRow] = []

    for network_name in ("student", "teacher"):
        model = load_from_checkpoint(config, ckpt_path, device, network_name)
        catcher = register_entropy_hooks(model)

        for img_path in image_paths:
            x, _rgb = preprocess_image(img_path, image_size)
            x = x.unsqueeze(0).to(device)

            catcher.reset()
            _ = model(x)

            for layer_idx, attn in catcher.per_layer.items():
                rows.extend(
                    summarize_attention(
                        attn=attn,
                        epoch=epoch,
                        network=network_name,
                        image_name=img_path.name,
                        layer=layer_idx,
                    )
                )

    return rows


def find_checkpoints(ckpt_dir: Path, ckpt_glob: str) -> List[Path]:
    ckpts = sorted(ckpt_dir.glob(ckpt_glob), key=parse_epoch_from_filename)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} matching '{ckpt_glob}'")
    return ckpts


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def plot_entropy_trends(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Last-layer CLS entropy, student vs teacher, averaged over images.
    last_layer = df["layer"].max()
    subset = (
        df[(df["layer"] == last_layer) & (df["region"] == "cls")]
        .groupby(["epoch", "network"], as_index=False)["entropy"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for network_name, group in subset.groupby("network"):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group["entropy"], marker="o", label=network_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized CLS-attention entropy (last layer)")
    ax.set_title("Last-layer CLS attention entropy: student vs teacher")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cls_entropy_last_layer.png", dpi=150)
    plt.close(fig)

    # 2) Per-layer entropy heatmap-ish line plot for the final checkpoint,
    #    student vs teacher, to see whether depth affects concentration.
    final_epoch = df["epoch"].max()
    per_layer = (
        df[(df["epoch"] == final_epoch) & (df["region"] == "cls")]
        .groupby(["layer", "network"], as_index=False)["entropy"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for network_name, group in per_layer.groupby("network"):
        group = group.sort_values("layer")
        ax.plot(group["layer"], group["entropy"], marker="o", label=network_name)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Normalized CLS-attention entropy")
    ax.set_title(f"Per-layer CLS attention entropy at epoch {final_epoch}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cls_entropy_per_layer_final_epoch.png", dpi=150)
    plt.close(fig)

    # 3) CLS vs patch entropy over epochs, last layer, student only
    #    (sharper CLS row than patch rows is a healthy sign -- CLS should
    #    be more selective than an average patch token).
    cls_patch = (
        df[(df["layer"] == last_layer) & (df["network"] == "student") & (df["region"].isin(["cls", "patch"]))]
        .groupby(["epoch", "region"], as_index=False)["entropy"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for region_name, group in cls_patch.groupby("region"):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group["entropy"], marker="o", label=region_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized entropy (last layer, student)")
    ax.set_title("CLS-row vs patch-row entropy over training (student)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cls_vs_patch_entropy_student.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path-config", type=str, default="./experiments/image_dino/dino_config.yaml", help="Path for the configs"
    )
    p.add_argument("--ckpt-dir", required=True, type=str)
    p.add_argument("--ckpt-glob", default="epoch_*.pth", type=str)
    p.add_argument("--images", nargs="+", required=True, type=str)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--output-dir", required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    config = load_yaml(Path(args.path_config), DINOConfig)
    image_size = args.image_size if args.image_size is not None else config.model_base_res

    image_paths = [Path(p) for p in args.images if Path(p).exists()]
    if not image_paths:
        raise FileNotFoundError("None of the provided --images paths exist.")

    ckpts = find_checkpoints(Path(args.ckpt_dir), args.ckpt_glob)
    print(f"Found {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

    all_rows: List[EntropyRow] = []
    for ckpt_path in ckpts:
        print(f"Processing {ckpt_path.name} ...")
        rows = compute_entropy_for_checkpoint(
            config=config,
            ckpt_path=ckpt_path,
            image_paths=image_paths,
            image_size=image_size,
            device=device,
        )
        all_rows.extend(rows)

    df = pd.DataFrame([r.__dict__ for r in all_rows])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attention_entropy.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw entropy table to {csv_path}")

    plot_entropy_trends(df, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
