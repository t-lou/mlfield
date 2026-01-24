from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common.params as params


def init_plot():
    if not params.DEBUG_PLOT_ON:
        return
    path = Path("debug_plots")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def export_bbox_heatmap_debug(pred, gt, id_epoch=0, id_sample=0) -> None:
    if not params.DEBUG_PLOT_ON:
        return

    # assert False, (gt.shape, pred.shape)

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Pred")
    plt.imshow(pred_np, cmap="hot")

    plt.subplot(1, 3, 2)
    plt.title("GT")
    plt.imshow(gt_np, cmap="hot")

    plt.subplot(1, 3, 3)
    plt.title("Pred - GT")
    plt.imshow(pred_np - gt_np, cmap="bwr")

    plt.tight_layout()
    path = Path("debug_plots")
    plt.savefig(path / f"debug_heatmaps_{id_epoch} _{id_sample}.png")
    plt.close()


def export_semantic_debug(self, pred_logits, gt_sem, class_to_color, id_epoch=0, id_sample=0):
    if not params.DEBUG_PLOT_ON:
        return

    # ---------------------------------------------------------
    # Convert tensors → numpy
    # pred_logits: (C, H, W)
    # gt_sem:      (H, W)
    # ---------------------------------------------------------
    pred_np = pred_logits.detach().cpu().argmax(dim=0).numpy()  # (H, W)
    gt_np = gt_sem.detach().cpu().numpy()  # (H, W)

    # ---------------------------------------------------------
    # Build RGB maps
    # ---------------------------------------------------------
    H, W = pred_np.shape
    pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # class_to_color is a list of (cid, (R,G,B))
    for cid, rgb in class_to_color:
        pred_rgb[pred_np == cid] = rgb
        gt_rgb[gt_np == cid] = rgb

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title("Pred (RGB)")
    plt.imshow(pred_rgb)

    plt.subplot(1, 2, 2)
    plt.title("GT (RGB)")
    plt.imshow(gt_rgb)

    plt.tight_layout()
    path = Path("debug_plots")
    plt.savefig(path / f"debug_sem_{id_epoch}_{id_sample}.png")
    plt.close()
