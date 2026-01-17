from pathlib import Path

import common.params as params
import matplotlib.pyplot as plt
from torch import Tensor


class DebugPloter:
    def __init__(self):
        self._active = params.DEBUG_PLOT_ON
        self._path = Path("debug_plots")
        if self._active and not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)
        self._scatter_output_counter = 0
        self._heatmap_counter = 0
        self._lidar_density_counter = 0
        self._debug_heatmap_counter = 0

    def plot_scatter_output(self, scatter_output: Tensor, idx: int | None = None) -> None:
        if not self._active:
            return
        if idx is None:
            idx = self._scatter_output_counter
            self._scatter_output_counter += 1
        density = (scatter_output[0].sum(dim=0) > 0).float().cpu().numpy()
        plt.imshow(density, cmap="gray")
        plt.savefig(self._path / f"scatter_output_{idx}.png")
        plt.close()

    def plot_heatmap(self, heatmap: Tensor, idx: int | None = None) -> None:
        if not self._active:
            return
        if idx is None:
            idx = self._heatmap_counter
            self._heatmap_counter += 1
        idx = idx if idx is not None else self._scatter_output_counter
        hm = heatmap[0, 0].detach().cpu().numpy()
        plt.imshow(hm, cmap="hot")
        plt.colorbar()
        plt.savefig(self._path / f"heatmap_pred_{idx}.png")
        plt.close()

    def plot_lidar_density(self, scatter_output: Tensor, idx: int | None = None) -> None:
        if not self._active:
            return
        if idx is None:
            idx = self._lidar_density_counter
            self._lidar_density_counter += 1
        idx = idx if idx is not None else self._scatter_output_counter
        density = (scatter_output[0].sum(dim=0) > 0).float().cpu().numpy()
        plt.imshow(density, cmap="gray")
        plt.savefig(self._path / f"lidar_density_{idx}.png")
        plt.close()

    def export_heatmap_debug(self, pred, gt, idx: int | None = None) -> None:
        if not self._active:
            return
        if idx is None:
            idx = self._debug_heatmap_counter
            self._debug_heatmap_counter += 1
        pred_np = pred[0, 0].detach().cpu().numpy()
        gt_np = gt[0, 0].detach().cpu().numpy()

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
        plt.savefig(self._path / f"debug_heatmaps_{idx}.png")
        plt.close()
