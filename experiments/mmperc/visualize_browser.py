#!/usr/bin/env python3
"""
Interactively browse frames of an A2D2Dataset one at a time, advancing with a
single keypress.

A2D2Dataset already shuffles its frame indexing once (fixed seed) at
construction time, so simply walking `dataset.indexing` in order visits
frames in a randomized-but-reproducible "reading order" - no extra
randomization needed here.

  - 2D mode: one matplotlib window, camera image + BEV plot side by side.
  - 3D mode: a PyVista window (point cloud + GT boxes) and a *separate*
    matplotlib window (camera image), kept in sync by frame index - press a
    key in either window and both update.

Keys:
    n / Right   -> next frame
    b / Left    -> previous frame
    q           -> quit

Usage:
    python visualize_browser.py --path-config ./mmperc_config.yaml --mode 2d
    python visualize_browser.py --path-config ./mmperc_config.yaml --mode 3d --start-index 10 --split val

Requires: numpy, matplotlib, pyvista (3D mode only), plus whatever
A2D2Dataset itself needs (torch, PIL, the surrounding `components` package).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from components.dataset.a2d2_dataset import A2D2Dataset, Split
from components.definitions.mmperc_params import MmpercParams
from components.mmperc.common.visualization import (
    add_boxes_3d,
    add_point_cloud,
    camera_chw_to_uint8_hwc,
    plot_bev,
    show_camera_image,
    strip_zero_padded_rows,
    to_numpy,
)
from components.utils.config import load_yaml

NEXT_KEYS = {"n", "right"}
PREV_KEYS = {"b", "left"}
QUIT_KEYS = {"q"}


def _load_sample(dataset: A2D2Dataset, index: int) -> dict:
    item = dataset[index]
    points = strip_zero_padded_rows(to_numpy(item["points"]))[:, :3]
    gt_boxes = strip_zero_padded_rows(to_numpy(item["gt_boxes"]))
    camera = camera_chw_to_uint8_hwc(to_numpy(item["camera"]))
    return {"points": points, "gt_boxes": gt_boxes, "camera": camera}


# ================================================================
# 2D browser (matplotlib only)
# ================================================================


class Browser2D:
    def __init__(self, dataset: A2D2Dataset, start_index: int = 0):
        import matplotlib.pyplot as plt

        self.dataset = dataset
        self.index = start_index % len(dataset)

        self.fig, (self.ax_img, self.ax_bev) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    def _render(self) -> None:
        sample = _load_sample(self.dataset, self.index)
        show_camera_image(self.ax_img, sample["camera"])
        plot_bev(
            self.ax_bev,
            sample["points"],
            box_sets=[(sample["gt_boxes"], "limegreen", "GT")],
            title=f"Frame {self.index}/{len(self.dataset) - 1}",
        )
        self.fig.canvas.draw_idle()
        print(
            f"[{self.index}/{len(self.dataset) - 1}] points={len(sample['points'])}, gt boxes={len(sample['gt_boxes'])}"
        )

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in NEXT_KEYS:
            self.index = (self.index + 1) % len(self.dataset)
            self._render()
        elif key in PREV_KEYS:
            self.index = (self.index - 1) % len(self.dataset)
            self._render()
        elif key in QUIT_KEYS:
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


# ================================================================
# 3D browser (PyVista window + synced matplotlib image window)
# ================================================================


class Browser3D:
    def __init__(self, dataset: A2D2Dataset, start_index: int = 0):
        import matplotlib.pyplot as plt
        import pyvista as pv

        self.dataset = dataset
        self.index = start_index % len(dataset)
        self._actors_3d = []

        # Camera image window (matplotlib) - non-blocking; the pyvista window
        # (opened next) is what actually blocks the process.
        self.fig, self.ax_img = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title("Camera (front center)")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_mpl)

        # 3D window (PyVista)
        self.plotter = pv.Plotter(title="LIDAR + GT boxes")
        self.plotter.add_axes_at_origin(labels_off=True, line_width=3)
        self.plotter.add_key_event("n", self.next_frame)
        self.plotter.add_key_event("Right", self.next_frame)
        self.plotter.add_key_event("b", self.prev_frame)
        self.plotter.add_key_event("Left", self.prev_frame)
        self.plotter.add_key_event("q", self._quit)

        self._render()

    def _render(self) -> None:
        sample = _load_sample(self.dataset, self.index)

        # Camera panel
        show_camera_image(self.ax_img, sample["camera"], title=f"Frame {self.index}/{len(self.dataset) - 1}")
        self.fig.canvas.draw_idle()

        # 3D panel
        for actor in self._actors_3d:
            self.plotter.remove_actor(actor, render=False)
        self._actors_3d = [add_point_cloud(self.plotter, sample["points"])]
        self._actors_3d += add_boxes_3d(self.plotter, sample["gt_boxes"], color=(0.0, 1.0, 0.0))
        self.plotter.render()

        print(
            f"[{self.index}/{len(self.dataset) - 1}] points={len(sample['points'])}, gt boxes={len(sample['gt_boxes'])}"
        )

    def next_frame(self) -> None:
        self.index = (self.index + 1) % len(self.dataset)
        self._render()

    def prev_frame(self) -> None:
        self.index = (self.index - 1) % len(self.dataset)
        self._render()

    def _on_key_mpl(self, event) -> None:
        key = (event.key or "").lower()
        if key in NEXT_KEYS:
            self.next_frame()
        elif key in PREV_KEYS:
            self.prev_frame()
        elif key in QUIT_KEYS:
            self._quit()

    def _quit(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)
        self.plotter.close()

    def show(self) -> None:
        import matplotlib.pyplot as plt

        # Open the PyVista window without blocking, so matplotlib's event
        # loop (started below) can drive both windows. A periodic timer pumps
        # the PyVista render window so it stays responsive while matplotlib's
        # loop is the one actually running.
        self.plotter.show(interactive_update=True, auto_close=False)

        timer = self.fig.canvas.new_timer(interval=50)
        timer.add_callback(lambda: self.plotter.update())
        timer.start()

        plt.show()  # blocks here; this is the main event loop for both windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively browse A2D2Dataset frames one at a time")
    parser.add_argument(
        "--path-config",
        type=str,
        default="./experiments/mmperc/mmperc_config.yaml",
        help="Path to MMPERC config YAML",
    )
    parser.add_argument("--mode", choices=["2d", "3d"], default="3d", help="Visualization mode (default: 3d)")
    parser.add_argument("--start-index", type=int, default=0, help="Frame index to start on (default: 0)")
    parser.add_argument(
        "--split", choices=["train", "val", "full"], default="full", help="Dataset split to browse (default: full)"
    )
    args = parser.parse_args()

    params = load_yaml(Path(args.path_config), MmpercParams)

    dataset = A2D2Dataset(params=params, split=Split(args.split))

    print("Keys: n/Right = next frame, b/Left = previous frame, q = quit")
    browser_cls = Browser2D if args.mode == "2d" else Browser3D
    browser = browser_cls(dataset, start_index=args.start_index)
    browser.show()


if __name__ == "__main__":
    main()
