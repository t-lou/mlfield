import argparse
from functools import partial
from pathlib import Path

import torch
from components.dataset.a2d2_dataset import A2D2Dataset, Split, bev_collate
from components.definitions.mmperc_params import MmpercParams
from components.mmperc.model.simple_model import SimpleModel
from components.utils.config import load_yaml
from components.utils.device import get_device
from torch.utils.data import DataLoader
from torchviz import make_dot


def main(params: MmpercParams):
    # ------------------------------------------------------------
    # 1. Build dataset + dataloader
    # ------------------------------------------------------------
    dataset = A2D2Dataset(path_tar=Path(params.path_data), params=params, split=Split.TRAIN)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=partial(bev_collate, params=params))

    # ------------------------------------------------------------
    # 2. Get one batch
    # ------------------------------------------------------------
    batch = next(iter(dataloader))

    points = batch["points"]  # (B, N, C)
    camera = batch["camera"]  # (B, 3, H, W)

    # ------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------
    device = get_device()
    model = SimpleModel(params=params).to(device)
    points = points.to(device)
    camera = camera.to(device)

    # ------------------------------------------------------------
    # 4. Forward pass
    # ------------------------------------------------------------
    model_out = model(points, camera)

    # Collect all tensors from the output dict
    tensors = []
    for v in model_out.values():
        if isinstance(v, torch.Tensor):
            tensors.append(v)

    # Combine into a single scalar
    y = sum(t.sum() for t in tensors)

    # ------------------------------------------------------------
    # 5. Build graph
    # ------------------------------------------------------------
    dot = make_dot(
        y,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )

    # ------------------------------------------------------------
    # 6. Export to PDF, PNG, DOT
    # ------------------------------------------------------------
    dot.render("simple_model_graph", format="pdf")
    dot.render("simple_model_graph", format="png")
    dot.save("simple_model_graph.dot")

    print("Exported: simple_model_graph.pdf, .png, .dot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMPERC model graph visualizer")
    parser.add_argument(
        "--path-config",
        type=str,
        default="./experiments/mmperc/mmperc_config.yaml",
        help="Path to MMPERC config YAML",
    )

    args = parser.parse_args()

    cfg = load_yaml(Path(args.path_config), MmpercParams)
    main(cfg)
