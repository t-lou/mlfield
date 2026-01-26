import torch
from torch.utils.data import DataLoader
from torchviz import make_dot

import common.params as params
from datasets.a2d2_dataset import A2D2Dataset, bev_collate
from model.simple_model import SimpleModel


def main():
    # ------------------------------------------------------------
    # 1. Build dataset + dataloader
    # ------------------------------------------------------------
    path_dataset = params.PATH_TRAIN
    dataset = A2D2Dataset(root=path_dataset)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=bev_collate)

    # ------------------------------------------------------------
    # 2. Get one batch
    # ------------------------------------------------------------
    batch = next(iter(dataloader))

    points = batch["points"]  # (B, P, C)
    camera = batch["camera"]  # (B, 3, H, W)

    # ------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------
    device = "cpu"
    model = SimpleModel().to(device)
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
    main()
