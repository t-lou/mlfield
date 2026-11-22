import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from components.mmperc.encoder.point_transformer_v3 import (
    PointGridSpec,
    PointTransformerV3Config,
    build_neighbors,
)
from components.utils.onnx_export import export_onnx
from experiments.ssl.sonata import (
    SonataBackboneExport,
    SonataConfig,
    SonataNet,
    build_train_loader,
)


def test_ply_semantic_labels_are_not_model_features():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "sample.ply"
        path.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar semantic\n"
            "property uchar intensity\n"
            "end_header\n"
            "0 0 0 1 4\n"
            "1 0 0 2 5\n"
        )

        dataset, _ = build_train_loader(temp_dir, points_per_cloud=2, ignore_fields=("semantic",))
        item = dataset[0]

        assert dataset.feature_fields == ("intensity",)
        assert dataset.in_channels == 1
        assert item["global_views"][0]["feat"].shape == (2, 1)


def test_sonata_backbone_onnx_export():
    backbone = PointTransformerV3Config(
        in_channels=3,
        order=("z",),
        enc_depths=(1, 1),
        enc_channels=(4, 8),
        enc_num_head=(1, 2),
        enc_patch_size=(8, 8),
        stride=(2,),
        drop_path=0.0,
        serial_depth=8,
        stem_kernel=3,
        enc_mode=True,
    )
    cfg = SonataConfig(
        backbone=backbone,
        up_cast_level=1,
        num_prototypes=4,
        head_hidden=8,
        head_bottleneck=4,
    )
    model = SonataBackboneExport(SonataNet(cfg).backbone).eval()
    point_ids = torch.arange(27)
    grid = torch.stack((point_ids // 9, (point_ids // 3) % 3, point_ids % 3), dim=1)
    feat = torch.randn(27, 3)
    batch = torch.zeros(27, dtype=torch.long)

    with TemporaryDirectory() as temp_dir:
        path = export_onnx(
            model,
            (
                feat,
                grid,
                batch,
            ),
            Path(temp_dir) / "sonata.onnx",
            input_names=("features", "grid_coord", "batch"),
            output_names=("encoded_features",),
            dynamic_axes={},
        )
        assert path.is_file()
        ort = __import__("onnxruntime")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            eager = model(feat, grid, batch)
        exported = torch.as_tensor(
            session.run(
                ["encoded_features"],
                {
                    "features": feat.numpy(),
                    "grid_coord": grid.numpy(),
                    "batch": batch.numpy(),
                },
            )[0]
        )
        torch.testing.assert_close(eager, exported, rtol=1e-4, atol=1e-5)


def test_neighbor_export_matches_eager():
    ort = __import__("onnxruntime")

    class NeighborProbe(nn.Module):
        def forward(self, grid_coord, batch):
            return build_neighbors(grid_coord, batch, 3, 16)

    torch.manual_seed(123)
    random_grid = torch.unique(torch.randint(0, 8, (64, 3), dtype=torch.long), dim=0)
    cases = [
        (
            torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 2, 2]], dtype=torch.long),
            torch.tensor([0, 0, 0, 1], dtype=torch.long),
        ),
        (
            random_grid,
            torch.zeros(random_grid.shape[0], dtype=torch.long),
        ),
    ]

    with TemporaryDirectory() as temp_dir:
        path = export_onnx(
            NeighborProbe().eval(),
            cases[1],
            Path(temp_dir) / "neighbors.onnx",
            input_names=("grid_coord", "batch"),
            output_names=("neighbors",),
            dynamic_axes={
                "grid_coord": {0: "points"},
                "batch": {0: "points"},
                "neighbors": {0: "points"},
            },
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for grid_coord, batch in cases:
        eager = build_neighbors(grid_coord, batch, 3, 16)
        exported = torch.as_tensor(
            session.run(
                ["neighbors"],
                {
                    "grid_coord": grid_coord.numpy(),
                    "batch": batch.numpy(),
                },
            )[0]
        )
        assert torch.equal(eager, exported)


def test_sonata_geometry_config():
    grid = PointGridSpec(
        min_xyz=(0.0, -48.0, -5.0),
        max_xyz=(128.0, 48.0, 3.0),
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    cfg = SonataConfig(input_grid=grid, output_voxel_size_xyz=8.0)

    assert cfg.input_grid == grid
    assert cfg.backbone.serial_depth == 7
    assert cfg.backbone.stride is not None
    assert math.prod(cfg.backbone.stride) == 8


def test_sonata_rejects_anisotropic_geometry_output():
    grid = PointGridSpec(
        min_xyz=(0.0, -48.0, -5.0),
        max_xyz=(128.0, 48.0, 3.0),
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    try:
        SonataConfig(input_grid=grid, output_voxel_size_xyz=(2.0, 4.0, 2.0))
    except ValueError as err:
        assert "isotropic" in str(err)
    else:
        raise AssertionError("Sonata must reject anisotropic scalar-stride output geometry")


if __name__ == "__main__":
    test_ply_semantic_labels_are_not_model_features()
    test_sonata_backbone_onnx_export()
    test_neighbor_export_matches_eager()
    test_sonata_geometry_config()
    test_sonata_rejects_anisotropic_geometry_output()
