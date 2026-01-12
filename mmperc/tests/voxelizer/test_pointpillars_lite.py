import torch
from src.common.device import get_best_device
from src.voxelizer.pointpillars_lite import TorchPillarVoxelizer, TorchPointCloud


def make_pc(points):
    """Helper to create a TorchPointCloud from a list of xyzit rows."""
    arr = torch.tensor(points, dtype=torch.float32)
    return TorchPointCloud(points=arr)


def test_basic_voxelization():
    device = get_best_device()

    # Two points in the same pillar
    pc = make_pc(
        [
            [0.1, 0.1, 0.0, 0.5, 0.0],
            [0.2, 0.1, 0.0, 0.6, 0.0],
        ]
    )

    voxelizer = TorchPillarVoxelizer(
        x_range=(0, 10), y_range=(0, 10), voxel_size=(1.0, 1.0, 8.0), max_points_per_pillar=10, max_pillars=100
    )

    out = voxelizer(pc.points.to(device))

    assert out["pillar_coords"].shape[0] == 1
    assert out["pillar_count"][0].item() == 2
    assert torch.allclose(out["pillars"][0, 0, :4], torch.tensor([0.1, 0.1, 0.0, 0.5], device=device))


def test_two_different_pillars():
    device = get_best_device()

    pc = make_pc(
        [
            [0.1, 0.1, 0.0, 0.5, 0.0],  # pillar (0,0)
            [1.2, 0.1, 0.0, 0.6, 0.0],  # pillar (1,0)
        ]
    )

    voxelizer = TorchPillarVoxelizer(
        x_range=(0, 10), y_range=(0, 10), voxel_size=(1.0, 1.0, 8.0), max_points_per_pillar=10, max_pillars=100
    )

    out = voxelizer(pc.points.to(device))

    coords = out["pillar_coords"].cpu().numpy()
    assert sorted(coords.tolist()) == [[0, 0], [1, 0]]
    assert out["pillar_count"].sum().item() == 2


def test_range_filtering():
    device = get_best_device()

    pc = make_pc(
        [
            [-5.0, 0.0, 0.0, 0.5, 0.0],  # outside x-range
            [2.0, 2.0, 0.0, 0.6, 0.0],  # inside
        ]
    )

    voxelizer = TorchPillarVoxelizer(
        x_range=(0, 10), y_range=(0, 10), voxel_size=(1.0, 1.0, 8.0), max_points_per_pillar=10, max_pillars=100
    )

    out = voxelizer(pc.points.to(device))

    assert out["pillar_coords"].shape[0] == 1
    assert out["pillar_count"][0].item() == 1


def test_max_points_per_pillar():
    device = get_best_device()

    # 5 points in same pillar, but limit is 3
    pc = make_pc(
        [
            [0.1, 0.1, 0.0, 0.1, 0.0],
            [0.2, 0.1, 0.0, 0.2, 0.0],
            [0.3, 0.1, 0.0, 0.3, 0.0],
            [0.4, 0.1, 0.0, 0.4, 0.0],
            [0.5, 0.1, 0.0, 0.5, 0.0],
        ]
    )

    voxelizer = TorchPillarVoxelizer(
        x_range=(0, 10), y_range=(0, 10), voxel_size=(1.0, 1.0, 8.0), max_points_per_pillar=3, max_pillars=100
    )

    out = voxelizer(pc.points.to(device))

    assert out["pillar_count"][0].item() == 3
    assert torch.all(out["pillars"][0, 3:, :].eq(0))  # remaining slots are zero


def test_max_pillars_limit():
    device = get_best_device()

    # Create 200 unique pillars, but limit is 50
    points = []
    for i in range(200):
        x = i * 1.0 + 0.1
        points.append([x, 0.1, 0.0, 1.0, 0.0])

    pc = make_pc(points)

    voxelizer = TorchPillarVoxelizer(
        x_range=(0, 1000), y_range=(0, 10), voxel_size=(1.0, 1.0, 8.0), max_points_per_pillar=5, max_pillars=50
    )

    out = voxelizer(pc.points.to(device))

    assert out["pillar_coords"].shape[0] == 50
    assert out["pillar_count"].sum().item() == 50
