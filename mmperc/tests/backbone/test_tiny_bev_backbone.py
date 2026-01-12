import torch
from src.backbone.tiny_bev_backbone import TinyBEVBackbone
from src.common.device import get_best_device


def test_forward_shape():
    device = get_best_device()

    model = TinyBEVBackbone(in_channels=64).to(device)
    x = torch.randn(1, 64, 200, 200, device=device)

    y = model(x)

    # Expect downsample by factor 2
    assert y.shape == (1, 128, 100, 100)


def test_batch_support():
    device = get_best_device()

    model = TinyBEVBackbone(in_channels=64).to(device)
    x = torch.randn(4, 64, 200, 200, device=device)

    y = model(x)

    assert y.shape == (4, 128, 100, 100)


def test_gradients_flow():
    device = get_best_device()

    model = TinyBEVBackbone(in_channels=64).to(device)
    x = torch.randn(2, 64, 200, 200, device=device, requires_grad=True)

    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_device_consistency():
    device = get_best_device()

    model = TinyBEVBackbone(in_channels=64).to(device)
    x = torch.randn(1, 64, 200, 200, device=device)

    y = model(x)

    assert y.device == device


def test_deterministic_forward():
    device = get_best_device()

    torch.manual_seed(123)
    model = TinyBEVBackbone(in_channels=64).to(device)
    x = torch.randn(1, 64, 200, 200, device=device)

    y1 = model(x)

    torch.manual_seed(123)
    model2 = TinyBEVBackbone(in_channels=64).to(device)
    x2 = torch.randn(1, 64, 200, 200, device=device)

    y2 = model2(x2)

    assert torch.allclose(y1, y2, atol=1e-6)
