import torch
from device_utils import get_best_device
from fusion.futr_fusion import FuTrFusionBlock


def test_futr_forward_shape_no_cam():
    device = get_best_device()

    B, C, H, W = 2, 128, 100, 100
    bev = torch.randn(B, C, H, W, device=device)

    fusion = FuTrFusionBlock(bev_channels=C).to(device)
    out = fusion(bev, cam_tokens=None)

    assert out.shape == (B, C, H, W)


def test_futr_forward_shape_with_cam():
    device = get_best_device()

    B, C, H, W = 1, 128, 100, 100
    bev = torch.randn(B, C, H, W, device=device)
    cam_tokens = torch.randn(B, 64, C, device=device)

    fusion = FuTrFusionBlock(bev_channels=C).to(device)
    out = fusion(bev, cam_tokens)

    assert out.shape == (B, C, H, W)


def test_futr_gradients():
    device = get_best_device()

    B, C, H, W = 1, 128, 50, 50
    bev = torch.randn(B, C, H, W, device=device, requires_grad=True)
    cam_tokens = torch.randn(B, 32, C, device=device)

    fusion = FuTrFusionBlock(bev_channels=C).to(device)
    out = fusion(bev, cam_tokens)

    loss = out.mean()
    loss.backward()

    assert bev.grad is not None
    assert torch.isfinite(bev.grad).all()


def test_futr_device_consistency():
    device = get_best_device()

    fusion = FuTrFusionBlock(bev_channels=128).to(device)
    bev = torch.randn(1, 128, 100, 100, device=device)
    cam_tokens = torch.randn(1, 32, 128, device=device)

    out = fusion(bev, cam_tokens)

    assert out.device == device


def test_futr_determinism():
    device = get_best_device()

    torch.manual_seed(42)
    fusion1 = FuTrFusionBlock(bev_channels=128).to(device)
    bev1 = torch.randn(1, 128, 50, 50, device=device)
    cam1 = torch.randn(1, 32, 128, device=device)
    out1 = fusion1(bev1, cam1)

    torch.manual_seed(42)
    fusion2 = FuTrFusionBlock(bev_channels=128).to(device)
    bev2 = torch.randn(1, 128, 50, 50, device=device)
    cam2 = torch.randn(1, 32, 128, device=device)
    out2 = fusion2(bev2, cam2)

    assert torch.allclose(out1, out2, atol=1e-6)
