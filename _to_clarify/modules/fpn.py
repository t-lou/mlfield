import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, C3, C4, C5, out_channels=256):
        super().__init__()
        # lateral 1×1 convs
        self.l3 = nn.Conv2d(C3, out_channels, 1)
        self.l4 = nn.Conv2d(C4, out_channels, 1)
        self.l5 = nn.Conv2d(C5, out_channels, 1)

        # output 3×3 convs
        self.o3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.o4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.o5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c4, c5):
        # top-down pathway
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        # smoothing
        p3 = self.o3(p3)
        p4 = self.o4(p4)
        p5 = self.o5(p5)

        return p3, p4, p5


class WeightedAdd(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, inputs):
        w = F.relu(self.w)
        w = w / (w.sum() + 1e-6)
        return sum(w[i] * inputs[i] for i in range(len(inputs)))


class BiFPN(nn.Module):
    def __init__(self, C3, C4, C5, out_channels=256):
        super().__init__()

        # unify channels
        self.p3_in = nn.Conv2d(C3, out_channels, 1)
        self.p4_in = nn.Conv2d(C4, out_channels, 1)
        self.p5_in = nn.Conv2d(C5, out_channels, 1)

        # top-down fusion
        self.w4_td = WeightedAdd(2)
        self.w3_td = WeightedAdd(2)

        # bottom-up fusion
        self.w4_bu = WeightedAdd(2)
        self.w5_bu = WeightedAdd(2)

        # output convs
        self.p3_out = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.p4_out = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.p5_out = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c4, c5):
        # unify
        p3 = self.p3_in(c3)
        p4 = self.p4_in(c4)
        p5 = self.p5_in(c5)

        # ----- top-down -----
        p4_td = self.w4_td([p4, F.interpolate(p5, scale_factor=2, mode="nearest")])
        p3_td = self.w3_td([p3, F.interpolate(p4_td, scale_factor=2, mode="nearest")])

        # ----- bottom-up -----
        p4_bu = self.w4_bu([p4_td, F.max_pool2d(p3_td, 2)])
        p5_bu = self.w5_bu([p5, F.max_pool2d(p4_bu, 2)])

        # output convs
        p3_out = self.p3_out(p3_td)
        p4_out = self.p4_out(p4_bu)
        p5_out = self.p5_out(p5_bu)

        return p3_out, p4_out, p5_out


c3 = torch.randn(1, 256, 64, 64)
c4 = torch.randn(1, 512, 32, 32)
c5 = torch.randn(1, 1024, 16, 16)

fpn = FPN(256, 512, 1024)
bifpn = BiFPN(256, 512, 1024)

p3_f, p4_f, p5_f = fpn(c3, c4, c5)
p3_b, p4_b, p5_b = bifpn(c3, c4, c5)

print(p3_f.shape, p4_f.shape, p5_f.shape)
print(p3_b.shape, p4_b.shape, p5_b.shape)
