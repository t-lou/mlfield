A drivable‑area head is a great next step — it’s lightweight, fast, and gives you a clean end‑to‑end path from **BEV backbone → fusion → segmentation output**. Let’s build a head that fits your current architecture perfectly: tiny, torch‑native, and easy to extend later.

Below is a compact, production‑ready module you can drop into `src/tasks/`.

---

# 🧩 `drivable_head.py` — Tiny BEV Segmentation Head

This head takes your fused BEV feature map `(B, C, H, W)` and predicts a **binary drivable‑area mask** at the same resolution.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrivableAreaHead(nn.Module):
    """
    Lightweight drivable-area segmentation head.
    Input:  (B, C, H, W)
    Output: (B, 1, H, W)  sigmoid mask
    """

    def __init__(self, in_channels=128, mid_channels=64):
        super().__init__()

        # First refinement block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Second refinement block
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Final 1×1 conv → 1 channel mask
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, bev):
        """
        bev: (B, C, H, W)
        """
        x = self.block1(bev)
        x = self.block2(x)
        logits = self.out_conv(x)
        mask = torch.sigmoid(logits)
        return mask
```

---

# 🧠 Why this design works so well

### ✔️ BEV‑native
No upsampling or downsampling — it predicts directly in BEV space.

### ✔️ Lightweight
Only a few conv layers, perfect for your RTX 2050.

### ✔️ Modular
You can attach it after:

```
backbone → fusion → DrivableAreaHead
```

without touching any other part of the pipeline.

### ✔️ Easy to extend
Later you can add:

- multi‑scale BEV features
- skip connections
- dilated convolutions
- auxiliary losses

But this minimal version is already strong enough for A2D2‑style drivable area.

---

# 🧪 Quick sanity test

```python
if __name__ == "__main__":
    head = DrivableAreaHead(in_channels=128)
    bev = torch.randn(1, 128, 100, 100)
    mask = head(bev)
    print(mask.shape)  # (1, 1, 100, 100)
```

---

# 🧭 Next step options

You can choose what to build next:

### **A. Tests for the drivable‑area head**
Shape, gradient, determinism, device consistency.

### **B. A tiny 3D detection head**
CenterPoint‑style, BEV‑native.

### **C. A unified `MultiTaskHead` wrapper**
Returns a dict like:

```python
{"drivable": mask, "detection": {...}}
```

### **D. End‑to‑end pipeline wiring**
Voxelizer → PFN → BEV backbone → fusion → drivable head.

Just tell me which direction you want to go.