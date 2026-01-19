Nice, this is a good moment to cleanly separate “what the model can do” from “which modalities you actually use”.

Here’s a simple, explicit way to make it configurable:

- one flag: `modality in {"lidar", "camera", "both"}`
- only build the encoders you need
- route the forward pass accordingly
- keep the detection heads identical (always operate on a BEV feature map)

---

### 1. Make the model modality‑aware

```python
class SimpleModel(nn.Module):
    def __init__(self, bev_channels: int = 128, modality: str = "both") -> None:
        super().__init__()
        assert modality in {"lidar", "camera", "both"}
        self.modality = modality

        # 1. Lidar encoder
        if modality in {"lidar", "both"}:
            self.lidar_encoder = PointPillarBEV()  # (B, C, H, W)
        else:
            self.lidar_encoder = None

        # 2. Camera encoder
        if modality in {"camera", "both"}:
            self.cam_encoder = TinyCameraEncoder()  # (B, N_cam, C)
        else:
            self.cam_encoder = None

        # 3. Fusion block (only needed when using both)
        if modality == "both":
            self.fusion = FuTrFusionBlock()
        else:
            self.fusion = None

        # 4. Optional: camera‑only BEV projector
        if modality == "camera":
            # map camera tokens → BEV feature map
            self.cam_bev_projector = CameraBEVProjector(bev_channels=bev_channels)
        else:
            self.cam_bev_projector = None

        # 5. Detection heads (shared)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, 1, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, 6, kernel_size=1),
        )
```

You can define `CameraBEVProjector` however you like (e.g., learned BEV queries + cross‑attention, or a simple MLP+reshape if you already know the BEV grid size).

---

### 2. Route the forward pass by modality

```python
    def forward(self, points: Tensor | None, images: Tensor | None) -> dict:
        # Lidar path
        if self.modality in {"lidar", "both"}:
            assert points is not None, "points must be provided for lidar or both"
            bev_lidar: Tensor = self.lidar_encoder(points)  # (B, C, H, W)
        else:
            bev_lidar = None

        # Camera path
        if self.modality in {"camera", "both"}:
            assert images is not None, "images must be provided for camera or both"
            images = rescale_image(images)
            cam_tokens: Tensor = self.cam_encoder(images)  # (B, N_cam, C)
        else:
            cam_tokens = None

        # Fusion / selection
        if self.modality == "both":
            bev_fused: Tensor = self.fusion(bev_lidar, cam_tokens)  # (B, C, H, W)
        elif self.modality == "lidar":
            bev_fused = bev_lidar
        elif self.modality == "camera":
            bev_fused = self.cam_bev_projector(cam_tokens)  # (B, C, H, W)
        else:
            raise RuntimeError("Unknown modality")

        # Heads
        heatmap = torch.sigmoid(self.heatmap_head(bev_fused))
        reg = self.reg_head(bev_fused)

        return {"heatmap": heatmap, "reg": reg}
```

---

### 3. How you’d actually use it

```python
# lidar‑only
model = SimpleModel(bev_channels=128, modality="lidar")

# camera‑only
model = SimpleModel(bev_channels=128, modality="camera")

# both
model = SimpleModel(bev_channels=128, modality="both")
```

The only extra piece you need to design is `CameraBEVProjector` (camera‑only path). For a first version, you can even start with lidar‑only + both, and leave camera‑only as a TODO until you decide how you want to lift camera tokens into BEV.