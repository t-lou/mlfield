# **mmperc — Modular Multimodal Perception Research Framework**

`mmperc` is a lightweight, flexible research framework for experimenting with **multimodal perception**, **BEV detection**, and **multi‑task learning**.  
It is designed for rapid prototyping, clean modularity, and easy switching between:

- **Single‑modal → Multi‑task**  
- **Multi‑modal → Single‑task**  
- **Multi‑modal → Multi‑task**  
- **Ablation‑friendly component swapping**

The project is intentionally simple, hackable, and built for iterative experimentation.

---

## 🌐 **Project Goals**

- Provide a **minimal but extensible** baseline for multimodal BEV perception.
- Allow **plug‑and‑play encoders**, fusion blocks, and task heads.
- Support **LiDAR**, **RGB**, and future modalities (radar, depth, semantics).
- Enable **multi‑task learning** (detection, segmentation, depth, flow, etc.).
- Keep the codebase **clean, modular, and easy to debug**.

---

## 🧱 **Core Architecture**

The current reference model is `SimpleModel`, a compact multimodal BEV detector:

### **1. LiDAR Encoder → BEV Feature Map**
- Converts raw point clouds `(B, N, 4)` into a BEV tensor `(B, C, H, W)`
- Default: `PointPillarBEV`
- Replaceable with:
  - VoxelNet
  - SparseConv BEV encoders
  - Lift‑splat‑shoot style encoders

### **2. Camera Encoder → Token Embeddings**
- Converts RGB images into a sequence of tokens `(B, N_cam, C)`
- Default: `TinyCameraEncoder`
- Replaceable with:
  - ResNet / EfficientNet backbones
  - ViT / Swin Transformer
  - Multi‑camera encoders

### **3. Fusion Block**
- Cross‑attention between BEV features and camera tokens
- Default: `FuTrFusionBlock`
- Replaceable with:
  - Deformable attention
  - BEVFormer‑style temporal fusion
  - Simple concatenation or FiLM conditioning

### **4. Task Heads**
Currently implemented:

| Task | Output | Head |
|------|--------|------|
| BEV object heatmap | `(B, 1, H, W)` | CenterNet‑style CNN |
| BEV regression | `(B, 6, H, W)` | dx, dy, log(w), log(l), sin(yaw), cos(yaw) |
| Semantic segmentation | `(B, C_sem, H_img, W_img)` | FullResSemHead |

Each head is modular and can be swapped or disabled.

---

## 🔧 **Model Flexibility**

The framework is designed so you can easily configure:

### **Modality selection**
- LiDAR‑only
- Camera‑only
- LiDAR + Camera
- Future: Radar, Depth, Semantics

### **Task selection**
- Detection only
- Segmentation only
- Detection + Segmentation
- Add your own tasks (depth, flow, occupancy)

### **Fusion strategies**
- No fusion (single‑modal)
- Early fusion
- Cross‑attention fusion
- Late fusion

### **Encoder choices**
- Swap LiDAR encoders
- Swap camera backbones
- Change BEV resolution or channels

---

## 📦 **Repository Structure**

```
mmperc/
│
├── src/
│   ├── common/          # utilities, config, shared helpers
│   ├── datasets/        # dataset loaders, converters, preprocessing
│   ├── encoder/         # LiDAR & camera encoders
│   ├── fusion/          # fusion modules (cross-attn, FiLM, etc.)
│   ├── heads/           # detection, segmentation, regression heads
│   ├── models/          # SimpleModel and future variants
│   └── training/        # loops, losses, metrics
│
├── tests/               # pytest-based unit tests
└── README.md
```

---

## 🚀 **Quick Start with A2D2 Dataset**

### **1. Build the development container**

### **2. Prepare dataset**

Mount your dataset via `.devcontainer/local.env`:

```
DATASET_DIR=/mnt/d/data
```

Then convert A2D2:

```
python3 -m src.datasets.a2d2_tar_dataset_converter \
    /mnt/dataset/camera_lidar_semantic_bboxes.tar \
    data/a2d2
```

### **3. Train**

```
python3 -m src.train
```
