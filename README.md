# **mlfield**

My private playground and notes collection for ML.

---

## 🚀 **Overview**

**mlfield** is a modular, experimental ML repository designed for:

- **Fast iteration** on diverse ML experiments and research ideas
- **Modular structure** with clean separation of components, experiments, and utilities
- **Reproducible environments** via containerized dev setup (Docker + VS Code devcontainers)
- **Rich documentation** with research notes, logs, and conceptual writeups
- **Multiple ML domains** including autonomous driving, self‑supervised learning (SSL), vision transformers, and point cloud processing

It's intentionally lightweight and flexible—a notebook for iterating on ML research.

---

## 📁 **Repository Structure**

```
mlfield/
│
├── .devcontainer/      # Docker setup & VS Code devcontainer config
│
├── components/         # Reusable ML building blocks
│   ├── dataset/        # Dataset loaders (A2D2, COCO, image-only)
│   ├── definitions/    # Config classes and hyperparameter definitions
│   ├── mmperc/         # Multi-modal perception model (backbone, encoder, decoder, losses)
│   ├── utils/          # Common utilities (calibration, BEV, logging, config)
│   └── vit/            # Vision transformer models (DINO, iJEPA, MAE)
│
├── experiments/        # Active experiments and training scripts
│   ├── image_dino/     # DINO vision transformer experiments
│   ├── image_jepa/     # iJEPA self-supervised learning
│   ├── image_mae/      # Masked autoencoder experiments
│   └── mmperc/         # Multi-modal perception training
│
├── notes/              # Research notes, design sketches, learnings
│   ├── ssl/            # Self-supervised learning notes
│   ├── mtl/            # Multi-task learning insights
│   └── (domain-specific folders)
│
├── tools/              # Standalone CLI utilities
│   ├── gpu_monitor.py
│   ├── clean_every_n.py
│   └── (checkpoint & environment helpers)
│
└── _to_clarify/        # Legacy code pending cleanup or migration
```

Each folder is self‑contained with its own scripts, notebooks, configs, and logs.

---

## 🛠️ **Getting Started**

### **Option 1: Docker Container (Recommended)**

```bash
git clone https://github.com/t-lou/mlfield
cd mlfield
bash .devcontainer/launch.sh
```

Alternatively, open with VS Code's "Dev Containers" extension for a seamless IDE experience.

**Custom base image (optional):**
If you prefer to build the CUDA base image from scratch:

```bash
bash .devcontainer/create_base_container.sh  # creates mlfield_cuda_base:latest
docker tag mlfield_cuda_base:latest tlou/mlfield_cuda_base:latest
```

### **Option 2: Local Environment**

If PyTorch and dependencies are already installed locally:

```bash
cd mlfield
source .envrc  # Load environment variables for import paths
```

### **Running Experiments**

Use one of these patterns:

```bash
# As a Python module (preferred for package organization)
python3 -m experiments.image_dino.dino

# Or with the runpy helper
runpy experiments/image_dino/dino.py
```

### ONNX Encoder Exports

Sonata reads its model, optimizer schedule, data, checkpoint, and export settings from
[`experiments/ssl/sonata_config.yml`](experiments/ssl/sonata_config.yml). Pass another YAML
file with `--path-config` to reload a different configuration for the run. Use `--small-gpu`
to apply the compact model profile and cap each cloud at 1024 points. Configure all other
runtime values in the YAML file.

```bash
python -m experiments.ssl.sonata --path-config experiments/ssl/sonata_config.yml
```

After training finishes, the image DINO, I-JEPA, and MAE scripts export their inference encoders to the configured
checkpoint directory (`image_dino_teacher_backbone.onnx`, `image_jepa_target_encoder.onnx`, and
`image_mae_encoder.onnx`). Install `onnx` in the active environment to enable these exports. The files accept
preprocessed NCHW images at the model's configured resolution and support a dynamic batch size.

Training saves the teacher encoder state dict, full teacher state dict, and both ONNX exports
using the configured paths. To export all four artifacts from an intermediate training
checkpoint, run the standalone exporter from the project root:

```bash
python -m experiments.ssl.export_sonata checkpoints/sonata/step_0000100.pt
```

By default, the outputs go into `checkpoints/sonata/exports`. The exporter produces
`sonata_teacher_backbone.pt` / `.onnx` (encoder only) and `sonata_full.pt` / `.onnx` (encoder
plus both prototype heads). New training checkpoints include the exact model configuration.
For older checkpoints, pass the matching YAML config with `--config`; add `--small-gpu` if that
checkpoint was trained with the compact profile. Use `--output-dir` and `--num-points` to
override the output location and fixed ONNX input point count.

Both ONNX exports accept `features`, `grid_coord`, and `batch`; the encoder returns
`encoded_features`, while the full export also returns `mask_logits` and `unmask_logits`.
Exports use a fixed point count and serialization grid; re-export if geometry, channels, or
model configuration change. PTv3 uses an ONNX-compatible neighborhood fallback during export.

### Visualizing Sonata Features

Use the visualizer beside the Sonata training script to color a PCD/PLY cloud by its PTv3
features. It accepts a `.pt` checkpoint or either Sonata ONNX export; by default, feature
channels are reduced to PCA RGB colors. `--component N` displays one scalar channel instead.

```bash
python -m experiments.ssl.vis_sonata sonata_teacher_backbone.pt cloud.pcd
python -m experiments.ssl.vis_sonata sonata_teacher_backbone.onnx cloud.ply --screenshot features.png
python -m experiments.ssl.vis_sonata sonata_full.onnx cloud.pcd --output unmask_logits
```

Fixed-count ONNX exports determine their input point count automatically; dynamic-count models
require `--num-points`. Clouds are deterministically sampled or repeated to match the count.
Because the current ONNX export is traced on a synthetic cubic
grid, the visualizer maps points to that lattice in spatial Z-order. Use the `.pt` checkpoint
path when preserving metric-coordinate geometry is important. Checkpoint inference uses up to 4096
points by default (`--num-points` changes the cap); nonstandard checkpoints need their training
config via `--cfg`. Sonata training and visualization use the shared `BevParams` ROI and
anisotropic voxel size by default, filtering out-of-ROI points and aggregating each voxel.
`--grid-size` overrides this with an isotropic voxel size. Checkpoints created before this
preprocessing change may require their original geometry/configuration for faithful results.
Install `pyvista` for visualization and `onnxruntime` to run ONNX models.

---

### 📁 Dataset Configuration

**Mount local datasets into the container:**

Create `.devcontainer/local.env` with:

```
DATASET_DIR=/path/to/your/dataset
```

The compose file mounts this read-only at `/mnt/dataset`, or safely falls back to `/dev/null` if unset.

---

## 📋 Datasets & Licensing

This repository includes experiments with multiple datasets (e.g., Open Images, ImageNet, KITTI, BDD100K, A2D2). **No dataset files are included**—you must download them from official sources and comply with their respective licenses.

Dataset-specific setup instructions and notes are provided in each experiment folder.

### **Dataset Citations**

```
@InProceedings{bdd100k,
  author    = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi
               and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht
               and Darrell, Trevor},
  title     = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2020}
}
```

```
@article{geyer2020a2d2,
  title   = {{A2D2}: {Audi Autonomous Driving Dataset}},
  author  = {Geyer, Jakob and Kassahun, Yohannes and Mahmudi, Mentar and
             Ricou, Xavier and Durgesh, Rupesh and Chung, Andrew S. and
             Hauswald, Lorenz and Pham, Viet Hoang and M{\"u}hlegg, Maximilian and
             Dorn, Sebastian and Fernandez, Tiffany and J{\"a}nicke, Martin and
             Mirashi, Sudesh and Savani, Chiragkumar and Sturm, Martin and
             Vorobiov, Oleksandr and Oelker, Martin and Garreis, Sebastian and
             Schuberth, Peter},
  journal = {arXiv preprint arXiv:2004.06320},
  year    = {2020}
}
```

```
@article{openimages,
  title={The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
  author={Kuznetsova, Alina and Rom, Hassan and Alldrin, Neil and Uijlings, Jasper and Krasin, Ivan and Pont-Tuset, Jordi and Kamali, Shahab and Popov, Stefan and Malloci, Matteo and Kolesnikov, Alexander and Duerig, Tom and Ferrari, Vittorio},
  journal={International Journal of Computer Vision (IJCV)},
  year={2020}
}
```
