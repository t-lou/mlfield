# **mlfield**

My private playground and notes collection for ML.

---

## 🚀 **Overview**

**mlfield** is a modular repository designed for:

- Fast iteration on ML experiments
- Clean separation of ideas into themed subfolders
- Reproducible development environments
- Notes, insights, and research logs that accompany experiments
- Future expansion into multimodal, SSL, and robotics‑related workflows

It’s intentionally lightweight and flexible — a “field notebook” for machine learning.

---

## 📁 **Repository Structure**

```
mlfield/
│
├── .devcontainer/      # Reproducible development environment (incl. VS Code devcontainer, direct init not supported)
│
├── components/         # Reusable layers, functions and models
│
├── experiments/        # The trainings and evaluations
│
├── notes/              # General research notes, logs, design sketches, and conceptual writeups
│
├── tools/              # Standalone utilities for editing checkpoints, for controlling WSL2 etc
│
├── _to_clarify/        # Legacy before restructering, which needs to be deleted or continued
│
└── (more coming...)
```

Each folder is self‑contained and may include scripts, notebooks, configs, and experiment logs.

---

## 🛠️ **Getting Started**

### **Clone the repository**
```bash
git clone https://github.com/t-lou/mlfield
bash mlfield/.devcontainer/launch.sh
```

You can either launch the container with launch.sh script or open it with VS Code extension "Dev Containers".

If you are cautious about the base docker image, you can also create the base image from scatch and rename it with "docker tag...".

```bash
bash .devcontainer/create_base_container.sh  # create mlfield_cuda_base:latest
docker tag mlfield_cuda_base:latest tlou/mlfield_cuda_base:latest
```

If you already have everything installed (mainly pytorch), then just load the env var to make import work:

```bash
source .envrc
```

Then run either with `python3 -m FOLDER1.FOLDER2.PART` or `runpy FOLDER1/FOLDER2/PART.py`.

---

### 📁 Optional Dataset Mount (`DATASET_DIR` via `local.env`)

You can mount a local dataset into the container by defining `DATASET_DIR` in your `.devcontainer/local.env` file:

```
DATASET_DIR=/path/to/your/dataset
```

The compose file uses:

```
- ${DATASET_DIR:-/dev/null}:/mnt/dataset:ro
```

If `DATASET_DIR` is set in `.devcontainer/local.env`, that directory is mounted read‑only at `/mnt/dataset`.
If it’s missing, the mount safely falls back to `/dev/null`.

## Dataset Usage Policy

This repository contains multiple experiments using different datasets
(e.g., Open Images, ImageNet, KITTI, audio datasets). No dataset files
are included in this repository.

Each dataset has its own license and usage terms. Users must download
datasets from their official sources and comply with their licenses.

Dataset-specific notes are provided in each experiment folder.

### Citations about dataset

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
