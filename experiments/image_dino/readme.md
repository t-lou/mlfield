# Image DINO Experiment

## 🧠 Introduction
This experiment implements **DINOv1 (self‑distillation with no labels)** from scratch to explore how self‑supervised vision transformers learn visual features.  
The goal is to understand the internal mechanisms of **feature emergence** and **representation learning** through self‑distillation, and to evaluate how different datasets influence the learned embeddings.

The visualization examples illustrate how the model interprets diverse scenes — from structured environments to natural and human‑made objects — and how feature maps evolve during training.

| **Open Images Dataset** | Apache 2.0 | Used for visualization examples (e.g., parking lot, lanterns, food, soldier) | Open Images [(storage.googleapis.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fstorage.googleapis.com%2Fopenimages%2Fweb%2Findex.html") |
| **Kaggle COCO Variant** | CC BY‑SA 4.0 | Used for additional training and evaluation | [Kaggle COCO](https://creativecommons.org/licenses/by-sa/4.0/) |
| **ImageNet‑256 (Kaggle)** | CC0 1.0 | Used for main training and feature extraction | ImageNet‑256 [(kaggle.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fdimensi0n%2Fimagenet-256") |

No dataset files are included in this repository.  
Users must download datasets from their official sources and comply with their respective licenses.

---

## ⚖️ Copyright and License Statement

All **images** used for visualization are sourced from datasets with explicit redistribution rights:
- Open Images (Apache 2.0)
- Kaggle COCO (CC BY‑SA 4.0)
- ImageNet‑256 (CC0 1.0)

No copyrighted is used or redistributed.
