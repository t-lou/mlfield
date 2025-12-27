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
├── hugging_face/       # Experiments using Hugging Face models, datasets, and pipelines
│
├── notes/              # Research notes, logs, design sketches, and conceptual writeups
│
├── ssl/                # Self-&Semi-supervised learning experiments and prototypes
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

After creation with bash, optionally reopen the folder with VS Code inside a container. Please reopen only, rebuilding will break.
