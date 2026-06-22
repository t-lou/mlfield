Here’s a focused “mini‑survey” of **interesting 2023–2026 point‑cloud transformer work**, grouped by role in a modern 3D stack. I’ll keep each paper to a tight, opinionated one‑liner so you can decide what to dig into.

---

### 1. Backbone evolution: scalable point-cloud transformers

- **Point Transformer V3 (PTv3), CVPR 2024 (Wu et al.)**   [arXiv.org](https://arxiv.org/abs/2312.10035)  [arXiv.org](https://arxiv.org/html/2312.10035v1)  
  **Role:** de‑facto *scalable* 3D transformer backbone.  
  **Key idea:** stop obsessing over attention variants—**serialize points via space‑filling curves**, form large patches, and run simple self‑attention; replaces per‑layer KNN with one‑shot serialization, expanding receptive field from 16→1024 points while being ~3× faster and ~10× more memory‑efficient than PTv2, and hitting SOTA on >20 indoor/outdoor tasks.

- **PTv3 ecosystem: PPT / PTv3‑Extreme (2024–2025)**   [Github](https://github.com/Pointcept/PointTransformerV3)  [Emergent Mind](https://www.emergentmind.com/topics/point-transformer-v3-ptv3)  
  **Role:** adaptations of PTv3 for large‑scale segmentation and multi‑dataset transfer.  
  **Key idea:** **Point Prompt Training (PPT)** treats datasets as “domains” and learns prompts over a shared PTv3 backbone; **PTv3‑EX** pushes resolution and receptive field further for huge scenes.

- **LitePT (2025, in Pointcept)**   [Github](https://github.com/Pointcept/Pointcept)  
  **Role:** lightweight PT‑style backbone.  
  **Key idea:** distills PTv3 design into a smaller, mobile‑friendlier architecture while keeping the serialize‑and‑patch philosophy.

---

### 2. Large‑scale 3D representation learning & SSL on point transformers

- **Sonata: Self‑Supervised Learning of Reliable Point Representations, CVPR 2025 Highlight**   [Github](https://github.com/Pointcept/Pointcept)  
  **Role:** strong SSL recipe on top of PTv3.  
  **Key idea:** uses PTv3 as backbone and designs **consistency‑focused objectives** (e.g., robust to density, occlusion, viewpoint) to pretrain on large, mixed 3D data; becomes a general “drop‑in” initialization for many 3D tasks.

- **Towards Large‑scale 3D Representation Learning with Multi‑dataset Point Prompt Training, CVPR 2024**   [Github](https://github.com/Pointcept/PointTransformerV3)  [Github](https://github.com/Pointcept/Pointcept)  
  **Role:** PTv3‑based large‑scale pretraining.  
  **Key idea:** **prompt‑style conditioning** on dataset/domain IDs so a single PTv3 can be trained jointly on many 3D datasets without catastrophic interference, improving transfer and zero‑shot behavior.

- **Utonia: Toward One Encoder for All Point Clouds, ICML 2026 (announced)**   [Github](https://github.com/Pointcept/Pointcept)  
  **Role:** “one encoder” vision for point clouds.  
  **Key idea (from project description):** unify indoor, outdoor, CAD, etc. under a single transformer encoder with appropriate pretraining and normalization across wildly different point distributions.

- **Concerto: Joint 2D–3D Self‑Supervised Learning Emerges Spatial Representations, NeurIPS 2025**   [Github](https://github.com/Pointcept/Pointcept)  
  **Role:** 2D–3D joint SSL with transformers.  
  **Key idea:** co‑train 2D and 3D backbones (transformers on images + point clouds) with shared objectives so that **3D gains from 2D data scale** and vice versa.

---

### 3. Scene‑scale and efficiency‑driven transformers (2023+)

*(Some of these start in 2022 but are very relevant in the 2023–2026 landscape.)*

- **Stratified Transformer / successors (2022→2023)**  
  **Role:** strong indoor scene backbone.  
  **Key idea:** **stratified sampling + multi‑scale local attention** over sparse 3D points; very competitive on ScanNet/S3DIS and often used as a baseline against PTv3‑style models.

- **OA‑CNNs: Omni‑Adaptive Sparse CNNs for 3D Semantic Segmentation, CVPR 2024**   [Github](https://github.com/Pointcept/Pointcept)  
  **Role:** hybrid sparse CNN + transformer‑style adaptivity.  
  **Key idea:** not a pure transformer, but introduces **omni‑adaptive kernels** that play a similar role to attention in adapting receptive fields to local geometry—often paired with transformer backbones in the same codebase.

---

### 4. Multimodal & autonomous‑driving‑oriented transformers

*(Here I’m synthesizing from the general trend rather than one single canonical paper.)*

- **LiDAR–camera fusion transformers (2023–2026)**  
  **Role:** BEV detection, segmentation, tracking.  
  **Key idea:** use **cross‑attention** between point‑cloud tokens (often PTv3/Stratified‑style) and image/BEV tokens; attention naturally handles long‑range correspondences and occlusions. Many autonomous‑driving stacks now use some variant of this for 3D detection.

- **Multi‑dataset / multi‑sensor pretraining (PPT, Concerto, Sonata)**   [Github](https://github.com/Pointcept/PointTransformerV3)  [Github](https://github.com/Pointcept/Pointcept)  
  **Role:** make 3D transformers robust across sensors and domains.  
  **Key idea:** treat dataset/sensor as a conditioning signal (prompts, adapters, or modality tokens) and pretrain a **single transformer backbone** across LiDAR, RGB‑D, CAD, etc.

---

### 5. How I’d read this if I were you

If you want a **practical, modern grasp** of point‑cloud transformers (2023–2026), I’d do:

1. **Backbone:** read PTv3 paper + skim code in Pointcept.   [arXiv.org](https://arxiv.org/abs/2312.10035)  [Github](https://github.com/Pointcept/PointTransformerV3)  
2. **SSL:** read Sonata + PPT sections in the Pointcept repo/papers.   [Github](https://github.com/Pointcept/Pointcept)  [Github](https://github.com/Pointcept/PointTransformerV3)  
3. **Compare:** mentally contrast PTv3 with older Point Transformer v1/v2 and Stratified Transformer to see how serialization vs KNN vs stratified sampling trade off.   [arXiv.org](https://arxiv.org/html/2312.10035v1)  
4. **Fusion:** look at one recent LiDAR–camera fusion transformer paper and map how they plug a PT‑style backbone into BEV detection.

If you tell me your main use case—e.g., **autonomous driving LiDAR**, **indoor mapping**, or **general 3D pretraining**—I can turn this into a very concrete reading + implementation roadmap.
