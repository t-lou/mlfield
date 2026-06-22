You’re basically asking: *“ViT changed 2D—what’s the transformer story for 3D point clouds?”* Let’s sketch the lineage and then list the models very briefly so you can zoom in later.

---

### Big-picture history

- **Pre-transformer era (~2017–2019)**  
  - **PointNet / PointNet++**: MLP + pooling, then hierarchical local neighborhoods—no attention, but set the template for point-based processing.  
  - **PointCNN, KPConv, sparse 3D convs**: convolutional views on point clouds and voxels.

- **Early transformer ideas in 3D (~2019–2020)**  
  - Attention mostly appears as **local refinement blocks** inside PointNet++/conv backbones rather than full ViT-style global token mixers.  
  - Work focuses on: how to define neighborhoods, positional encodings in \(\mathbb{R}^3\), and complexity on large point sets.   [arXiv.org](https://arxiv.org/pdf/2205.07417)  

- **Transformer becomes a first-class 3D backbone (~2020–2022)**  
  - Dedicated **point-cloud transformers** emerge: global or hierarchical self-attention over points/patches.  
  - Tasks: classification, part segmentation, semantic segmentation, detection.  
  - Multiple taxonomies (implementation-, representation-, task-based) are summarized in Lu et al., *Transformers in 3D Point Clouds: A Survey*.   [arXiv.org](https://arxiv.org/abs/2205.07417)  

- **Modern 3D transformers (~2022–2024)**  
  - Strong focus on **efficiency** (sparse attention, stratified sampling, voxel/patch tokens) and **pretraining** (masked point modeling, contrastive).  
  - Transformers are now competitive or superior to conv/PointNet++ baselines on ShapeNet, ModelNet, ScanNet, S3DIS, Waymo/LiDAR benchmarks.   [IEEE Xplore](https://ieeexplore.ieee.org/document/9857927)  [IEEE Xplore](https://ieeexplore.ieee.org/document/9393615)  

If you want, we can later go into “why attention is actually nice for LiDAR/autonomous driving” (long-range context, multi-sensor fusion, etc.).

---

### Key families of transformer-based point cloud models (very brief)

I’ll keep each to a one-liner so you can pick what to dive into.

#### Early / core point-cloud transformers

- **Point Transformer (Zhao et al.)**  
  **Idea:** Local self-attention with learned positional encodings over point neighborhoods; drop-in replacement for PointNet++ set abstraction.   [IEEE Xplore](https://ieeexplore.ieee.org/document/9393615)  

- **PCT – Point Cloud Transformer**  
  **Idea:** Global transformer over points with offset-attention and neighborhood aggregation; mainly for classification/segmentation.

- **Point Cloud Transformer variants (generic)**  
  **Idea:** Many works tweak neighborhood construction, positional encoding, and attention sparsity to handle irregular, unordered points.   [Springer](https://link.springer.com/article/10.1007/s11760-025-04116-5)  

#### Hierarchical / voxel / patch-based transformers

- **Voxel/patch transformers for 3D**  
  **Idea:** Convert point clouds to voxels or local patches, treat each as a token, then run ViT-style attention—better scalability to large scenes.   [IEEE Xplore](https://ieeexplore.ieee.org/document/9857927)  

- **Stratified / multi-scale transformers**  
  **Idea:** Stratified sampling (fine + coarse levels) with attention across scales; good for large indoor scenes (ScanNet, S3DIS).

#### Pretraining and masked modeling on point clouds

- **Point-BERT**  
  **Idea:** BERT-style masked token prediction on point patches; learns a discrete “codebook” of local geometry and then fine-tunes for downstream tasks.

- **Point-MAE**  
  **Idea:** MAE-style masked autoencoding on point clouds; reconstruct masked regions from visible tokens, similar spirit to ViT MAE but in 3D.

- **Other masked/contrastive 3D transformers**  
  **Idea:** Mix masked reconstruction, contrastive learning, and multi-view augmentation to build strong 3D backbones for classification/segmentation/detection.

#### Scene-level and LiDAR-focused transformers

- **Transformer backbones for LiDAR segmentation/detection**  
  **Idea:** Replace or augment sparse 3D convs with attention blocks; often hybrid (conv + transformer) for efficiency on large point sets.   [IEEE Xplore](https://ieeexplore.ieee.org/document/9393615)  

- **Fusion transformers (multi-modal 3D)**  
  **Idea:** Use attention to fuse point clouds with images, radar, or BEV features—very relevant for autonomous driving pipelines.   [Springer](https://link.springer.com/article/10.1007/s11760-025-04116-5)  

---

### Surveys you might like

- **Transformers in 3D Point Clouds: A Survey (Lu et al., 2022)**  
  **Why useful:** Gives taxonomies (implementation, representation, task), compares many methods, and discusses self-attention variants in 3D.   [arXiv.org](https://arxiv.org/abs/2205.07417)  [arXiv.org](https://arxiv.org/pdf/2205.07417)  

- **Updated survey on transformers for point cloud processing (IEEE)**  
  **Why useful:** More recent view, with emphasis on depth sensors, 3D scanners, and practical applications.   [IEEE Xplore](https://ieeexplore.ieee.org/document/9857927)  

---

You know what—rather than me dumping everything, which of these buckets is most interesting for you right now?

- **(A)** Point-level architectures (Point Transformer / PCT)  
- **(B)** Masked pretraining (Point-BERT / Point-MAE)  
- **(C)** LiDAR/autonomous driving backbones and fusion transformers  
- **(D)** Efficiency tricks (stratified, voxel/patch tokens, sparse attention)

Pick one and I’ll go deep, including equations and implementation details.

# Point Transformers

**Short answer:**  
**Point Transformer (v1)** introduces *vector attention* over local point neighborhoods with learned relative positional encoding, giving a permutation‑invariant, geometry‑aware transformer for point clouds. Its closest improvements are **Point Transformer v2** (grouped vector attention + partition pooling) and **Point Transformer v3** (scaling‑focused, simpler, faster, larger receptive field).  
Below is a clean, citation‑grounded, structured explanation.

---

# 🧩 Point Transformer (v1) — Original Paper (Engel et al.)  




### 1. Core idea  
Point Transformer v1 is a transformer **designed directly for unordered 3D point sets**, using attention to aggregate local neighborhoods instead of PointNet++‑style pooling. It introduces:

- **Local–global attention**: attention over spatial neighborhoods to capture geometric relations.  
- **Vector attention**: attention weights computed from *relative positional encodings* between points.  
- **SortNet**: a learned permutation‑invariant point selection module.  
- **Permutation invariance**: the architecture ensures invariance globally while allowing local flexibility.  

These details come from the original repository and abstract.   [Github](https://github.com/engelnico/point-transformer)

---

### 2. Architecture components (v1)

- **Neighborhood construction**: k‑NN or radius neighborhoods.  
- **Relative positional encoding**: encodes \(\Delta x = x_j - x_i\) to inject geometry.  
- **Vector attention**:  
  \[
  \text{Attn}(i,j) = f_\theta(x_j - x_i) \cdot g_\phi(h_j)
  \]
  where \(f_\theta\) learns geometric relations and \(g_\phi\) transforms features.  
- **Local–global fusion**: local attention + global feature mixing.  
- **SortNet**: selects representative points for invariance and downsampling.  

The model is evaluated on ModelNet40 and ShapeNet part segmentation, showing competitive results.   [Github](https://github.com/engelnico/point-transformer)

---

# 🔄 Close Improvements

## 1. **Point Transformer v2 (PTv2)** — Grouped Vector Attention  




PTv2 introduces two major improvements:

### (a) **Grouped Vector Attention**  
- Splits attention into groups for efficiency.  
- Improves expressiveness while reducing compute.  

### (b) **Partition‑based pooling**  
- Replaces SortNet with a more efficient partitioning strategy.  

PTv2 is referenced in PTv3’s documentation as the predecessor.   [Github](https://github.com/Pointcept/PointTransformerV3)

---

## 2. **Point Transformer v3 (PTv3)** — Simpler, Faster, Stronger (CVPR 2024 Oral)  




PTv3 is the most important successor. It is **not** focused on inventing new attention mechanisms; instead, it focuses on **scaling**.

### Key innovations  
- **Serialized neighbor mapping** replaces expensive KNN search.  
- **Massively expanded receptive field**:  
  - From **16 → 1024 points** (64× increase).  
- **3× faster** inference and **10× lower memory** than PTv2.  
- **Simplified design**: removes complex positional encodings and uses sparse conv layers.  
- **State‑of‑the‑art** on >20 indoor/outdoor tasks.  

These details are from the PTv3 paper and summary.   [arXiv.org](https://arxiv.org/html/2312.10035v2)  [arXiv.org](https://arxiv.org/abs/2312.10035)  [lrjconan.github.io](https://lrjconan.github.io/UBC-EECE571F-DL-Structures/assets/slides_2025/paper_02_point_transformer.pdf)

---

# 🧱 Summary Table

| **Model** | **Key Idea** | **Main Improvements** | **Source** |
|----------|--------------|------------------------|------------|
| **Point Transformer v1** | Vector attention + relative geometry + SortNet | First transformer fully designed for point sets |   [Github](https://github.com/engelnico/point-transformer) |
| **Point Transformer v2** | Grouped vector attention + partition pooling | More efficient, more expressive |   [Github](https://github.com/Pointcept/PointTransformerV3) |
| **Point Transformer v3** | Scaling‑focused, simplified architecture | 3× faster, 10× memory savings, 1024‑point receptive field |   [arXiv.org](https://arxiv.org/html/2312.10035v2)  [arXiv.org](https://arxiv.org/abs/2312.10035)  [lrjconan.github.io](https://lrjconan.github.io/UBC-EECE571F-DL-Structures/assets/slides_2025/paper_02_point_transformer.pdf) |

---

# Want deeper detail?

Choose what you want next:

- **Vector attention math**  
- **PTv2 grouped vector attention**  
- **PTv3 scaling principle**  
- **Comparison with PointNet++**

# PCT

Below is a **clean, structured, technically accurate explanation of PCT (Point Cloud Transformer)** based strictly on the **original paper** and the **closest improvements** that followed it.  
I’ll keep it compact but deep, and you can branch into any part via the Guided Links.

---

# 🧩 What PCT (Point Cloud Transformer) Actually Introduced  




**Paper:** *“PCT: Point Cloud Transformer”* (Guo et al., 2021)

**Core motivation:**  
PointNet++‑style local MLPs + pooling were too limited in capturing **global shape context**.  
PCT proposes a **pure transformer backbone** for point clouds with two key innovations:

---

## 1. **Offset-Attention (OA)** — the heart of PCT  
This is the main contribution.

Instead of standard self-attention, PCT uses:

\[
\text{OA}(x_i) = x_i + \sum_{j \in \mathcal{N}(i)} \alpha_{ij}(x_j - x_i)
\]

Where:

- \(x_j - x_i\) is the **offset** (relative feature difference)  
- \(\alpha_{ij}\) is attention weight  
- The residual structure preserves stability  
- The offset term injects **local geometric variation** directly into attention  

**Why it matters:**  
It makes attention **geometry-aware** without heavy positional encodings.

---

## 2. **Neighbor Embedding**  
Before attention, PCT applies a **lightweight local embedding**:

- kNN grouping  
- Shared MLP  
- Feature normalization  

This replaces PointNet++’s Set Abstraction with a simpler, transformer-friendly block.

---

## 3. **Multi-Headed OA + Global Context**  
PCT stacks **four OA layers**, each with:

- Multi-head offset attention  
- Feed-forward MLP  
- LayerNorm  
- Residuals  

This gives PCT a **global receptive field** even though neighborhoods are local.

---

## 4. **Performance**  
PCT achieved strong results on:

- **ModelNet40 classification**  
- **ShapeNet part segmentation**  

It was one of the first *pure transformer* point-cloud backbones that beat PointNet++ consistently.

---

# 🔄 Closest Improvements After PCT  
These are the models that directly build on PCT’s ideas (offset attention, geometry-aware attention, global context).

---

# 1. **Point Transformer v1** — Local vector attention  




**Relation to PCT:**  
Both use **relative geometry** inside attention.  
Point Transformer v1 generalizes OA into **vector attention**:

\[
\text{Attn}(i,j) = \phi(x_j - x_i) \cdot \psi(h_j)
\]

This is a more expressive version of PCT’s offset mechanism.

**Improvement over PCT:**  
- Better positional encoding  
- More flexible attention  
- Stronger segmentation performance  

---

# 2. **Point-BERT** — Masked pretraining on PCT-like patch tokens  




**Relation to PCT:**  
Point-BERT uses **DGCNN/PCT-style local patches** as tokens.  
It applies **BERT-style masked modeling** to learn strong 3D representations.

**Improvement over PCT:**  
- Pretraining boosts downstream accuracy  
- Learns a discrete codebook of local geometry  

---

# 3. **Point-MAE** — MAE for point clouds  




**Relation to PCT:**  
Point-MAE uses **transformer encoders** similar to PCT but trains them with **masked autoencoding**.

**Improvement over PCT:**  
- Much stronger global features  
- Better scaling  
- State-of-the-art on classification/segmentation  

---

# 4. **PCT Variants (2021–2022)**  
These are direct architectural tweaks:

### **PCTv2**  
- Adds **cross-attention** between local and global tokens  
- Improves segmentation stability  

### **PCT-Conv**  
- Hybridizes PCT with lightweight convolutions  
- Faster and more stable on large scenes  

### **PCT-Lite**  
- Reduces OA complexity  
- Mobile-friendly  

---

# 5. **Stratified Transformer (STransformer)**  




**Relation to PCT:**  
Still uses **local attention with geometry**, but introduces:

- Stratified sampling  
- Multi-scale neighborhoods  
- Sparse attention  

**Improvement over PCT:**  
- Scales to millions of points  
- State-of-the-art on ScanNet/S3DIS  

---

# 🧱 Summary Table

| **Model** | **Relation to PCT** | **Main Improvement** |
|----------|---------------------|-----------------------|
| **PCT (original)** | Offset-attention, pure transformer | Geometry-aware attention |
| **Point Transformer v1** | Generalizes OA → vector attention | More expressive, stronger segmentation |
| **Point-BERT** | Uses PCT-like patches | Masked pretraining |
| **Point-MAE** | Transformer encoder like PCT | MAE pretraining, huge gains |
| **PCTv2 / PCT-Lite** | Direct successors | Efficiency + stability |
| **Stratified Transformer** | Same philosophy (local geometry + attention) | Scales to large scenes |

---

# If you want to go deeper  
Pick one and I’ll expand with equations + implementation details:

- **Offset-Attention math**  
- **PCT vs Point Transformer**  
- **PCT pretraining variants**  
- **Implement PCT in PyTorch**

# PT vs PCT

Here’s the clean, **definitive** clarification you’re looking for — because yes, the naming is confusing, and no, **PCT (Point Cloud Transformer)** and **Point Transformer** are *not* the same family.

---

# 🎯 **Short, crisp answer**
**PCT (Point Cloud Transformer)** and **Point Transformer** are **two completely different transformer architectures** for point clouds, created by **different authors**, with **different mechanisms**, and **different goals**.

They are **not versions of each other**, and neither is a “first generation” of the other.

They just happen to have similar names.

---

# 🧩 **Why the confusion happens**
Because both papers:

- came out around **2020–2021**
- use transformers for point clouds  
- have names that differ by only one word  
- both propose “geometry-aware attention”  

But internally, they are **very different**.

---

# 🆚 **PCT vs Point Transformer — the real difference**

## 1. **PCT — Point Cloud Transformer (Guo et al., 2021)**  




- Introduces **Offset Attention (OA)**  
  \[
  \text{OA}(i) = x_i + \sum_j \alpha_{ij}(x_j - x_i)
  \]
- Focus: **global context** + simple local embedding  
- Architecture: **pure transformer**, no hierarchical downsampling  
- Tasks: classification, part segmentation  
- Lightweight, simple, early transformer baseline  

---

## 2. **Point Transformer (Zhao et al., 2021)**  




- Introduces **Vector Attention**  
  \[
  \text{Attn}(i,j) = \phi(x_j - x_i) \cdot \psi(h_j)
  \]
- Focus: **hierarchical architecture** (like PointNet++)  
- Architecture: downsample → local attention → upsample  
- Tasks: segmentation, detection, large scenes  
- Much more expressive and widely adopted  

---

# 🧱 **How to think about them**
Use this analogy:

- **PCT** = “ViT-like” transformer for point clouds  
  - simple  
  - global  
  - early prototype  
  - good for small datasets  

- **Point Transformer** = “ResNet-like” transformer for point clouds  
  - hierarchical  
  - strong geometric modeling  
  - widely used in 3D perception  
  - basis for PTv2, PTv3  

They solve similar problems but come from **different design philosophies**.

---

# 🧬 **Are they related historically?**
Not directly.

### Timeline:
- **PCT**: early 2021  
- **Point Transformer**: late 2020 / early 2021  

They were developed **independently**.

### Influence:
- Later works (Point-BERT, Point-MAE, PTv2, PTv3)  
  → mostly build on **Point Transformer**, not PCT  
- PCT influenced some lightweight variants, but not the mainstream line  

---

# 📌 **Summary Table**

| Model | Family | Key Mechanism | Architecture | Influence |
|------|--------|----------------|--------------|-----------|
| **PCT** | PCT family | Offset Attention | Flat transformer | Early baseline |
| **Point Transformer v1** | PT family | Vector Attention | Hierarchical | Very influential |
| **PTv2** | PT family | Grouped vector attention | Hierarchical | Strong |
| **PTv3** | PT family | Scalable sparse attention | Hierarchical | SOTA |

---

# If you want, I can now go deeper into:

- **Offset Attention vs Vector Attention**  
- **Why Point Transformer became the dominant family**  
- **Which one is better for LiDAR detection**  

Just pick one and I’ll expand.
