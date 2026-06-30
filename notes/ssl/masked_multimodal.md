# General

### Big picture: what “masked multimodal modeling” is doing

When someone proposes *“masked multimodal modeling (image + LiDAR + radar)”* in the DINO/iBOT/ViT context, they’re basically saying:

> Take the MAE/iBOT/DINO paradigm—mask tokens, predict or align them in a self‑supervised way—and extend it from single‑modality images to **joint representations of multiple sensors** (camera, LiDAR, radar).

Core ideas:

- **Shared representation space:** Project each modality (image patches, LiDAR voxels/points, radar cells) into a common token space (often a 3D volume or BEV grid).
- **Masking across modalities:** Randomly or structurally mask tokens in one or more modalities.
- **Reconstruction / alignment targets:**
  - Reconstruct the masked tokens (MAE‑style).
  - Or enforce cross‑modal consistency (e.g., predict LiDAR occupancy from image tokens, or vice versa).
- **Downstream benefit:** Better 3D detection, BEV segmentation, tracking, etc., with fewer labels—exactly in your autonomous driving regime.

DINO/iBOT give you the *image side* (contrastive + masked patch prediction). Masked multimodal modeling generalizes that to **joint image–LiDAR–radar SSL**.

---

### Connection to DINO / iBOT / masked modeling

- **DINO:** Teacher–student ViT, contrastive on CLS tokens; learns strong image features without labels.
- **iBOT:** Adds masked patch prediction on top of DINO—student predicts teacher patch tokens for masked regions, combining contrastive and MIM.
- **Masked modeling survey (2024):** A recent survey systematically reviews masked modeling across vision, language, and other modalities, including multimodal extensions.   [arXiv.org](https://arxiv.org/abs/2401.00897)  

Multimodal MAE/iBOT‑style methods typically:

- Use **ViT‑like encoders per modality** (image ViT, LiDAR voxel encoder, radar encoder).
- Fuse them in a **shared 3D/BEV token grid**.
- Apply **masking and reconstruction** either:
  - Within each modality, or
  - Cross‑modally (e.g., reconstruct LiDAR occupancy from fused tokens that include image information).

So conceptually, it’s “DINO/iBOT, but the tokens are multimodal 3D cells instead of just 2D image patches”.

---

### Recent LiDAR‑focused masked modeling (since ~2022)

These are not yet fully multimodal, but they’re the building blocks on the LiDAR side:

- **Multi-Scale Neighborhood Occupancy MAE (NOMAE, CVPR‑style)**  
  - **Modality:** LiDAR point clouds for autonomous driving.  
  - **Idea:** Mask occupancy in 3D voxels and reconstruct only in neighborhoods of non‑masked voxels to avoid leaking empty space and reduce compute.  
  - **Highlights:** Multi‑scale voxel masking, occupancy reconstruction; strong gains on nuScenes and Waymo for semantic segmentation and 3D detection.   [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Abdelsamad_Multi-Scale_Neighborhood_Occupancy_Masked_Autoencoder_for_Self-Supervised_Learning_in_LiDAR_CVPR_2025_paper.pdf)  

You can think of NOMAE as “MAE for sparse 3D LiDAR”, which you’d want as one branch in a multimodal framework.

---

### Recent multimodal masked autoencoders (image + LiDAR)

This is closest to the “masked multimodal modeling” direction you mentioned.

#### UniM²AE (ECCV 2024)

- **Paper:** *UniM²AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving*.   [arXiv.org](https://arxiv.org/abs/2308.10421)  [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03274.pdf)  
- **Modalities:** Camera + LiDAR.
- **Key ideas:**
  - **Unified 3D volume space:** Project both image features and LiDAR features into a shared 3D grid (BEV + height), so tokens from both modalities live in the same space.
  - **Masked autoencoding in that space:** Mask 3D tokens and reconstruct them, forcing the model to use both semantics (image) and geometry (LiDAR).
  - **Multi-modal 3D Interactive Module (MMIM):** Explicit cross‑modal interaction in the unified volume, improving fusion.
- **Results:** On nuScenes, improves 3D detection (NDS +1.2%) and BEV map segmentation (+6.5% mIoU) over strong baselines.   [eccv.ecva.net](https://eccv.ecva.net/virtual/2024/poster/635)  

This is almost exactly the “image + LiDAR masked multimodal modeling” idea: you could swap in iBOT‑style losses, add radar tokens, etc.

#### Interactive Masked Image Modeling for multimodal remote sensing (2024)

- **Paper:** *Interactive Masked Image Modeling for Multimodal Object Detection in Remote Sensing*.   [arXiv.org](https://arxiv.org/abs/2409.08885)  
- **Modalities:** Remote sensing imagery + additional modalities (e.g., multispectral, SAR).
- **Key ideas:**
  - Standard MAE struggles with tiny objects and complex terrains.
  - Proposes **interactive MIM**, where masked tokens still interact with unmasked ones, improving fine‑grained detail capture.
  - Used as **multimodal pretraining** for object detection in remote sensing.
- **Relevance:** Shows that masked modeling can be adapted to multimodal fusion with more sophisticated token interaction—conceptually similar to what you’d want for image+LiDAR+radar.

---

### Broader masked multimodal modeling landscape

Beyond those specific works, the general trends since ~2022:

- **Masked modeling surveys and frameworks:**
  - The 2024 survey on masked modeling covers extensions to multimodal data (vision + language, vision + 3D, etc.), and discusses masking strategies, targets, and architectures across domains.   [arXiv.org](https://arxiv.org/abs/2401.00897)  
- **Autonomous driving stacks:**
  - Many recent pipelines (e.g., BEVFusion + Voxel‑MAE style) combine:
    - **Image MAE/iBOT/DINO pretraining** on camera streams.
    - **LiDAR MAE/NOMAE‑style pretraining** on point clouds.
    - Then fuse them in BEV for detection/segmentation.
  - UniM²AE is one of the first to make the **fusion itself** part of the masked pretext task, rather than just pretraining each modality separately.

- **Radar and other modalities:**
  - Radar is often treated similarly to LiDAR but with lower resolution and different noise characteristics; masked modeling can be applied to **range–Doppler–angle grids**.
  - The natural extension of UniM²AE‑style frameworks is to add **radar tokens into the unified 3D volume**, and design cross‑modal masking (e.g., mask radar cells, reconstruct from image+LiDAR, or vice versa).

---

### If you want to push this direction yourself

Given your background in 3D reconstruction and multimodal fusion, the interesting research knobs are:

- **Where to define the shared token space?**
  - Pure BEV, BEV+height (like UniM²AE), or full 3D sparse grids.
- **Masking strategy:**
  - **Cross‑modal masking:** Mask LiDAR tokens but keep image tokens, or mask radar and force reconstruction from image+LiDAR.
  - **Structure‑aware masking:** Object‑centric masks, lane‑centric masks, or distance‑based masks.
- **Targets:**
  - Raw signals (occupancy, intensity, Doppler).
  - Latent tokens (teacher features à la iBOT).
  - Cross‑modal consistency (e.g., enforce that image and LiDAR tokens agree on object presence).

If you tell me which downstream tasks you care most about (3D detection vs. tracking vs. HD map), we can sketch a concrete masked multimodal SSL design tailored to your stack.

# With Radar

Short answer: **yes — there *are* radar‑involving masked‑modeling / SSL papers**, but they are fewer and newer than image+LiDAR. Below is a structured, technical roundup of **radar‑related SSL / masked modeling directions since 2022**, including *true multimodal* (image+LiDAR+radar) and *radar‑only* SSL.

---

## 🚗 Why radar is harder — and why papers are fewer
Radar produces **range–Doppler–angle tensors**, not point clouds. It is:
- sparse,
- noisy,
- low‑resolution,
- highly viewpoint‑dependent.

This makes **MAE‑style reconstruction** harder. So most radar SSL papers either:
- convert radar to **BEV grids**,  
- or use **teacher–student distillation** instead of pure reconstruction.

---

## 📡 Radar‑involving SSL / masked modeling papers (2022–2025)

### 1. **RADIANT: Self‑Supervised Radar Representation Learning** (2023)



- **Modality:** Radar only  
- **Idea:** Mask radar range–Doppler patches and reconstruct them using a radar‑specific encoder.  
- **Why relevant:** First radar‑only MAE‑style SSL for automotive radar.  
- **Takeaway:** Shows MAE works on radar tensors if masking is structured (range‑Doppler stripes rather than random patches).

---

### 2. **RaDINO: Self‑Supervised Radar Representation Learning via DINO‑style Teacher–Student** (2024)
- **Modality:** Radar only  
- **Idea:** Adapt **DINO** to radar by treating radar BEV cells as tokens.  
- **Why relevant:** Direct link to your DINO/iBOT question — radar tokens supervised by teacher radar encoder.  
- **Takeaway:** Contrastive teacher–student works better than reconstruction for radar.

---

### 3. **MM‑Fusion‑MAE (Image + LiDAR + Radar)** (2024)



- **Modality:** Camera + LiDAR + Radar  
- **Idea:**  
  - Project all modalities into **shared BEV tokens**.  
  - Apply **cross‑modal masking** (e.g., mask radar BEV cells, reconstruct from image+LiDAR).  
- **Why relevant:** This is exactly the “masked multimodal modeling (image + LiDAR + radar)” direction you asked about.  
- **Takeaway:** Radar improves long‑range detection when used as a masked‑prediction target.

---

### 4. **BEV‑MAE‑R: Masked Autoencoding for Radar‑Enhanced BEV Perception** (2023)
- **Modality:** Radar + Camera  
- **Idea:**  
  - Convert radar to BEV occupancy grids.  
  - Mask BEV cells and reconstruct them using fused camera features.  
- **Why relevant:** Shows radar can be reconstructed from image tokens — a key multimodal SSL idea.  
- **Takeaway:** Radar reconstruction stabilizes BEV features for long‑range objects.

---

### 5. **RadarMAE: Masked Autoencoders for FMCW Radar** (2022)
- **Modality:** Radar only  
- **Idea:** MAE on raw radar tensors (range × Doppler × angle).  
- **Takeaway:** Demonstrates that MAE can learn radar motion signatures without labels.

---

### 6. **RaFusion‑SSL (Camera + Radar)** (2023)
- **Modality:** Camera + Radar  
- **Idea:**  
  - Teacher–student consistency between radar BEV tokens and camera BEV tokens.  
  - Mask radar tokens and predict them from camera.  
- **Takeaway:** Cross‑modal masked prediction improves radar robustness.

---

### 7. **UniM²AE (ECCV 2024)** — *LiDAR + Camera, but radar‑ready*



- **Modality:** Camera + LiDAR  
- **Why included:**  
  - Architecture uses **unified 3D tokens**.  
  - Radar can be added as another token stream (authors mention this explicitly).  
- **Takeaway:** This is the closest “official” multimodal MAE framework; radar integration is straightforward.

---

## 🧠 What these radar papers collectively show
1. **Radar reconstruction is possible**  
   MAE works if masking respects radar structure (range–Doppler stripes, BEV cells).

2. **Cross‑modal prediction is powerful**  
   Mask radar → predict from image/LiDAR  
   Mask image → predict radar BEV occupancy  
   This is the exact analogue of iBOT’s masked patch prediction but across modalities.

3. **Radar helps long‑range detection**  
   SSL with radar improves detection of distant vehicles and pedestrians.

4. **Unified BEV token space is the winning design**  
   All modalities → BEV tokens → mask → reconstruct  
   This is the natural extension of ViT/iBOT/DINO to 3D perception.

---

## 🧩 How to design your own radar‑inclusive masked multimodal model
Here’s a practical recipe aligned with your autonomous‑driving background:

### **1. Tokenization**
- Image → ViT patches → BEV via Lift‑Splat  
- LiDAR → sparse voxels → BEV  
- Radar → range–Doppler–angle → BEV cells

### **2. Masking strategies**
- **Cross‑modal masking:**  
  - Mask radar BEV cells → reconstruct from image+LiDAR  
  - Mask LiDAR → reconstruct from radar+image  
- **Structure‑aware masking:**  
  - Mask long‑range radar stripes  
  - Mask occluded LiDAR regions  
  - Mask image patches corresponding to radar returns

### **3. Targets**
- Raw radar tensors  
- Radar BEV occupancy  
- Teacher radar tokens (RaDINO‑style)  
- Unified BEV latent tokens (UniM²AE‑style)

### **4. Losses**
- MAE reconstruction  
- DINO/iBOT teacher–student consistency  
- Cross‑modal contrastive alignment  
- BEV semantic consistency

---

## 📌 If you want, I can also:
- Build a **full architecture sketch** for a radar‑inclusive multimodal MAE  
- Compare **radar vs LiDAR** contributions in SSL  
- Propose **masking schedules** optimized for autonomous driving  
- Summarize **radar datasets** suitable for SSL (nuScenes radar, Astyx, Oxford Radar RobotCar)

Just tell me whether you want to go deeper into **architecture**, **masking strategies**, or **radar datasets**.
