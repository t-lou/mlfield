Perfect — Day 4 is where everything becomes **real**.  
Today you’ll **build a working multi‑task learning system**: a shared backbone, two task heads, dynamic loss balancing, and the logging you need to diagnose gradient conflict.

I’ll give you a clean, practical blueprint that fits your engineering style: minimal, reproducible, and easy to extend.

---

# 🟧 **DAY 4 — Build a Real Multi‑Task Model**  
*Goal: implement a full MTL pipeline with shared backbone, two heads, and dynamic loss balancing.*

---

# 1️⃣ Choose a Dataset (fast + practical)

You want something that:
- has multiple labels per image  
- is small enough to train quickly  
- demonstrates real MTL dynamics  

### **Recommended options**
### **Option A — NYUv2 (best for MTL)**
- RGB → segmentation + depth + normals  
- Classic MTL benchmark  
- Shows gradient conflict clearly  

### **Option B — Cityscapes**
- segmentation + instance segmentation  
- closer to autonomous driving  

### **Option C — CIFAR‑10 (toy MTL)**
- classification + coarse label (superclass)  
- fastest to iterate  

If you want speed, start with **CIFAR‑10**.  
If you want realism, start with **NYUv2**.

---

# 2️⃣ Architecture Blueprint (shared backbone + two heads)

Here’s the exact structure you should implement today.

```
                ┌───────────────┐
                │   Input RGB    │
                └───────┬───────┘
                        ▼
                ┌────────────────┐
                │  Shared Backbone│  ← ResNet50 / Swin / ViT
                └───────┬────────┘
        ┌───────────────┼────────────────┐
        ▼                               ▼
┌──────────────┐               ┌────────────────┐
│ Seg Head      │               │ Depth Head     │
│ (decoder)     │               │ (decoder)      │
└──────┬────────┘               └──────┬────────┘
       ▼                               ▼
 Segmentation Map                 Depth Map
```

### **Backbone**
- ResNet50 or Swin‑T  
- Pretrained weights recommended  
- Freeze first 1–2 stages for stability  

### **Heads**
- Segmentation head:  
  - FPN → 1×1 conv → upsample → softmax  
- Depth head:  
  - FPN → 1×1 conv → upsample → regression  

### **Why this works**
- Shared features capture geometry + semantics  
- Heads specialize  
- FPN gives multi‑scale features for both tasks  

---

# 3️⃣ Implement Dynamic Loss Balancing

Start with **uncertainty weighting** (Day 3’s default).

### **Loss**
\[
L = \frac{1}{2\sigma_s^2} L_{seg} + \frac{1}{2\sigma_d^2} L_{depth} + \log \sigma_s + \log \sigma_d
\]

Where:
- \( \sigma_s \) and \( \sigma_d \) are learnable scalars  
- They automatically adjust task weights  

### **Why this is perfect for Day 4**
- No hyperparameter tuning  
- Very stable  
- Works for heterogeneous tasks  

---

# 4️⃣ Add Gradient Logging (your debugging superpower)

Log:
- gradient norms per task  
- cosine similarity between task gradients  
- per‑task loss curves  
- backbone gradient variance  

This gives you:
- early detection of negative transfer  
- insight into whether PCGrad or GradNorm is needed  
- understanding of which layers conflict  

You’ll use this heavily in Day 5.

---

# 5️⃣ Training Loop (the minimal version)

### **Steps**
1. Forward pass through backbone  
2. Forward pass through each head  
3. Compute each task loss  
4. Combine with uncertainty weighting  
5. Backprop once  
6. Log gradients  
7. Update backbone + heads + σ parameters  

### **Training tips**
- Use separate optimizers for heads if unstable  
- Use gradient clipping (1.0)  
- Use mixed precision for speed  
- Freeze early backbone layers for first 1–2 epochs  

---

# 6️⃣ Evaluation (simple but essential)

Evaluate each task separately:
- segmentation: mIoU  
- depth: RMSE or scale‑invariant loss  

Track:
- whether both tasks improve  
- whether one task collapses  
- whether dynamic weights stabilize  

If one task collapses → gradient conflict → use PCGrad tomorrow.

---

# 7️⃣ Your Day‑4 Deliverable

By the end of today, you should have:

### **A working MTL training script** with:
- shared backbone  
- two heads  
- uncertainty weighting  
- gradient logging  

### **A short note** on:
- which task dominates  
- whether gradients conflict  
- whether uncertainty weighting stabilizes training  

This sets you up perfectly for Day 5 (advanced topics + scaling).

---

# If you want, I can now give you:
- a **clean PyTorch code template** for this exact architecture  
- a **DETR‑style multi‑decoder version**  
- a **Cityscapes‑specific MTL design**  
- a **gradient logging snippet**  
- or move straight to **Day 5**  

Just tell me what direction you want.
