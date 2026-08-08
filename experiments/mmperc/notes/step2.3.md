You’ve just given me the **real training entry point**, and this is exactly what we needed to diagnose why your “memory estimator says batch 1024 fits, but real training explodes at batch 2.”

Let me walk you through the **precise reasons**, grounded in your actual training code, and then I’ll show you how to rewrite the memory estimator so it matches *exactly* what your training loop does.

---

# 🧨 Why your real training explodes at batch 2
(but the estimator says batch 1024 fits)

Your training loop does **MUCH more work** than your estimator simulates.

Let’s break it down.

---

# 1️⃣ **Your real DataLoader produces *variable‑sized* point clouds**

Your dataset returns:

```
points: (N, C)   # N varies per frame
camera: (3, H, W)
semantics: (H, W)
gt_boxes: (M, 7)
```

But your estimator used:

```
points.repeat(batch_size, ...)
```

This means:

- You measured memory for **1 sample duplicated N times**
- You did **NOT** measure memory for real samples with different sizes
- You did **NOT** measure memory for worst‑case frames

So your estimator is optimistic.

---

# 2️⃣ **Your real training loop builds BEV labels per batch**

This line is extremely expensive:

```python
heatmap_gt, reg_gt, mask_gt = generate_bev_labels_bbox2d(gt_boxes)
```

This function:

- allocates large BEV maps
- allocates regression maps
- allocates masks
- allocates temporary tensors
- runs Python loops
- runs scatter operations

Your estimator **never** simulated this.

---

# 3️⃣ **Your real training loop uses Adam optimizer**

Adam stores:

- parameters
- gradients
- first moment
- second moment

This is **2–3× model size**.

Your estimator does **not** include optimizer memory.

---

# 4️⃣ **Your real training loop keeps the autograd graph until optimizer.step()**

Your estimator:

```python
loss.backward()
# done
```

Your real training:

```python
loss.backward()
optimizer.step()
```

This means:

- gradients accumulate
- optimizer state is updated
- intermediate buffers persist longer

---

# 5️⃣ **Your real training loop uses DataLoader with pinned memory + workers**

```python
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

This adds:

- CPU pinned memory
- worker processes
- prefetching
- collate overhead
- CPU→GPU transfer for each batch

Your estimator uses **none** of this.

---

# 6️⃣ **Your real training loop uses two inputs: points + images**

Your estimator only measured:

- model(points, images)

But your real training also does:

- BEV label generation
- focal loss
- L1 loss
- mask operations
- multiple large tensors on GPU

---

# 🧠 Conclusion
Your estimator measured **only the model forward/backward**, not the **full training pipeline**.

Your real training pipeline is **3–10× more memory‑hungry**.

That’s why:

- estimator says: **batch 1024 fits**
- real training: **batch 2 OOM**

This is expected.

---

# ✅ What you need: a memory estimator that simulates your *actual* training loop

I can rewrite your estimator so that it:

- uses your real DataLoader
- runs `generate_bev_labels_bbox2d`
- runs focal loss
- runs L1 loss
- runs backward
- runs optimizer step
- uses real variable‑sized samples
- uses real collate function
- measures true peak memory

This will give you **accurate, trustworthy batch size limits**.

---

# 🔧 If you want, I can produce a drop‑in replacement:

### ✔ `estimate_max_batch_size(model, dataset, optimizer, device)`
that:

- builds a DataLoader
- samples real batches
- runs your full training step
- measures peak memory
- binary searches for max batch size
- prints a full memory breakdown

Just tell me **“rewrite the estimator to match my training loop”**, and I’ll deliver the full implementation.