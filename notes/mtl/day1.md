Great — let’s dive straight into **Day 1**, but in a way that actually *builds the mental model you’ll use for the rest of your MTL work*. I’ll keep it tight, structured, and tuned to how you think: conceptual clarity + practical signals you can use in real training runs.

---

# 🟦 **DAY 1 — Build the Multi‑Task Learning Mental Model**  
*Your goal today: understand the landscape, the failure modes, and the decision rules.*

---

# 1️⃣ The Core Idea (in practitioner terms)

Multi‑task learning = **shared representation + task‑specific specialization**.

The entire game is balancing:
- **Shared features** (good for generalization, efficiency)
- **Task‑specific features** (good for specialization)
- **Gradient interactions** (the real battlefield)

If you understand how gradients from different tasks interact, you understand MTL.

---

# 2️⃣ The Taxonomy You Actually Need

## **A. Hard Parameter Sharing (most common)**
- One shared backbone  
- Multiple task heads  
- Shared optimizer

**Pros:** simple, efficient, strong regularization  
**Cons:** gradient conflict, negative transfer

Use when:
- Tasks are related  
- You want efficiency  
- You have limited data  

---

## **B. Soft Parameter Sharing**
Each task has its own model, but parameters are regularized to be similar.

Examples:
- Cross‑stitch networks  
- Sluice networks  
- Adapter‑based MTL

Use when:
- Tasks are loosely related  
- You want to avoid negative transfer  
- You have enough compute

---

## **C. Hybrid / Modern Architectures**
- Shared backbone + task‑specific adapters  
- Multi‑decoder transformers  
- Cross‑task attention modules  
- HyperNetworks generating task‑specific weights

Use when:
- Tasks differ in modality or difficulty  
- You want scalable MTL  
- You’re working with transformers

---

# 3️⃣ Task Relationships (the “Taskonomy” intuition)

Tasks can be:
- **Synergistic** (segmentation ↔ depth)  
- **Neutral** (classification ↔ keypoints)  
- **Conflicting** (detection ↔ surface normals)

You want to identify:
- **Which tasks should share early layers**  
- **Which tasks need separate decoders**  
- **Which tasks need adapters or routing**

A simple rule:
- **Low‑level tasks share early layers**  
- **High‑level tasks share late layers**  
- **Heterogeneous tasks share almost nothing**

---

# 4️⃣ The Real Enemy: Gradient Conflict

This is the heart of MTL.

Each task produces a gradient:  
\[
g_1, g_2, ..., g_T
\]

If:
- gradients point in similar directions → **positive transfer**  
- gradients oppose each other → **negative transfer**

You can detect conflict by:
- cosine similarity between gradients  
- per‑task loss curves diverging  
- one task improving while another degrades  
- backbone gradients oscillating

This is why loss balancing matters.

---

# 5️⃣ Loss Balancing Strategies (the essential ones)

You don’t need all of them — just understand the patterns.

## **A. Static weighting**
\[
L = \sum_i \lambda_i L_i
\]

Simple but brittle.

---

## **B. Uncertainty Weighting (Kendall et al.)**
Learns weights based on task uncertainty.

Pros:
- Works surprisingly well  
- Easy to implement  
- Stable  

Use as your **default baseline**.

---

## **C. GradNorm**
Equalizes gradient magnitudes across tasks.

Pros:
- Very effective  
- Good for tasks with different scales  

Use when:
- One task dominates  
- You see gradient imbalance

---

## **D. PCGrad**
Projects conflicting gradients to avoid destructive interference.

Pros:
- Best when tasks are *conflicting*  
- Very stable  

Use when:
- Tasks fight each other  
- You see oscillation or collapse

---

# 6️⃣ Negative Transfer: How to Detect It Early

You want to catch this before wasting compute.

### **Signals**
- One task improves, another degrades  
- Backbone gradients oscillate  
- Loss curves diverge  
- Shared layers show high gradient variance  
- Task‑specific heads learn fine, but shared layers stagnate  

### **Fixes**
- Add adapters  
- Reduce sharing  
- Use PCGrad  
- Use task‑specific batch norms  
- Use separate optimizers for heads

---

# 7️⃣ Your Day‑1 Deliverable (I’ll help you produce it)

A **one‑page cheat sheet** containing:

### **A. When MTL helps**
- Tasks share structure  
- Data is limited  
- You want efficiency  
- Tasks are synergistic  

### **B. When MTL hurts**
- Tasks are conflicting  
- Loss scales differ  
- One task is much harder  
- Backbone is too small  

### **C. How to detect negative transfer**
- Gradient conflict  
- Diverging losses  
- Oscillating backbone gradients  
- One task collapsing  

### **D. What to try first**
1. Shared backbone + task heads  
2. Uncertainty weighting  
3. Gradient logging  
4. PCGrad if conflict appears  

---

# If you want, I can now:
- **Generate your Day‑1 cheat sheet** in a clean, printable format  
- **Give you a gradient‑conflict diagnostic checklist**  
- **Give you a visual diagram of the MTL design space**  
- **Move to Day 2 (architectures)**

Just tell me what you want next.
