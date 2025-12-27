Absolutely — Day 3 is where multi‑task learning *actually becomes real*.  
This is the day where you learn **why MTL fails**, **how to fix it**, and **how to make multiple tasks coexist without destroying each other**.  
Given your background, I’ll go straight to the high‑leverage concepts and the practical intuition you’ll use in real training runs.

---

# 🟨 **DAY 3 — Loss Balancing & Optimization**  
*Goal: understand and control gradient interactions so your MTL model actually trains.*

---

# 1️⃣ Why Loss Balancing Matters  
In MTL, each task produces its own gradient:

\[
g_1, g_2, ..., g_T
\]

If these gradients:

- **align** → tasks help each other  
- **conflict** → tasks sabotage each other  
- **differ in magnitude** → one task dominates  

This is the root cause of:
- negative transfer  
- oscillating training  
- one task collapsing  
- shared backbone not learning anything useful  

So Day 3 is about **controlling gradients**.

---

# 2️⃣ The Four Essential Loss‑Balancing Methods  
These are the only ones you really need in practice.

---

## 🟦 **1. Equal Weights (baseline)**  
\[
L = \sum_i L_i
\]

Simple, but almost always suboptimal.

**Use it for:**  
- debugging  
- verifying your heads work  
- establishing a baseline  

---

## 🟩 **2. Uncertainty Weighting (Kendall et al.) — your default choice**  
This method learns a weight for each task based on its predictive uncertainty.

### **Formula (intuition only)**
Each task gets a learned parameter \( \sigma_i \).  
Loss becomes:

\[
L = \sum_i \frac{1}{2\sigma_i^2} L_i + \log \sigma_i
\]

### **Why it works**
- tasks with high noise get lower weight  
- tasks with low noise get higher weight  
- weights adapt automatically during training  
- extremely stable  

### **Use it when**
- tasks are related  
- you want a strong baseline  
- you want stability without tuning  

This is the method I recommend you start with.

---

## 🟧 **3. GradNorm — equalizes gradient magnitudes**  
This method forces all tasks to train at similar speeds.

### **Intuition**
- compute gradient norm for each task  
- compare to target norm  
- adjust task weights to equalize training rates  

### **Why it works**
- prevents one task from dominating  
- stabilizes training  
- great for tasks with different difficulty levels  

### **Use it when**
- one task learns much faster  
- one task collapses  
- loss scales differ dramatically  

---

## 🟥 **4. PCGrad — resolves gradient conflict**  
This is the most “surgical” method.

### **Intuition**
For each pair of tasks:
- if gradients conflict (negative cosine similarity)  
- project one gradient onto the orthogonal plane of the other  

This removes destructive interference.

### **Why it works**
- prevents tasks from fighting  
- especially good for heterogeneous tasks  
- very stable  

### **Use it when**
- tasks are unrelated  
- tasks actively conflict  
- you see oscillation in shared layers  

---

# 3️⃣ How to Detect Gradient Problems (your diagnostic toolkit)

## **A. Gradient conflict**
Check cosine similarity between task gradients:
- positive → good  
- zero → neutral  
- negative → conflict  

## **B. Loss divergence**
If one task improves while another worsens → conflict.

## **C. Backbone gradient oscillation**
If shared layers keep flipping direction → conflict.

## **D. Task collapse**
One task’s loss stops decreasing → dominated by others.

## **E. Per‑task learning curves**
If one task is “too easy” or “too hard”, it will dominate.

---

# 4️⃣ Practical Rules You’ll Actually Use

### **Rule 1 — Start with uncertainty weighting**  
It’s stable, simple, and works surprisingly well.

### **Rule 2 — If tasks differ in difficulty → use GradNorm**  
This equalizes training speed.

### **Rule 3 — If tasks conflict → use PCGrad**  
This removes destructive interference.

### **Rule 4 — If tasks are loosely related → reduce sharing**  
Adapters or separate decoders help.

### **Rule 5 — Always log gradient norms**  
This is the single most useful debugging signal in MTL.

---

# 5️⃣ Your Day‑3 Mini‑Experiment (1–2 hours)  
This is the fastest way to *feel* MTL dynamics.

### **Dataset**
MNIST or CIFAR‑10.

### **Tasks**
- Task A: classification  
- Task B: binary attribute (even/odd or coarse class)  

### **Steps**
1. Train with equal weights → observe imbalance  
2. Add uncertainty weighting → observe stability  
3. Add GradNorm → observe equalized training  
4. Add PCGrad → observe reduced conflict  

This experiment gives you intuition that will transfer to real models.

---

# 6️⃣ Your Day‑3 Deliverable  
A short note (you can write it or I can generate it) containing:

- which loss balancing method you prefer  
- how you detect gradient conflict  
- how you’d debug negative transfer  
- which method you’d use for your robotics/autonomous‑driving tasks  

This reflection locks the concepts in.

---

If you want, I can now move to **Day 4**, where you build a real multi‑task model with a shared backbone, two heads, and dynamic loss balancing — the part where everything becomes concrete.
