Absolutely, Tongxi — and I’m really glad to hear your PyTorch CUDA check finally passed. That’s the moment when all the Debian pain suddenly feels worth it. Let me help you turn your experience into a **clean, readable, copy‑friendly installation guide** that someone else could follow without hitting the traps you hit.

I’ll keep it structured, concise, and technically accurate — but still in your voice and with the reasoning behind each step.

---

# 🧭 **Installing Debian 13 on External Drive + NVIDIA RTX 2050 + CUDA ≥ 12.9 (CUDA 13.1)**
**Hardware:** Intel i5 + NVIDIA GeForce RTX 2050 (Optimus laptop)
**OS layout:** Windows on internal SSD, Debian 13 on external SSD
**Goal:** Fresh Debian 13 with working NVIDIA proprietary driver + CUDA ≥ 12.9
**Why:** Debian is clean, minimal, stable, and no longer outdated — perfect for private ML projects without Ubuntu’s noise.

This guide reflects the exact steps required to get **CUDA 13.1** working on Debian 13, avoiding all the pitfalls (nouveau, DKMS failures, keyring issues, driver/toolkit mismatch).

---

# ✅ **Final Result**
A fresh Debian 13 system with:

- NVIDIA proprietary driver (from NVIDIA, not Debian)
- CUDA 13.1 toolkit
- PyTorch CUDA working
- No nouveau
- No DKMS conflicts
- No Debian driver interference

---

# 🧩 **Step 1 — Disable nouveau (critical on Debian 13)**
Debian is stricter than Ubuntu: nouveau will load even if blacklisted unless you update initramfs *and* GRUB.
If nouveau loads, the NVIDIA driver cannot build or load.

Create blacklist:

```bash
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```

Add **all three lines**:

```
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
```

Update initramfs:

```bash
sudo update-initramfs -u
```

Add kernel parameter (important on Debian 13):

```bash
sudo nano /etc/default/grub
```

Modify:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nouveau.modeset=0"
```

Apply:

```bash
sudo update-grub
sudo reboot
```

Verify nouveau is gone:

```bash
lsmod | grep nouveau
```

No output = success.

> **Note:** On Debian 13, nouveau deactivation is stricter than other distros.
> If nouveau is not fully disabled, the proprietary NVIDIA driver will fail to build or load.

---

# 🧩 **Step 2 — Install CUDA keyring for Debian 13**
This adds NVIDIA’s official repository.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

⚠️ **Important:**
Do **NOT** install Debian’s `nvidia-driver` package before this.
Debian’s driver is older and will conflict with CUDA 13.1.

---

# 🧩 **Step 3 — Install NVIDIA’s official driver (`cuda-drivers`)**
This installs the newest NVIDIA driver (560+), compatible with CUDA 13.1.

```bash
sudo apt install cuda-drivers
```

Reboot:

```bash
sudo reboot
```

Check:

```bash
nvidia-smi
```

You should see:

- Driver version 560+
- CUDA Version: 13.1

---

# 🧩 **Step 4 — Install CUDA toolkit (host development)**
If you’re developing directly on the host (not Docker):

```bash
sudo apt install cuda-toolkit-13-1
```

This installs:

- nvcc
- CUDA libraries
- headers
- samples

And **does not** overwrite the driver.

---

# 🎉 **Done — verify with PyTorch**
In Python:

```python
import torch
torch.cuda.is_available()
torch.cuda.get_device_name(0)
```

Should return:

- `True`
- `"NVIDIA GeForce RTX 2050"`

---

# 🧠 Optional Notes

### • Why not Debian’s `nvidia-driver`?
It’s older and only supports CUDA 12.4.
CUDA 13.1 requires a newer driver.

### • Why not Ubuntu?
Debian is cleaner, quieter, and more predictable — perfect for private ML projects.

### • Why external SSD?
Keeps Windows untouched and avoids dual‑boot headaches.

---

If you want, I can help you turn this into a **Markdown README**, a **blog post**, or a **GitHub gist**.
