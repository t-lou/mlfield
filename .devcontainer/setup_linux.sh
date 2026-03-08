#!/bin/bash
set -e

# This script handles GPU and Docker setup for Linux (WSL/Ubuntu/Debian)

# =============================
# Helper functions
# =============================

install_nvidia_repo() {
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt update
}

# =============================
# GPU CHECK 1: nvidia-smi in host
# =============================
echo "Checking GPU availability in host..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ nvidia-smi not found in host."
    exit 1
fi

nvidia-smi || {
    echo "❌ nvidia-smi failed inside host."
    echo "→ GPU is not exposed to host. Reboot or reinstall NVIDIA driver."
    exit 1
}
echo "✅ GPU detected in host."
echo

# =============================
# GPU CHECK 2: NVIDIA container runtime
# =============================
echo "Checking NVIDIA container runtime..."
if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
    echo "⚠️ NVIDIA container runtime missing. Installing..."

    install_nvidia_repo

    sudo apt install -y nvidia-container-toolkit

    echo "Configuring Docker runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker

    echo "✅ NVIDIA container toolkit installed."
    echo "Please restart host and re-run this script (in WSL explicit reboot is needed)."
    exit 0
fi

echo "✅ NVIDIA container runtime installed."
echo

# =============================
# GPU CHECK 3: Docker GPU support
# =============================
echo "Checking Docker GPU support..."
if ! docker info | grep -qi nvidia; then
    echo "❌ Docker does not list NVIDIA runtime."
    bash ./.devcontainer/install_docker.sh
    echo "Docker installed, reboot."
    exit 1
fi
echo "✅ Docker recognizes NVIDIA runtime."
echo

# =============================
# GPU CHECK 4: Test GPU passthrough
# =============================
echo "Testing GPU passthrough with CUDA container..."
if ! docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "❌ Docker cannot access GPU."
    echo "→ Likely missing toolkit or Docker misconfiguration."
    exit 1
fi
echo "✅ GPU passthrough works."
echo
