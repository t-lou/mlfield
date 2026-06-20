#!/bin/bash
set -euo pipefail

BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR"

# --- Detect NVIDIA GPU availability ---
echo "Checking for NVIDIA GPU..."

HAS_NVIDIA=false

# Case 1: nvidia-smi exists (host has NVIDIA driver)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✔ nvidia-smi found — NVIDIA GPU detected"
    HAS_NVIDIA=true

# Case 2: Docker runtime exposes NVIDIA (common in WSL2)
elif docker info 2>/dev/null | grep -qi "nvidia"; then
    echo "✔ Docker reports NVIDIA runtime — GPU available"
    HAS_NVIDIA=true

# Case 3: Check for /dev/nvidia* devices
elif ls /dev/nvidia* >/dev/null 2>&1; then
    echo "✔ /dev/nvidia* devices found — GPU available"
    HAS_NVIDIA=true

else
    echo "✘ No NVIDIA GPU detected — falling back to CPU build"
fi

# --- Build correct base image ---
if [ "$HAS_NVIDIA" = true ]; then
    echo "Building CUDA base image..."
    docker build -f Dockerfile.cuda.base -t mlfield_cuda_base:latest .
else
    echo "Building CPU base image..."
    docker build -f Dockerfile.cpu.base -t mlfield_cpu_base:latest .
fi

echo "Done."
