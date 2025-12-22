#!/bin/bash
set -e

# -----------------------------
# Resolve workspace directory
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WS_DIR"

# -----------------------------
# Export environment variables
# -----------------------------
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export USERNAME="$(whoami)"
export WS_DIR="$WS_DIR"

echo "Workspace: $WS_DIR"
echo "User: $USERNAME ($HOST_UID:$HOST_GID)"
echo

# -----------------------------
# Substitue the docker-compose
# -----------------------------
envsubst < .devenv/docker-compose.yml.template > .devenv/docker-compose.yml

# -----------------------------
# Detect distro (Debian/Ubuntu)
# -----------------------------
DISTRO_ID=$(grep '^ID=' /etc/os-release | cut -d= -f2)
DISTRO_VERSION=$(grep '^VERSION_ID=' /etc/os-release | cut -d= -f2 | tr -d '"')

echo "Detected distro: $DISTRO_ID $DISTRO_VERSION"
echo

# -----------------------------
# Function: install NVIDIA repo
# -----------------------------
install_nvidia_repo() {
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt update
}

# -----------------------------
# GPU CHECK 1: nvidia-smi in WSL
# -----------------------------
echo "Checking GPU availability in WSL..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ nvidia-smi not found in WSL."
    echo "→ Install the NVIDIA WSL2 driver on Windows."
    exit 1
fi

nvidia-smi || {
    echo "❌ nvidia-smi failed inside WSL."
    echo "→ GPU is not exposed to WSL. Reboot or reinstall WSL2 NVIDIA driver."
    exit 1
}
echo "✅ GPU detected in WSL."
echo

# -----------------------------
# GPU CHECK 2: NVIDIA container runtime
# -----------------------------
echo "Checking NVIDIA container runtime..."
if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
    echo "⚠️ NVIDIA container runtime missing. Installing..."

    install_nvidia_repo

    sudo apt install -y nvidia-container-toolkit

    echo "Configuring Docker runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker

    echo "✅ NVIDIA container toolkit installed."
    echo "Please restart Docker Desktop and re-run this script."
    exit 0
fi

echo "✅ NVIDIA container runtime installed."
echo

# -----------------------------
# GPU CHECK 3: Docker GPU support
# -----------------------------
echo "Checking Docker GPU support..."
if ! docker info | grep -qi nvidia; then
    echo "❌ Docker does not list NVIDIA runtime."
    echo "→ Enable GPU support in Docker Desktop:"
    echo "   Settings → Resources → WSL Integration → Enable GPU"
    exit 1
fi
echo "✅ Docker recognizes NVIDIA runtime."
echo

# -----------------------------
# GPU CHECK 4: Test GPU passthrough
# -----------------------------
echo "Testing GPU passthrough with CUDA container..."
if ! docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "❌ Docker cannot access GPU."
    echo "→ Likely missing toolkit or Docker misconfiguration."
    exit 1
fi
echo "✅ GPU passthrough works."
echo

# -----------------------------
# Build container
# -----------------------------
echo "Building development container..."
docker compose -f .devenv/docker-compose.yml build \
    --build-arg HOST_UID="$HOST_UID" \
    --build-arg HOST_GID="$HOST_GID" \
    --build-arg USERNAME="$USERNAME"

# -----------------------------
# Check container
# -----------------------------
echo "Testing user and GPU..."
echo "- uname -a"
docker run --gpus all -it devenv-ml_devenv:latest uname -a
echo "- whoami"
docker run --gpus all -it devenv-ml_devenv:latest whoami
echo "- torch.cuda.is_available()"
docker run --gpus all -it devenv-ml_devenv:latest python -c "import torch;print(torch.cuda.is_available())"

# -----------------------------
# Run container with GPU
# -----------------------------
echo "Launching container with GPU..."
docker compose  -f .devenv/docker-compose.yml up -d
docker compose  -f .devenv/docker-compose.yml exec ml_devenv bash
