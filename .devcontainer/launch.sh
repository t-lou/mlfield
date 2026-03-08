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
# Detect OS
# -----------------------------
OS_TYPE=$(uname -s)
case "$OS_TYPE" in
  Darwin)
    OS_NAME="macOS"
    ;;
  Linux)
    OS_NAME="Linux"
    ;;
  *)
    echo "❌ Unsupported OS: $OS_TYPE"
    exit 1
    ;;
esac

echo "Detected OS: $OS_NAME"
export OS_NAME="$OS_NAME"

# Detect Linux distro if applicable
if [ "$OS_NAME" = "Linux" ]; then
  DISTRO_ID=$(grep '^ID=' /etc/os-release | cut -d= -f2)
  DISTRO_VERSION=$(grep '^VERSION_ID=' /etc/os-release | cut -d= -f2 | tr -d '"')
  echo "Detected distro: $DISTRO_ID $DISTRO_VERSION"
fi
echo

# Set platform-specific docker parameters
if [ "$OS_NAME" = "macOS" ]; then
  export DOCKER_RUNTIME=""  # No GPU runtime on macOS
  export DOCKER_X11_VOLUME=""  # No X11 on macOS
  export DOCKER_NETWORK_MODE=""  # Use default bridge network
else
  export DOCKER_RUNTIME="runtime: nvidia"
  export DOCKER_X11_VOLUME="- /tmp/.X11-unix:/tmp/.X11-unix"
  export DOCKER_NETWORK_MODE="network_mode: host"
fi

# Generate the docker-compose.yml based on platform
# =============================
bash ./.devcontainer/generate_docker_compose.sh "$OS_NAME" "$WS_DIR" "$HOST_UID" "$HOST_GID" "$USERNAME" "$DISPLAY" "${DATASET_DIR:-}" ".devcontainer/docker-compose.yml"

# Load optional local overrides
if [ -f ".devcontainer/local.env" ]; then
    echo "Loading local overrides from .devcontainer/local.env"
    set -a
    source .devcontainer/local.env
    set +a
else
    echo "No local.env found, using defaults"
fi
echo

# =============================
# PLATFORM-SPECIFIC SETUP
# =============================

if [ "$OS_NAME" = "macOS" ]; then
    echo "📱 Running macOS-specific setup..."
    bash ./.devcontainer/setup_mac.sh
elif [ "$OS_NAME" = "Linux" ]; then
    echo "🐧 Running Linux-specific setup..."
    bash ./.devcontainer/setup_linux.sh
fi
echo

# -----------------------------
# Build container
# -----------------------------
echo "Building development container..."
docker compose -f .devcontainer/docker-compose.yml build \
    --build-arg HOST_UID="$HOST_UID" \
    --build-arg HOST_GID="$HOST_GID" \
    --build-arg USERNAME="$USERNAME"
docker compose  -f .devcontainer/docker-compose.yml up -d

# -----------------------------
# Check container
# -----------------------------
echo "Testing user and GPU..."
echo "- uname -a"
docker compose -f .devcontainer/docker-compose.yml exec mlfield uname -a
echo "- whoami"
docker compose -f .devcontainer/docker-compose.yml exec mlfield whoami
echo "- torch.cuda.is_available()"
docker compose -f .devcontainer/docker-compose.yml exec mlfield python3 -c "import torch;print(torch.cuda.is_available())" || {
    echo "⚠️ CUDA not available (expected on macOS without GPU)"
}

# Run container
echo "Launching container..."
docker compose -f .devcontainer/docker-compose.yml exec mlfield bash
