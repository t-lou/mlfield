#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WS_DIR"

eval "$($SCRIPT_DIR/detect-platform.sh --shell)"

export WS_DIR

echo "Workspace: $WS_DIR"
echo "User: $HOST_USER ($HOST_UID:$HOST_GID) / group: $HOST_GROUP ($HOST_GID)"
echo

OS_TYPE="$(uname -s)"
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

export OS_NAME

echo "Detected OS: $OS_NAME"
if [ "$OS_NAME" = "Linux" ]; then
  DISTRO_ID="$(grep '^ID=' /etc/os-release | cut -d= -f2)"
  DISTRO_VERSION="$(grep '^VERSION_ID=' /etc/os-release | cut -d= -f2 | tr -d '"')"
  echo "Detected distro: $DISTRO_ID $DISTRO_VERSION"
fi

echo

export DOCKER_RUNTIME=""
export DOCKER_X11_VOLUME=""
export DOCKER_NETWORK_MODE=""

if [ "$OS_NAME" = "Linux" ]; then
  if [ "$BASE_IMAGE" = "mlfield_cuda_base:latest" ]; then
    export DOCKER_RUNTIME="runtime: nvidia"
    export DOCKER_X11_VOLUME="- /tmp/.X11-unix:/tmp/.X11-unix"
    export DOCKER_NETWORK_MODE="network_mode: host"
  else
    export DOCKER_RUNTIME=""
    export DOCKER_X11_VOLUME="- /tmp/.X11-unix:/tmp/.X11-unix"
    export DOCKER_NETWORK_MODE="network_mode: host"
  fi
fi

bash ./.devcontainer/generate_docker_compose.sh \
  "$OS_NAME" \
  "$WS_DIR" \
  "$HOST_UID" \
  "$HOST_GID" \
  "$HOST_USER" \
  "$HOST_GROUP" \
  "$DISPLAY" \
  "${DATASET_DIR:-}" \
  ".devcontainer/docker-compose.yml" \
  "$BASE_IMAGE"

if [ -f ".devcontainer/local.env" ]; then
    echo "Loading local overrides from .devcontainer/local.env"
    set -a
    source .devcontainer/local.env
    set +a
else
    echo "No local.env found, using defaults"
fi

echo

if [ "$OS_NAME" = "macOS" ]; then
    echo "📱 Running macOS-specific setup..."
    bash ./.devcontainer/setup_mac.sh
elif [ "$OS_NAME" = "Linux" ]; then
    echo "🐧 Running Linux-specific setup..."
    bash ./.devcontainer/setup_linux.sh
fi

echo

echo "Ensuring the required local base image exists: $BASE_IMAGE"
bash ./.devcontainer/create_base_container.sh

echo "Building development container from $BASE_IMAGE..."
docker compose -f .devcontainer/docker-compose.yml build \
    --build-arg HOST_UID="$HOST_UID" \
    --build-arg HOST_GID="$HOST_GID" \
    --build-arg HOST_USER="$HOST_USER" \
    --build-arg HOST_GROUP="$HOST_GROUP" \
    --build-arg BASE_IMAGE="$BASE_IMAGE"

docker compose -f .devcontainer/docker-compose.yml up -d

echo "Testing user and GPU..."
echo "- uname -a"
docker compose -f .devcontainer/docker-compose.yml exec mlfield uname -a
echo "- whoami"
docker compose -f .devcontainer/docker-compose.yml exec mlfield whoami
echo "- id -u -g"
docker compose -f .devcontainer/docker-compose.yml exec mlfield sh -lc 'id -u && id -g && id -un && id -gn'
echo "- torch.cuda.is_available()"
if ! docker compose -f .devcontainer/docker-compose.yml exec mlfield python3 -c "import torch; print(torch.cuda.is_available())"; then
    echo "⚠️ CUDA not available (expected on CPU fallback)"
fi

echo "Launching container, don't forget to run the following command or add to .bashrc outside the container..."
echo 'eval "$(direnv hook bash)"'
docker compose -f .devcontainer/docker-compose.yml exec mlfield bash -l
