#!/bin/bash
set -euo pipefail

OS_NAME="${1:-}"
WS_DIR="${2:-}"
HOST_UID="${3:-}"
HOST_GID="${4:-}"
HOST_USER="${5:-}"
HOST_GROUP="${6:-}"
DISPLAY="${7:-}"
DATASET_DIR="${8:-}"
DOCKER_COMPOSE_OUTPUT="${9:-.devcontainer/docker-compose.yml}"
BASE_IMAGE="${10:-mlfield_cuda_base:latest}"

if [ -z "$OS_NAME" ] || [ -z "$WS_DIR" ] || [ -z "$HOST_UID" ] || [ -z "$HOST_GID" ] || [ -z "$HOST_USER" ]; then
    echo "Usage: generate_docker_compose.sh <OS_NAME> <WS_DIR> <HOST_UID> <HOST_GID> <HOST_USER> [HOST_GROUP] [DISPLAY] [DATASET_DIR] [OUTPUT_FILE] [BASE_IMAGE]"
    exit 1
fi

if [ -z "$HOST_GROUP" ]; then
    HOST_GROUP="$HOST_USER"
fi

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    RUNTIME="nvidia"
else
    RUNTIME=""
fi

if [ "$OS_NAME" = "macOS" ]; then
    RUNTIME_CONFIG=""
    NETWORK_CONFIG=""
    X11_CONFIG=""
    DOCKERFILE_BASE="${BASE_IMAGE:-mlfield_cpu_base:latest}"
elif [ "$BASE_IMAGE" = "mlfield_cuda_base:latest" ] && [ -n "$RUNTIME" ]; then
    RUNTIME_CONFIG="    runtime: $RUNTIME"
    NETWORK_CONFIG="    network_mode: host"
    X11_CONFIG="      - /tmp/.X11-unix:/tmp/.X11-unix"
    DOCKERFILE_BASE="$BASE_IMAGE"
else
    echo "⚠️ No CUDA runtime available — running CPU-only container."
    RUNTIME_CONFIG=""
    NETWORK_CONFIG="    network_mode: host"
    X11_CONFIG="      - /tmp/.X11-unix:/tmp/.X11-unix"
    DOCKERFILE_BASE="mlfield_cpu_base:latest"
fi

cat > "$DOCKER_COMPOSE_OUTPUT" <<EOF
name: mlfield

services:
  mlfield:
    image: mlfield:latest
    build:
      context: .
      dockerfile: Dockerfile
      network: host
      args:
        HOST_UID: $HOST_UID
        HOST_GID: $HOST_GID
        HOST_USER: $HOST_USER
        HOST_GROUP: $HOST_GROUP
        BASE_IMAGE: $DOCKERFILE_BASE
    container_name: mlfield-${HOST_USER}
    shm_size: "2gb"
    volumes:
      - ${WS_DIR}:/repo
      - mlfield-home-${HOST_USER}:/home/${HOST_USER}
      - ${DATASET_DIR:-/dev/null}:/mnt/dataset:ro
$X11_CONFIG
    environment:
      - DISPLAY=${DISPLAY}
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - HOST_UID=${HOST_UID}
      - HOST_GID=${HOST_GID}
      - HOST_USER=${HOST_USER}
      - HOST_GROUP=${HOST_GROUP}
      - PYTHONUNBUFFERED=1
      - PIP_DISABLE_PIP_VERSION_CHECK=1
    user: "${HOST_UID}:${HOST_GID}"
    stdin_open: true
    tty: true
$RUNTIME_CONFIG
$NETWORK_CONFIG
    command: bash

volumes:
  mlfield-home-${HOST_USER}:
    driver: local
EOF

echo "✅ Generated: $DOCKER_COMPOSE_OUTPUT (base: $DOCKERFILE_BASE)"
