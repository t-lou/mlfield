#!/bin/bash
set -e

# Generate docker-compose.yml based on the platform
# This script creates a properly formatted docker-compose file for the current OS

OS_NAME="${1:-}"
WS_DIR="${2:-}"
HOST_UID="${3:-}"
HOST_GID="${4:-}"
USERNAME="${5:-}"
DISPLAY="${6:-}"
DATASET_DIR="${7:-}"
DOCKER_COMPOSE_OUTPUT="${8:-.devcontainer/docker-compose.yml}"

if [ -z "$OS_NAME" ] || [ -z "$WS_DIR" ] || [ -z "$HOST_UID" ] || [ -z "$HOST_GID" ] || [ -z "$USERNAME" ]; then
    echo "Usage: generate_docker_compose.sh <OS_NAME> <WS_DIR> <HOST_UID> <HOST_GID> <USERNAME> [DISPLAY] [DATASET_DIR] [OUTPUT_FILE]"
    exit 1
fi

# Detect NVIDIA runtime name
if docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
    RUNTIME="nvidia"
else
    RUNTIME=""
fi

# Select Dockerfile
if [ "$OS_NAME" = "macOS" ]; then
    RUNTIME_CONFIG=""
    NETWORK_CONFIG=""
    X11_CONFIG=""
    BASE_IMAGE="mlfield_cpu_base:latest"
else
    NETWORK_CONFIG="    network_mode: host"
    X11_CONFIG="      - /tmp/.X11-unix:/tmp/.X11-unix"
    BASE_IMAGE="mlfield_cuda_base:latest"
    if [ -n "$RUNTIME" ]; then
        RUNTIME_CONFIG="    runtime: $RUNTIME"
    else
        echo "⚠️ No NVIDIA runtime detected — running CPU-only container."
        RUNTIME_CONFIG=""
        BASE_IMAGE="mlfield_cpu_base:latest"
    fi
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
        USERNAME: $USERNAME
        BASE_IMAGE: $BASE_IMAGE
    container_name: mlfield-${USERNAME}
    shm_size: "2gb"
    volumes:
      - ${WS_DIR}:/repo
      - mlfield-home-${USERNAME}:/home/${USERNAME}
      - ${DATASET_DIR:-/dev/null}:/mnt/dataset:ro
$X11_CONFIG
    environment:
      - DISPLAY=${DISPLAY}
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - HOST_UID=${HOST_UID}
      - HOST_GID=${HOST_GID}
      - USERNAME=${USERNAME}
      - PYTHONUNBUFFERED=1
      - PIP_DISABLE_PIP_VERSION_CHECK=1
    user: "${USERNAME}"
    stdin_open: true
    tty: true
$RUNTIME_CONFIG
$NETWORK_CONFIG
    command: bash

volumes:
  mlfield-home-${USERNAME}:
    driver: local
EOF

echo "✅ Generated: $DOCKER_COMPOSE_OUTPUT (using $DOCKERFILE)"
