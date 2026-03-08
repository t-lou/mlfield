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

# Select appropriate Dockerfile based on platform
if [ "$OS_NAME" = "macOS" ]; then
    DOCKERFILE="Dockerfile.cpu"
    RUNTIME_CONFIG=""
    NETWORK_CONFIG=""
    X11_CONFIG=""
else
    DOCKERFILE="Dockerfile.cuda"
    RUNTIME_CONFIG="    runtime: nvidia"
    NETWORK_CONFIG="    network_mode: host"
    X11_CONFIG="      - /tmp/.X11-unix:/tmp/.X11-unix"
fi

cat > "$DOCKER_COMPOSE_OUTPUT" <<EOF
name: mlfield

services:
  mlfield:
    build:
      context: .
      dockerfile: $DOCKERFILE
      args:
        HOST_UID: $HOST_UID
        HOST_GID: $HOST_GID
    container_name: mlfield-${USERNAME}
    shm_size: "2gb"
    volumes:
      - ${WS_DIR}:/workspace
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
