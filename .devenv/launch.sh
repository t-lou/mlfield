#!/bin/bash

# Get the directory where the script itself is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname ${SCRIPT_DIR})"

cd ${WS_DIR}

# Export UID/GID for Compose
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export USERNAME="$(whoami)"

export WS_DIR=${WS_DIR}

# Build the container
docker compose -f .devenv/docker-compose.yml build \
  --build-arg HOST_UID=${HOST_UID} \
  --build-arg HOST_GID=${HOST_GID} \
  --build-arg USERNAME=${USERNAME}

docker run -it devenv-ml_devenv:latest bash
