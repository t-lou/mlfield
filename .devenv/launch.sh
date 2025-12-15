#!/bin/bash

# Get the directory where the script itself is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR" || exit

echo "Now in parent folder: $(pwd)"

# Name of the service defined in docker-compose.yml
SERVICE_NAME="mlfield"

# Export UID/GID for Compose
export UID=$(id -u)
export GID=$(id -g)

export WS_DIR=$(realpath "$(pwd)")

# Build and start the container
docker compose up -d --build

# # Attach to the container shell
docker compose exec $SERVICE_NAME bash