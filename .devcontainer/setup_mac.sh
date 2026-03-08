#!/bin/bash
set -e

# This script handles Docker setup for macOS
# macOS does NOT support GPU passthrough directly; GPU is available only to Docker Desktop

echo "Setting up for macOS..."
echo

# =============================
# Check Docker Desktop
# =============================
echo "Checking for Docker Desktop..."

if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found."
    echo "→ Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker is installed."

# Check Docker version
DOCKER_VERSION=$(docker --version)
echo "→ $DOCKER_VERSION"
echo

# =============================
# Check Docker daemon
# =============================
echo "Checking if Docker daemon is running..."
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker daemon is not running."
    echo "→ Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker daemon is running."
echo

# =============================
# GPU availability warning (informational)
# =============================
echo "ℹ️ GPU Support on macOS:"
echo "   - Docker Desktop on macOS (Apple Silicon) may have GPU acceleration"
echo "   - GPU passthrough for NVIDIA/CUDA is not supported on macOS"
echo "   - The container will run with CPU-only PyTorch by default"
if [[ $(uname -m) == "arm64" ]]; then
    echo "   - Detected Apple Silicon (M1/M2/M3) - Metal acceleration available"
else
    echo "   - Detected Intel Mac - CPU-only mode"
fi
echo

# =============================
# Check Docker resources
# =============================
echo "Checking Docker resources..."
DOCKER_MEMORY=$(docker info --format '{{json .}}' | grep -o '"MemTotal":[0-9]*' | cut -d: -f2)
if [ ! -z "$DOCKER_MEMORY" ]; then
    DOCKER_MEMORY_GB=$((DOCKER_MEMORY / 1024 / 1024 / 1024))
    echo "→ Docker available memory: ${DOCKER_MEMORY_GB}GB"
    if [ "$DOCKER_MEMORY_GB" -lt 4 ]; then
        echo "⚠️ Warning: Docker has less than 4GB available"
        echo "   Consider increasing memory in Docker Desktop settings"
    fi
else
    echo "⚠️ Could not determine Docker memory"
fi
echo

# =============================
# Docker compose check
# =============================
echo "Checking Docker Compose..."
if ! docker compose version >/dev/null 2>&1; then
    echo "❌ Docker Compose not found."
    echo "→ Please ensure Docker Desktop is up to date"
    exit 1
fi

COMPOSE_VERSION=$(docker compose version --short)
echo "✅ Docker Compose available (version $COMPOSE_VERSION)"
echo

echo "✅ macOS setup complete!"
