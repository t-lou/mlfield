#!/bin/bash

# Verify the cross-platform dev environment setup
# Run this to check that all components are properly installed

echo "🔍 Verifying cross-platform development environment..."
echo

# Check required files
REQUIRED_FILES=(
    "launch.sh"
    "setup_mac.sh"
    "setup_linux.sh"
    "generate_docker_compose.sh"
    "Dockerfile.cpu"
    "Dockerfile.cuda"
    "README.md"
    "devcontainer.json"
)

echo "📋 Checking files..."
MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done
echo

# Check script permissions
echo "🔐 Checking script permissions..."
for script in launch.sh setup_mac.sh setup_linux.sh generate_docker_compose.sh; do
    if [ -x "$script" ]; then
        echo "  ✅ $script is executable"
    else
        echo "  ⚠️  $script is not executable (fixing...)"
        chmod +x "$script"
    fi
done
echo

# Detect current OS
echo "🖥️  Detecting current OS..."
OS_TYPE=$(uname -s)
case "$OS_TYPE" in
  Darwin)
    OS_NAME="macOS"
    EXPECTED_DOCKERFILE="Dockerfile.cpu"
    ;;
  Linux)
    OS_NAME="Linux"
    EXPECTED_DOCKERFILE="Dockerfile.cuda"
    ;;
  *)
    echo "  ❌ Unknown OS: $OS_TYPE"
    exit 1
    ;;
esac
echo "  ✅ Detected: $OS_NAME"
echo "  → Expected to use: $EXPECTED_DOCKERFILE"
echo

# Check Docker
echo "🐳 Checking Docker installation..."
if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version)
    echo "  ✅ Docker installed: $DOCKER_VERSION"
else
    echo "  ❌ Docker not found"
    exit 1
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_VERSION=$(docker compose version --short)
    echo "  ✅ Docker Compose installed: $COMPOSE_VERSION"
else
    echo "  ❌ Docker Compose not found"
    exit 1
fi
echo

# Check if Docker daemon is running
echo "⚙️  Checking Docker daemon..."
if docker ps >/dev/null 2>&1; then
    echo "  ✅ Docker daemon is running"
else
    echo "  ❌ Docker daemon is not running"
    echo "     → Start Docker Desktop and try again"
    exit 1
fi
echo

# Platform-specific checks
if [ "$OS_NAME" = "macOS" ]; then
    echo "🍎 macOS-specific checks..."
    
    DOCKER_MEMORY=$(docker info --format '{{json .}}' 2>/dev/null | grep -o '"MemTotal":[0-9]*' | cut -d: -f2)
    if [ ! -z "$DOCKER_MEMORY" ]; then
        DOCKER_MEMORY_GB=$((DOCKER_MEMORY / 1024 / 1024 / 1024))
        echo "  ℹ️  Docker memory: ${DOCKER_MEMORY_GB}GB"
        if [ "$DOCKER_MEMORY_GB" -lt 4 ]; then
            echo "  ⚠️  Warning: Less than 4GB (consider increasing in Docker Desktop preferences)"
        else
            echo "  ✅ Memory allocation is sufficient"
        fi
    fi
    echo "  ℹ️  PyTorch will run in CPU mode (no GPU support on macOS)"
    echo
    
elif [ "$OS_NAME" = "Linux" ]; then
    echo "🐧 Linux-specific checks..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  ✅ NVIDIA GPU tools found"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/    → /'
    else
        echo "  ⚠️  nvidia-smi not found (GPU not available)"
    fi
    
    if command -v nvidia-container-runtime >/dev/null 2>&1; then
        echo "  ✅ NVIDIA container runtime installed"
    else
        echo "  ⚠️  NVIDIA container runtime not found"
        echo "     → Run: ./.devcontainer/launch.sh (will install if needed)"
    fi
    echo
fi

# Summary
echo "================================================"
if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ All checks passed! Environment is ready."
    echo
    echo "   To start: ./.devcontainer/launch.sh"
else
    echo "❌ $MISSING_FILES file(s) missing. Please check setup."
    exit 1
fi
echo "================================================"
