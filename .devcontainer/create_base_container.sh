#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

eval "$($BASE_DIR/detect-platform.sh --shell)"

build_cuda=false
build_cpu=false

if [ "${1:-}" = "--cuda" ]; then
  build_cuda=true
elif [ "${1:-}" = "--cpu" ]; then
  build_cpu=true
elif [ "${1:-}" = "--all" ]; then
  build_cuda=true
  build_cpu=true
else
  if [ "$BASE_IMAGE" = "mlfield_cuda_base:latest" ]; then
    echo "✔ NVIDIA GPU detected — building CUDA base image."
    build_cuda=true
  else
    echo "✘ No NVIDIA GPU detected — building CPU base image."
    build_cpu=true
  fi
fi

if [ "$build_cuda" = true ]; then
  echo "Building CUDA base image..."
  docker build -f Dockerfile.cuda.base -t mlfield_cuda_base:latest .
fi

if [ "$build_cpu" = true ]; then
  echo "Building CPU base image..."
  docker build -f Dockerfile.cpu.base -t mlfield_cpu_base:latest .
fi

echo "Base image build complete."
