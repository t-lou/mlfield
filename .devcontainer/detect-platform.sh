#!/bin/bash
set -euo pipefail

HOST_USER="$(id -un)"
HOST_GROUP="$(id -gn)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
  Darwin)
    BASE_IMAGE="mlfield_cpu_base:latest"
    ;;
  Linux)
    if command -v nvidia-smi >/dev/null 2>&1 || ls /dev/nvidia* >/dev/null 2>&1 || docker info 2>/dev/null | grep -qi "nvidia"; then
      BASE_IMAGE="mlfield_cuda_base:latest"
    else
      BASE_IMAGE="mlfield_cpu_base:latest"
    fi
    ;;
  *)
    BASE_IMAGE="mlfield_cuda_base:latest"
    ;;
esac

export HOST_USER HOST_GROUP HOST_UID HOST_GID BASE_IMAGE

if [ "${1:-}" = "--shell" ]; then
  printf 'export HOST_USER=%q\n' "$HOST_USER"
  printf 'export HOST_GROUP=%q\n' "$HOST_GROUP"
  printf 'export HOST_UID=%q\n' "$HOST_UID"
  printf 'export HOST_GID=%q\n' "$HOST_GID"
  printf 'export BASE_IMAGE=%q\n' "$BASE_IMAGE"
  exit 0
fi

case "$OS_TYPE" in
  Darwin) echo "Detected macOS - using CPU base image" ;;
  Linux)
    if [ "$BASE_IMAGE" = "mlfield_cuda_base:latest" ]; then
      echo "Detected Linux with NVIDIA GPU - using CUDA base image"
    else
      echo "Detected Linux without GPU - using CPU base image"
    fi
    ;;
  *) echo "Unknown OS: $OS_TYPE - defaulting to CUDA base image" ;;
esac

printf 'BASE_IMAGE=%s\n' "$BASE_IMAGE"
