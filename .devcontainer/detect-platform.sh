#!/bin/bash
# Auto-detect platform and GPU availability to set BASE_IMAGE

OS_TYPE=$(uname -s)

case "$OS_TYPE" in
  Darwin)
    echo "Detected macOS - using CPU base image"
    export BASE_IMAGE="mlfield_cpu_base:latest"
    ;;
  Linux)
    # Check for NVIDIA GPU on Linux
    if command -v nvidia-smi &> /dev/null || [ -e /dev/nvidia0 ]; then
      echo "Detected Linux with NVIDIA GPU - using CUDA base image"
      export BASE_IMAGE="mlfield_cuda_base:latest"
    else
      echo "Detected Linux without GPU - using CPU base image"
      export BASE_IMAGE="mlfield_cpu_base:latest"
    fi
    ;;
  *)
    echo "Unknown OS: $OS_TYPE - defaulting to CUDA base image"
    export BASE_IMAGE="mlfield_cuda_base:latest"
    ;;
esac

echo "BASE_IMAGE=$BASE_IMAGE"
