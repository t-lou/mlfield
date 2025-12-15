FROM ubuntu:24.04

# Install Python3 and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip git vim tree && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version for simplicity)
RUN pip3 install torch==2.9.0+cu126 --no-cache-dir --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu126

# Default working directory
WORKDIR /workspace