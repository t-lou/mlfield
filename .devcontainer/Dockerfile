# Start from NVIDIA's official CUDA base image
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python-is-python3 \
        git vim tree wget direnv curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Remove the "externally managed" marker
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Install PyTorch with CUDA 12.9 support
# (this pulls the correct wheel from PyTorch's official index)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Optional: common ML packages
RUN pip3 install --no-cache-dir numpy scipy scikit-learn matplotlib jupyterlab notebook pytest

# Install new deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

# 1. Define Build Arguments with defaults (e.g., 1000 is common for non-root users)
ARG HOST_UID
ARG HOST_GID
ARG USERNAME

# Rename the existing user/group if UID/GID already exist
RUN if getent passwd $HOST_UID; then \
        usermod -l $USERNAME -d /home/$USERNAME -m $(getent passwd $HOST_UID | cut -d: -f1); \
    else \
        groupadd -g $HOST_GID $USERNAME && \
        useradd -m -u $HOST_UID -g $HOST_GID -s /bin/bash $USERNAME; \
    fi

# For OpenCV and other graphical libraries, we need to install some additional dependencies
RUN apt-get update && apt-get install libgl1 libglib2.0-0 python3.10-venv -y && rm -rf /var/lib/apt/lists/*

# 3. Set the new user as the default user for the container
USER $USERNAME
ENV HOME=/home/$USERNAME
RUN echo 'eval "$(direnv hook bash)"' >> $HOME/.bashrc

CMD ["bash"]
