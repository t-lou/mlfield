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

# 1. Define Build Arguments with defaults (e.g., 1000 is common for non-root users)
ARG HOST_UID
ARG HOST_GID
ARG USERNAME

# 2. Add a new group and user using the passed-in IDs
RUN groupadd -g $HOST_GID $USERNAME || true \
    && useradd -m -u $HOST_UID -g $HOST_GID $USERNAME || true

# Ensure home directory is owned by the new user
RUN chown -R $HOST_UID:$HOST_GID /home/$USERNAME \
    && chmod -R 755 /home/$USERNAME

# 3. Set the new user as the default user for the container
USER $USERNAME