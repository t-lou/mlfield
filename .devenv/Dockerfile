FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# RUN pip3 install torch==2.9.0+cu126 --no-cache-dir \
#     --index-url https://download.pytorch.org/whl/cu126

# Default working directory
WORKDIR /workspace

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


# 3. Set the new user as the default user for the container
USER $USERNAME
