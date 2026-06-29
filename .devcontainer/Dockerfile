ARG BASE_IMAGE
FROM tlou/${BASE_IMAGE}

# -----------------------------
# Create user matching host
# -----------------------------
ARG HOST_UID
ARG HOST_GID
ARG USERNAME

RUN if getent group "$HOST_GID" >/dev/null; then \
        groupmod -n "$USERNAME" "$(getent group "$HOST_GID" | cut -d: -f1)"; \
    else \
        groupadd -g "$HOST_GID" "$USERNAME"; \
    fi

RUN if getent passwd "$HOST_UID" >/dev/null; then \
        usermod -l "$USERNAME" -d "/home/$USERNAME" -m "$(getent passwd "$HOST_UID" | cut -d: -f1)"; \
    else \
        useradd -m -u "$HOST_UID" -g "$HOST_GID" -s /bin/bash "$USERNAME"; \
    fi

# Check ownership of the virtual environment and fix it if necessary
RUN owner_uid=$(stat -c "%u" /venv/bin/activate) && \
    owner_gid=$(stat -c "%g" /venv/bin/activate) && \
    if [ "$owner_uid" != "$HOST_UID" ] || [ "$owner_gid" != "$HOST_GID" ]; then \
        echo "Fixing venv ownership..."; \
        chown -R "$HOST_UID:$HOST_GID" /venv; \
    else \
        echo "Venv ownership already correct, skipping chown."; \
    fi


USER "$USERNAME"
ENV HOME=/home/$USERNAME

WORKDIR /repo

CMD ["bash"]
