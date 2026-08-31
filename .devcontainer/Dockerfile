ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HOST_UID
ARG HOST_GID
ARG HOST_USER
ARG HOST_GROUP

RUN set -eux; \
    if getent group "$HOST_GID" >/dev/null 2>&1; then \
        existing_group="$(getent group "$HOST_GID" | cut -d: -f1)"; \
        if [ "$existing_group" != "$HOST_GROUP" ]; then groupmod -n "$HOST_GROUP" "$existing_group"; fi; \
    else \
        groupadd -g "$HOST_GID" "$HOST_GROUP"; \
    fi; \
    if getent passwd "$HOST_UID" >/dev/null 2>&1; then \
        existing_user="$(getent passwd "$HOST_UID" | cut -d: -f1)"; \
        if [ "$existing_user" != "$HOST_USER" ]; then \
            usermod -l "$HOST_USER" -g "$HOST_GROUP" -d "/home/$HOST_USER" -m "$existing_user"; \
        else \
            usermod -g "$HOST_GROUP" "$existing_user"; \
        fi; \
    else \
        useradd -m -u "$HOST_UID" -g "$HOST_GROUP" -s /bin/bash "$HOST_USER"; \
    fi; \
    mkdir -p "/home/$HOST_USER"; \
    chown -R "$HOST_UID:$HOST_GID" "/home/$HOST_USER"; \
    if [ -d /venv ]; then \
        owner_uid="$(stat -c '%u' /venv/bin/activate)"; \
        owner_gid="$(stat -c '%g' /venv/bin/activate)"; \
        if [ "$owner_uid" != "$HOST_UID" ] || [ "$owner_gid" != "$HOST_GID" ]; then \
            echo "Fixing venv ownership..."; \
            chown -R "$HOST_UID:$HOST_GID" /venv; \
        else \
            echo "Venv ownership already correct, skipping chown."; \
        fi; \
    fi

WORKDIR /repo

SHELL ["/bin/bash", "-lc"]
USER ${HOST_USER}
