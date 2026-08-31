# Devcontainer base-image and runtime policy

This folder is intentionally kept lean and aligned with a single policy:

- prefer CUDA on Linux/WSL when NVIDIA is available
- fall back to CPU when GPU is unavailable
- support macOS via a CPU-only base image
- keep the same host user and group identity in the build and runtime config
- preserve a base-image build flow for easy local rebuilds

## Active files

- launch.sh: picks the target OS and base image, then starts the devcontainer
- create_base_container.sh: builds the shared base images for CUDA or CPU
- generate_docker_compose.sh: generates the runtime compose file with the host UID/GID/user/group
- Dockerfile: creates the final devcontainer user matching the host identity
- Dockerfile.cuda.base: CUDA base image for Linux/WSL
- Dockerfile.cpu.base: CPU base image for macOS or no-GPU fallback
- devcontainer.json: VS Code devcontainer metadata
- setup_linux.sh: Linux/WSL GPU validation
- setup_mac.sh: macOS sanity checks
- detect-platform.sh: shared platform and GPU detector

## Runtime behavior

- Linux + NVIDIA: use `mlfield_cuda_base:latest`
- Linux without NVIDIA: use `mlfield_cpu_base:latest`
- macOS: use `mlfield_cpu_base:latest`

## Important note

The generated compose file should use the host user and group from the launcher, not a hardcoded username, so files created from inside the container retain the correct ownership on the host.
