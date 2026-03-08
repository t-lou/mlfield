# Development Environment Setup Guide

This development environment supports **macOS**, **Linux (WSL/Ubuntu)**, and **Debian-based systems** with Docker and GPU support where applicable.

## Platform-Specific Features

### Linux (WSL2/Ubuntu/Debian)
- ✅ **GPU Support**: NVIDIA GPUs with CUDA passthrough
- ✅ **X11 Forwarding**: Display forwarding for GUI applications
- ✅ **Native Performance**: Direct hardware access
- Docker Runtime: nvidia

### macOS
- ⚠️ **GPU**: CPU-only (Metal acceleration available on Apple Silicon)
- ❌ **X11**: Not available (headless mode or VNC alternative needed)
- ✅ **Easy Setup**: Works with Docker Desktop
- Docker Runtime: default

## Quick Start

### Prerequisites

#### macOS
- **Docker Desktop**: [Download](https://www.docker.com/products/docker-desktop)
- Minimum 4GB RAM allocated to Docker
- macOS 11 or later

#### Linux/WSL2
- **Docker**: Install via package manager or [official guide](https://docs.docker.com/engine/install/)
- **NVIDIA Docker**: For GPU support
- **NVIDIA Driver**: Installed on host

### Launch Development Container

```bash
cd mlfield
./.devcontainer/launch.sh
```

The script will:
1. ✅ Detect your OS (macOS/Linux)
2. ✅ Run platform-specific setup checks
3. ✅ Generate appropriate docker-compose configuration
4. ✅ Build the container
5. ✅ Launch an interactive bash session

## Project Structure

```
.devcontainer/
├── launch.sh                      # Main entry point (cross-platform)
├── setup_mac.sh                   # macOS-specific checks
├── setup_linux.sh                 # Linux GPU setup
├── generate_docker_compose.sh     # Generates docker-compose.yml
├── Dockerfile.cuda                # Linux with CUDA/GPU support
├── Dockerfile.cpu                 # macOS with CPU-only PyTorch
├── docker-compose.yml.template    # (Legacy, kept for reference)
├── install_docker.sh              # Linux Docker installation helper
└── local.env                      # Optional local configuration (git-ignored)
```

## Configuration

### Optional: Local Environment Overrides

Create `.devcontainer/local.env` to customize variables:

```bash
# Example local.env
DATASET_DIR=/path/to/your/datasets
DISPLAY=:0
```

Available variables:
- `DATASET_DIR`: Mount point for datasets (read-only at `/mnt/dataset`)
- `DISPLAY`: X11 display for Linux only
- Any other environment variables

## Platform-Specific Details

### macOS

**What to expect:**
- No GPU acceleration (PyTorch runs on CPU)
- Apple Silicon Macs support Metal optimizations
- No display forwarding needed

**If you have datasets:**
```bash
# In .devcontainer/local.env
DATASET_DIR=/Users/you/path/to/datasets
```

**Inside container:** Available at `/mnt/dataset`

**Docker Desktop settings:**
- Recommended: 8+ GB memory
- At least 4 CPU cores
- 20+ GB disk space

### Linux/WSL2

**What to expect:**
- Full GPU support with CUDA
- Display forwarding works (if X server configured)
- Native performance

**GPU Setup:**
The script `setup_linux.sh` will:
1. Check for `nvidia-smi` on host
2. Verify NVIDIA container runtime
3. Test GPU passthrough with CUDA container
4. Install missing components if needed

**If setup fails:**
```bash
# Manual GPU setup on Linux
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Then re-run
./.devcontainer/launch.sh
```

## Inside the Container

### Available Tools
- Python 3 with PyTorch, NumPy, SciPy, scikit-learn
- Matplotlib, JupyterLab, Jupyter Notebook
- Git, Vim, Curl, and build tools
- Graphviz for graph visualization

### Test GPU (Linux only)
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Mount Points
- `/workspace`: Your mlfield directory
- `/home/{username}`: Persistent home directory
- `/mnt/dataset`: Your datasets (if DATASET_DIR set)

### Mount New Directories

Edit generated `docker-compose.yml`:
```yaml
volumes:
  - /host/path:/container/path
```

Then restart:
```bash
docker compose -f .devcontainer/docker-compose.yml up -d
./.devcontainer/launch.sh
```

## Troubleshooting

### macOS: "Docker daemon is not running"
→ Start Docker Desktop from Applications

### Linux: "nvidia-smi not found in host"
→ Install NVIDIA driver: `sudo ubuntu-drivers autoinstall`

### Linux: "Docker does not list NVIDIA runtime"
→ Run `setup_linux.sh` which will install it, then reboot

### Container won't start
→ Check docker compose logs:
```bash
docker compose -f .devcontainer/docker-compose.yml logs mlfield
```

### Permission issues on Linux
→ The script uses your host UID/GID automatically. If issues persist:
```bash
docker compose -f .devcontainer/docker-compose.yml exec mlfield bash
# Now inside, check permissions
ls -la /workspace/
```

## Advanced: Manual Container Control

```bash
# Build only
docker compose -f .devcontainer/docker-compose.yml build

# Start in background
docker compose -f .devcontainer/docker-compose.yml up -d

# Execute command
docker compose -f .devcontainer/docker-compose.yml exec mlfield python3 script.py

# View logs
docker compose -f .devcontainer/docker-compose.yml logs -f mlfield

# Stop
docker compose -f .devcontainer/docker-compose.yml down

# Rebuild (useful after Dockerfile changes)
docker compose -f .devcontainer/docker-compose.yml build --no-cache
docker compose -f .devcontainer/docker-compose.yml up -d
```

## VS Code Remote Containers (Optional)

To use VS Code remote containers:

1. Install [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open mlfield in VS Code
3. Press `Cmd+Shift+P` → "Remote-Containers: Reopen in Container"

Note: The `devcontainer.json` is configured to use the generated docker-compose file.

## File Persistence

- `mlfield-home-{username}` volume: Home directory changes persist between container restarts
- `/workspace`: Synced with your host directory
- Other files created in container root are lost when container stops

## Updating the Environment

### Add new Python packages

**Quick (inside container):**
```bash
pip3 install new-package
```

**Persistent (rebuild Dockerfile):**
Add to `Dockerfile.cuda` or `Dockerfile.cpu`, then rebuild:
```bash
docker compose -f .devcontainer/docker-compose.yml build --no-cache
```

### Update base images

Edit `Dockerfile.cuda` or `Dockerfile.cpu` as needed, then rebuild.

## Notes

- The `docker-compose.yml` file is **generated dynamically** by `generate_docker_compose.sh` - do not edit it directly
- `docker-compose.yml.template` is kept for reference but is no longer used
- To switch configurations, delete the generated `docker-compose.yml` and run `launch.sh` again
