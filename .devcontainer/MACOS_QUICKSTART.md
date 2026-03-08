# Quick Start - macOS

Your development environment is now ready for **macOS**! 🍎

## Prerequisites (5 minutes)

1. **Install Docker Desktop**
   - Download: https://www.docker.com/products/docker-desktop
   - Open and complete the setup wizard
   - Wait for "Docker is running" to appear in menu bar

2. **Verify Docker is working**
   ```bash
   docker --version
   docker compose version
   ```

## Launch Development Environment

```bash
cd mlfield
./.devcontainer/launch.sh
```

That's it! The script will:
- ✅ Verify Docker is running
- ✅ Generate configuration for macOS
- ✅ Build the container (first run takes 2-5 minutes)
- ✅ Start the container
- ✅ Open a bash shell

## Inside the Container

You now have a complete ML development environment:

```bash
# Test Python & PyTorch
python3 -c "import torch; print(torch.__version__)"

# Install packages
pip3 install some-package

# Start Jupyter
jupyter lab --ip=0.0.0.0 --no-browser

# Clone repos, run scripts, etc.
```

## Manage Container

```bash
# Exit container and return to macOS
exit

# View container logs (if it crashes)
docker compose -f .devcontainer/docker-compose.yml logs mlfield

# Restart container
docker compose -f .devcontainer/docker-compose.yml up -d

# Stop container
docker compose -f .devcontainer/docker-compose.yml down

# Re-enter running container
docker compose -f .devcontainer/docker-compose.yml exec mlfield bash
```

## Important Notes

⚠️ **CPU-Only**: This container uses PyTorch CPU on macOS (no CUDA). GPU acceleration requires Linux with NVIDIA hardware.

💾 **Persistent Storage**: Your `/home` directory persists between container restarts. Modify `docker-compose.yml` if you want to mount additional folders.

📊 **Resources**: If container is slow:
- Open Docker Desktop → Preferences → Resources
- Increase CPU cores and Memory allocation
- Restart Docker

## Datasets & Additional Folders

To mount your datasets:

1. Create `.devcontainer/local.env`:
   ```bash
   DATASET_DIR=/Users/you/path/to/datasets
   ```

2. Delete the generated `docker-compose.yml`:
   ```bash
   rm .devcontainer/docker-compose.yml
   ```

3. Re-run:
   ```bash
   ./.devcontainer/launch.sh
   ```

Your datasets will be available at `/mnt/dataset` inside the container.

## Next Steps

- Read [README.md](./README.md) for comprehensive guide
- Check [ADAPTATION_SUMMARY.md](./ADAPTATION_SUMMARY.md) for technical details
- Modify `Dockerfile.cpu` if you need additional packages

---

**Enjoy!** Your ML development environment is ready. 🚀
