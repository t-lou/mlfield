# Development Environment Documentation Index

## 📍 Where to Start

### 🍎 macOS Users
1. Read: [MACOS_QUICKSTART.md](./MACOS_QUICKSTART.md) (5 min read)
2. Run: `./.devcontainer/launch.sh`
3. Done! ✅

### 🐧 Linux/WSL2 Users  
1. Run: `./.devcontainer/launch.sh`
2. (First time: GPU setup may trigger installation)
3. Done! ✅

### 🔍 Verify Your Setup
```bash
./.devcontainer/verify.sh
```

---

## 📚 Documentation Files

| File | Audience | Read Time |
|------|----------|-----------|
| **MACOS_QUICKSTART.md** | macOS users | 5 min |
| **README.md** | All users (comprehensive) | 15 min |
| **IMPLEMENTATION_SUMMARY.md** | Developers/maintainers | 10 min |
| **ADAPTATION_SUMMARY.md** | Technical details | 10 min |

---

## 🚀 Quick Commands

```bash
# Launch development container
./.devcontainer/launch.sh

# Verify setup is correct
./.devcontainer/verify.sh

# Manually manage container
docker compose -f .devcontainer/docker-compose.yml ps
docker compose -f .devcontainer/docker-compose.yml logs mlfield
docker compose -f .devcontainer/docker-compose.yml down
```

---

## 🛠️ What's Included

### Scripts
- **launch.sh** - Main entry point (cross-platform)
- **setup_mac.sh** - macOS setup verification
- **setup_linux.sh** - Linux GPU setup
- **generate_docker_compose.sh** - Docker configuration generator
- **verify.sh** - Environment verification tool

### Dockerfiles
- **Dockerfile.cuda** - Linux with NVIDIA CUDA & GPU support
- **Dockerfile.cpu** - macOS & Linux without GPU (CPU-only)

### Configuration
- **devcontainer.json** - VS Code remote containers setup
- **docker-compose.yml** - Generated dynamically (don't edit)
- **docker-compose.yml.template** - Legacy template (reference only)

---

## 🎯 Common Tasks

### Mount Your Datasets
Create `.devcontainer/local.env`:
```bash
DATASET_DIR=/path/to/your/datasets
```
Then delete and regenerate docker-compose:
```bash
rm .devcontainer/docker-compose.yml
./.devcontainer/launch.sh
```

### Add Python Packages
Inside container:
```bash
pip3 install package-name
```

Or make persistent by editing the appropriate Dockerfile and rebuilding.

### Check if GPU is Available (Linux)
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
(Will be `False` on macOS - CPU-only)

### View Container Logs
```bash
docker compose -f .devcontainer/docker-compose.yml logs -f mlfield
```

---

## ⚡ Performance Tips

### macOS
- Docker Desktop → Preferences → Resources
- Allocate: 8+ GB memory, 4+ CPU cores
- Disk: 20+ GB available space

### Linux/WSL2
- Ensure NVIDIA driver is up to date
- WSL2 should have sufficient disk allocation
- Check: `nvidia-smi` on host

---

## 🐛 Troubleshooting

### "Docker daemon is not running" (macOS)
→ Start Docker Desktop from Applications

### "nvidia-smi not found" (Linux)
→ Install NVIDIA drivers: `sudo ubuntu-drivers autoinstall`

### Container won't build
→ Check logs: `docker compose -f .devcontainer/docker-compose.yml logs`

### Slow performance
→ Increase Docker resources in Desktop settings

See [README.md](./README.md#troubleshooting) for more troubleshooting.

---

## 📖 Platform Comparison

| Feature | macOS | Linux |
|---------|-------|-------|
| GPU Support | ❌ CPU-only | ✅ Full CUDA |
| X11 Forwarding | ❌ | ✅ |
| Setup Time | < 5 min | 5-15 min |
| Docker Image | Dockerfile.cpu | Dockerfile.cuda |
| Network Mode | bridge | host |

---

## 🔄 Upgrading/Rebuilding

To rebuild the container (e.g., after modifying Dockerfile):

```bash
# Option 1: Full rebuild
docker compose -f .devcontainer/docker-compose.yml build --no-cache
docker compose -f .devcontainer/docker-compose.yml up -d

# Option 2: Use launch script (regenerates everything)
./.devcontainer/launch.sh
```

---

## 📞 Support Resources

- **Docker Installation**: https://www.docker.com/products/docker-desktop
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **PyTorch**: https://pytorch.org
- **NVIDIA Container Toolkit**: https://github.com/NVIDIA/nvidia-docker

---

## ✅ Verification Checklist

Before starting development:
- [ ] Docker installed and running
- [ ] `docker --version` works
- [ ] `docker compose version` works
- [ ] `./.devcontainer/verify.sh` passes
- [ ] Can run `./.devcontainer/launch.sh`

---

**Last Updated**: 2024
**Status**: ✅ Production Ready
**Platforms Supported**: macOS, Linux, WSL2
