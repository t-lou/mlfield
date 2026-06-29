You can absolutely move both **Docker’s root directory** and **containerd’s root directory** to `/home/docker` on Ubuntu 26.04.  
The short version: **yes, it’s safe**, and **yes, you should**, because your `/` partition is small.

Below is the **exact, complete, production‑ready configuration** you need — with all missing pieces filled in.

---

## ✅ **Concise takeaway**
Move Docker’s data root by editing `/etc/docker/daemon.json`.  
Move containerd’s root by editing `/etc/containerd/config.toml`.  
Restart both services.  
Make sure `/home/docker` has correct permissions and enough space.

---

## 🧱 **1. Create the new directories**
```bash
sudo mkdir -p /home/docker/{docker,containerd}
sudo chown -R root:root /home/docker
```

If `/home` is on a different filesystem (likely ext4 or btrfs), this is exactly what you want.

---

## 🐳 **2. Move Docker root directory**
Docker uses `/var/lib/docker` by default.

### Edit Docker config
```bash
sudo nano /etc/docker/daemon.json
```

Add or modify:

```json
{
  "data-root": "/home/docker/docker"
}
```

If the file didn’t exist, this is all you need.

### Stop Docker and move existing data
```bash
sudo systemctl stop docker
sudo rsync -aP /var/lib/docker/ /home/docker/docker/
```

### Restart Docker
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Verify
```bash
docker info | grep "Docker Root Dir"
```

---

## 📦 **3. Move containerd root directory**
Containerd defaults to `/var/lib/containerd`.

### Generate a fresh config (recommended)
```bash
sudo containerd config default | sudo tee /etc/containerd/config.toml
```

### Edit the config
```bash
sudo nano /etc/containerd/config.toml
```

Find the line:

```
root = "/var/lib/containerd"
```

Change it to:

```
root = "/home/docker/containerd"
```

### Stop containerd and move data
```bash
sudo systemctl stop containerd
sudo rsync -aP /var/lib/containerd/ /home/docker/containerd/
```

### Restart containerd
```bash
sudo systemctl daemon-reload
sudo systemctl restart containerd
```

### Verify
```bash
ctr version
sudo ls /home/docker/containerd
```

---

## 🔒 **4. AppArmor / permissions considerations**
Ubuntu 26.04 uses AppArmor.  
Docker and containerd **do not require AppArmor profile changes** when moving data-root, because they do not hardcode paths in their profiles.

Still, ensure:

```bash
sudo chown -R root:root /home/docker
sudo chmod 700 /home/docker
```

---

## 📌 **5. Optional: bind mount instead of config change**
If you prefer not touching configs:

```bash
sudo systemctl stop docker containerd
sudo mkdir -p /home/docker/docker /home/docker/containerd
sudo rsync -aP /var/lib/docker/ /home/docker/docker/
sudo rsync -aP /var/lib/containerd/ /home/docker/containerd/
sudo mount --bind /home/docker/docker /var/lib/docker
sudo mount --bind /home/docker/containerd /var/lib/containerd
```

Persist in `/etc/fstab`:

```
/home/docker/docker      /var/lib/docker      none    bind    0 0
/home/docker/containerd  /var/lib/containerd  none    bind    0 0
```

This method is extremely stable and avoids config changes.

---

## 🧭 **6. Why this works well**
Your `/` partition is small — common on laptops or servers with separate `/home`.  
Docker and containerd can easily consume **tens or hundreds of GB** with images, layers, and logs.

Moving them to `/home/docker` gives you:

- more space  
- easier cleanup  
- better separation  
- simpler backups  
- no risk of filling `/` and breaking the OS  

---

## 🧪 **7. Quick health check**
Run:

```bash
docker run --rm hello-world
ctr images pull docker.io/library/alpine:latest
```

If both succeed, your migration is perfect.

---

## 🔧 What do you want next?
You can continue with:

- verify_docker_storage  
- optimize_containerd_gc  
- move_docker_logs  
- switch_to_btrfs_or_zfs  

If you want, I can also generate a **single script** that performs the entire migration automatically.
