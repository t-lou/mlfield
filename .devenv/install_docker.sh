#!/bin/bash
set -e

# 1. Remove old versions
sudo apt remove -y docker docker-engine docker.io containerd runc || true

# 2. Install dependencies
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# 3. Add Docker’s official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 4. Add Docker repository (Ubuntu 24.04 = noble)
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu noble stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Install Docker Engine + CLI + Compose
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 6. Allow running Docker without sudo
sudo usermod -aG docker $USER