# Remote Development Setup Guide

## Quick Migration from Dev Container to Remote-SSH

### 1. On Ubuntu Machine

```bash
# Install SSH server (if not already)
sudo apt-get update
sudo apt-get install openssh-server
sudo systemctl enable ssh --now

# Clone/sync your code
cd ~
git clone <your-repo-url> ads_ld
cd ads_ld

# Install Python dependencies NATIVELY
pip3 install --user -r requirements.txt

# Install CARLA Python client
pip3 install --user carla==0.9.15

# Verify CARLA is accessible
ls /opt/carla-simulator/  # or wherever CARLA is installed
```

### 2. On MacBook (VSCode)

```bash
# Install Remote-SSH extension
# Extensions → Search "Remote - SSH" → Install

# Add SSH config (⌘+⇧+P → "Remote-SSH: Open SSH Configuration File")
# Add to ~/.ssh/config:
Host ubuntu-carla
    HostName <UBUNTU_IP>
    User <USERNAME>
    Port 22
    ForwardAgent yes
```

### 3. Connect & Run

```bash
# In VSCode:
# ⌘+⇧+P → "Remote-SSH: Connect to Host" → "ubuntu-carla"
# Open folder: ~/ads_ld

# Terminal 1: CARLA Server (native!)
/opt/carla-simulator/CarlaUE4.sh -quality-level=Low -fps=30 -prefernvidia

# Terminal 2: Detection Server
cd ~/ads_ld
python3 detection/main.py --method cv --port 5556

# Terminal 3: Main Client (with latency tracking!)
python3 simulation/main.py --detector-url tcp://localhost:5556 --latency
```

## Performance Comparison

Run with `--latency` flag to see detailed breakdown:

```bash
# This will print latency reports every 90 frames
python3 simulation/main.py \
    --detector-url tcp://localhost:5556 \
    --latency \
    --no-sync  # Try async mode for higher FPS
```

Expected improvements:
- **Detection Processing**: 2-3x faster (no container overhead)
- **Network Latency**: 10-20x faster (native localhost vs Docker bridge)
- **Total End-to-End**: 2-3x improvement

## Troubleshooting

### If lane detection is still slow:

1. **Check which method is running:**
```bash
# CV method (faster, no GPU needed)
python3 detection/main.py --method cv

# DL method (slower without GPU)
python3 detection/main.py --method dl --gpu 0
```

2. **Profile the detector:**
```python
# The latency tracker will show bottleneck:
# ⚠ Primary Bottleneck: Detection Processing (45.23ms)
```

3. **Optimize CV parameters** (if using CV method):
```bash
# Edit config.yaml - reduce Canny thresholds, Hough parameters
# Lower values = faster but less accurate
```

4. **Use GPU** (if available and using DL):
```bash
python3 detection/main.py --method dl --gpu 0
```

## Port Forwarding for Web Viewer

If you want to view visualization on MacBook:

```bash
# In VSCode Remote-SSH session:
python3 simulation/main.py \
    --detector-url tcp://localhost:5556 \
    --viewer web --web-port 8080

# VSCode will auto-forward port 8080
# Open browser: http://localhost:8080
```

## Reverting to Dev Container

If you need to go back:

```bash
# Just reopen in dev container
# ⌘+⇧+P → "Dev Containers: Reopen in Container"
```
