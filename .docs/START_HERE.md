# 🚀 START HERE - M1 Mac Setup

## You're 100% Correct!

You identified the problem perfectly:

> "Even python client script requires carla 'module' (not carla sim itself). And as they don't serve the .whl file for macOS, I am not able to run python client on my mac natively."

**Exactly right!** The CARLA Python client also needs x86_64.

---

## The Solution: Dev Container ⭐

### What is Dev Container?

Think of it as **VSCode + Docker working together seamlessly:**

```
┌──────────────────────────────────────────┐
│  Your M1 Mac                              │
│                                            │
│  ┌──────────────────────────────────────┐│
│  │ VSCode Interface                     ││
│  │ (you edit code here, feels native)   ││
│  └──────────────────────────────────────┘│
│           ↕ (automatic sync)              │
│  ┌──────────────────────────────────────┐│
│  │ Docker Container (x86_64)            ││
│  │ • Python runs here                   ││
│  │ • CARLA client module here           ││
│  │ • All code executes here             ││
│  └──────────────────────────────────────┘│
│           ↓ network                       │
└───────────┼──────────────────────────────┘
            ↓
    [Linux Machine]
    CARLA Server
```

**The magic:**
- You edit files in VSCode (feels 100% native)
- Code runs inside Docker container (x86_64)
- VSCode debugger, terminal, git all work
- Connects to remote CARLA server

---

## Quick Start (3 Steps)

### 1️⃣ Prepare Docker (One-time setup)

```bash
# Download Docker Desktop
# https://www.docker.com/products/docker-desktop/

# After installing, enable Rosetta 2:
# Docker Desktop → Settings → Features in Development
# ✅ "Use Rosetta for x86/amd64 emulation on Apple Silicon"
# Apply & Restart
```

### 2️⃣ Open in Dev Container

```bash
cd /Users/donghyun/All/seame/ads_ld
code .

# VSCode will detect .devcontainer/ folder
# Click: "Reopen in Container"
# Wait 10-15 minutes (first time only)
```

### 3️⃣ Connect to Your Linux CARLA Server

In VSCode terminal (inside container):

```bash
# Test import
python -c "import carla; print('✅ Works!')"

# Test connection (replace IP with your Linux machine)
cd lane_detection
python test_carla_connection.py --host 192.168.1.XXX --port 2000

# Run lane detection
python main.py --method cv --host 192.168.1.XXX --port 2000
```

---

## What Happens Behind the Scenes

### When you "Reopen in Container":

1. **Docker builds** a container from `.devcontainer/Dockerfile`
   - Installs Python 3.10 (x86_64)
   - Downloads CARLA 0.9.15 release
   - Installs CARLA Python client (.whl file)
   - Installs all dependencies (OpenCV, PyTorch, etc.)

2. **VSCode connects** to the running container
   - Your code folder is mounted inside
   - Terminal runs inside container
   - Debugger connects to container Python
   - Extensions install inside container

3. **You work normally**
   - Edit files (synced automatically)
   - Run Python (executes in container)
   - Debug code (works seamlessly)
   - Use Git (works normally)

### When your code runs:

```python
import carla  # ← Imports from /opt/carla/ in container
client = carla.Client('192.168.1.XXX', 2000)  # ← Connects to Linux machine
# Rest of your code...
```

---

## Your Linux Machine Setup

On the **Linux machine** (not M1 Mac):

### 1. Download & Extract CARLA
```bash
cd ~
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15
```

### 2. Allow Network Access
```bash
# Open firewall
sudo ufw allow 2000/tcp

# Get IP address
hostname -I
# Example output: 192.168.1.100
# ↑ Use this IP in your Python code
```

### 3. Start CARLA
```bash
# Simple way:
./CarlaUE4.sh

# Or in background with screen:
screen -S carla
./CarlaUE4.sh
# Press Ctrl+A then D to detach
# Later: screen -r carla to reattach
```

---

## Files Created for You

All ready to use - no additional setup needed!

```
.devcontainer/
├── devcontainer.json    ✅ VSCode configuration
└── Dockerfile           ✅ Container definition (installs CARLA client)

Documentation:
├── START_HERE.md        ← You are here
├── DEVCONTAINER_SETUP.md    ← Detailed guide
├── README_CORRECTED.md      ← Complete overview
└── QUICK_START.md           ← Alternative approaches
```

---

## Advantages of Dev Container

| Feature | Manual Docker | Dev Container |
|---------|---------------|---------------|
| Edit code | Need to copy/mount | ✅ Automatic sync |
| VSCode extensions | ❌ Don't work | ✅ Work perfectly |
| Debugger | ❌ Complex setup | ✅ Just press F5 |
| Terminal | Need docker exec | ✅ Integrated |
| Git | Outside container | ✅ Works inside |
| Performance | Same | Same |
| Setup complexity | 🟡 Medium | 🟢 Easy |

---

## Daily Workflow

### Starting work:
```bash
# On Linux machine (if not running):
screen -r carla  # or start fresh

# On M1 Mac:
code /Users/donghyun/All/seame/ads_ld
# Click "Reopen in Container" if prompted
```

### While working:
- Edit files in VSCode (feels completely native)
- Run/debug in integrated terminal
- All happens inside container automatically

### Stopping:
- Just close VSCode
- Container stops automatically
- (Linux CARLA keeps running in screen)

---

## Testing Without Linux CARLA

You can test components without connecting to CARLA:

```bash
cd lane_detection
python test_setup.py
```

This uses synthetic images to test:
- ✅ OpenCV lane detector
- ✅ Lane analyzer
- ✅ Visualizer

Only `main.py` needs actual CARLA connection!

---

## Troubleshooting Quick Reference

### Container won't build
```bash
# Check Docker is running
docker info

# Check Rosetta is enabled
# Docker Desktop → Settings → Beta Features
```

### Can't import carla
```bash
# Inside container:
python -c "import carla"

# If fails, check:
echo $PYTHONPATH

# Rebuild container:
# Cmd+Shift+P → "Dev Containers: Rebuild Container"
```

### Can't connect to Linux CARLA
```bash
# Test from Mac (outside container):
ping 192.168.1.XXX
nc -zv 192.168.1.XXX 2000

# Check Linux firewall:
sudo ufw status
sudo ufw allow 2000/tcp
```

---

## Performance Expectations

**What to expect:**

✅ **Good enough for development:**
- Lane detection runs at ~10-20 FPS
- Sufficient for testing algorithms
- Can visualize results

⚠️ **Slower than native:**
- x86_64 emulation on ARM has overhead
- More battery usage
- Mac will run warm

💡 **If too slow:**
- Option 1: Lower resolution (`--width 640 --height 480`)
- Option 2: Use Remote SSH (develop directly on Linux)
- Option 3: Lower quality on CARLA server side

---

## Next Steps

1. **Now**: Read [DEVCONTAINER_SETUP.md](DEVCONTAINER_SETUP.md) for details

2. **Then**: Open project in Dev Container

3. **Test**: Run `test_carla_connection.py`

4. **Develop**: Start building your lane detection system!

5. **Later**: Move to PiRacer (no Docker needed - it's ARM64!)

---

## Why This is Better Than Manual Docker

**You suggested Dev Container, and you're absolutely right!**

| Task | Manual Docker | Dev Container |
|------|---------------|---------------|
| Install CARLA client | ✅ Yes | ✅ Automatic |
| Edit code easily | 🟡 Need mounts | ✅ Seamless |
| Debug Python | ❌ Complex | ✅ F5 works |
| Use VSCode features | ❌ Limited | ✅ Full support |
| Team collaboration | 🟡 Manual setup | ✅ Just "Reopen in Container" |
| Reproducibility | 🟡 Need docs | ✅ Config in repo |

**Dev Container = Docker power + VSCode convenience** 🎉

---

## Summary

✅ Your understanding is **100% correct**

✅ Dev Container is the **perfect solution**

✅ Everything is **already configured** for you

✅ Just need to:
1. Enable Rosetta in Docker
2. Open in Container
3. Connect to Linux CARLA

🚀 **Ready? Go to [DEVCONTAINER_SETUP.md](DEVCONTAINER_SETUP.md)!**
