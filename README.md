# Lane Detection for LKAS - Complete Setup Guide

## ⚠️ IMPORTANT: Understanding the Architecture

### What You Actually Have

```
┌─────────────────────────────────────────────────────────────┐
│  Your M1 Mac (macOS ARM64)                                  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  VSCode + Dev Container                                │ │
│  │                                                         │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  Docker Container (linux/amd64 via Rosetta)      │  │ │
│  │  │                                                   │  │ │
│  │  │  🐍 Python 3.10 (x86_64)                         │  │ │
│  │  │  📦 CARLA Python Client Module (x86_64 .whl)     │  │ │
│  │  │  🚗 Your Lane Detection Code                     │  │ │
│  │  │  📊 OpenCV, PyTorch, NumPy, etc.                 │  │ │
│  │  │                                                   │  │ │
│  │  │  Connects to remote CARLA ───────────────────┐   │  │ │
│  │  └──────────────────────────────────────────────│───┘  │ │
│  └───────────────────────────────────────────────┼───────┘ │
└────────────────────────────────────────────────┼───────────┘
                                                  │
                                    Network (TCP/IP)
                                    port 2000
                                                  │
                                                  ↓
                  ┌──────────────────────────────────────────┐
                  │  Remote Linux Machine (x86_64)           │
                  │                                           │
                  │  🖥️  CARLA Simulator Server              │
                  │  🎮 Unreal Engine 4                      │
                  │  📡 Listening on port 2000               │
                  │                                           │
                  └──────────────────────────────────────────┘
```

### The Key Point

**BOTH the CARLA Python client AND CARLA server need x86_64:**

❌ **What DOESN'T work on M1:**
- Native CARLA Python client (no ARM64 .whl files)
- Native CARLA server (no macOS build at all)

✅ **What DOES work:**
- **CARLA Python client** runs in **Docker on your M1** (x86_64 via Rosetta 2)
- **CARLA server** runs on **remote Linux machine** (native x86_64)
- They **connect over network** (localhost or LAN)

---

## 🚀 Quick Start (3 Options)

### Option 1: Dev Container (RECOMMENDED) ⭐

**Best for:** Daily development on M1 Mac

1. Open project in VSCode
2. Install "Dev Containers" extension
3. `Cmd+Shift+P` → "Reopen in Container"
4. Wait for build (~10-15 min first time)
5. Connect to remote CARLA server

📖 **Full guide:** [DEVCONTAINER_SETUP.md](DEVCONTAINER_SETUP.md)

**Pros:**
- ✅ Seamless VSCode integration
- ✅ Debugger works
- ✅ All extensions work
- ✅ Easy to use

**Cons:**
- 🟡 Slower than native (emulation overhead)
- 🟡 First build takes time

---

### Option 2: Manual Docker

**Best for:** If you don't use VSCode

```bash
# Build container
docker build -t lane-detection-carla .devcontainer/

# Run with network access
docker run -it --rm \
  --platform linux/amd64 \
  -v $(pwd):/workspace \
  -w /workspace \
  lane-detection-carla \
  bash

# Inside container:
cd lane_detection
python main.py --host <REMOTE_IP> --port 2000
```

---

### Option 3: Remote Development (SSH)

**Best for:** Maximum performance

1. Install "Remote - SSH" extension in VSCode
2. SSH to Linux machine
3. Develop directly on Linux (where CARLA server runs)
4. No emulation needed!

---

## 📋 Setup Steps

### Step 1: Prepare Your M1 Mac

#### Install Docker Desktop
```bash
# Download from https://www.docker.com/products/docker-desktop/
```

#### Enable Rosetta 2 in Docker
1. Docker Desktop → Settings
2. Features in development → Beta Features
3. ✅ "Use Rosetta for x86/amd64 emulation on Apple Silicon"
4. Apply & Restart

#### Install VSCode Extensions
- Dev Containers (ms-vscode-remote.remote-containers)
- Python (ms-python.python)

### Step 2: Set Up Linux Machine (CARLA Server)

On your **Linux machine** (not Mac!):

```bash
# Download CARLA (on Linux machine)
cd ~
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15

# Start CARLA server
./CarlaUE4.sh

# Or in background with screen:
screen -S carla
./CarlaUE4.sh
# Press Ctrl+A then D to detach
```

#### Allow Network Access
```bash
# Check firewall
sudo ufw status

# Allow CARLA port
sudo ufw allow 2000/tcp

# Get IP address
hostname -I
# Note this IP (e.g., 192.168.1.100)
```

### Step 3: Open Project in Dev Container

On your **M1 Mac**:

```bash
cd /Users/donghyun/All/seame/ads_ld
code .
```

In VSCode:
1. `Cmd+Shift+P`
2. "Dev Containers: Reopen in Container"
3. Wait for build

### Step 4: Test Connection

In VSCode terminal (inside container):

```bash
# Test CARLA module
python -c "import carla; print('✅ Works!')"

# Test connection to remote server
cd lane_detection
python test_carla_connection.py --host 192.168.1.100 --port 2000
```

Replace `192.168.1.100` with your Linux machine's IP.

### Step 5: Run Lane Detection

```bash
cd lane_detection
python main.py --method cv --host 192.168.1.100 --port 2000
```

---

## 📁 Project Structure

```
ads_ld/
├── .devcontainer/              # Dev Container configuration ⭐
│   ├── devcontainer.json      # VSCode config
│   └── Dockerfile             # Container definition
│
├── lane_detection/            # Main code directory
│   ├── main.py               # Entry point
│   ├── carla_interface.py    # CARLA connection
│   ├── config.yaml           # Configuration
│   ├── test_carla_connection.py  # Connection tester
│   ├── test_setup.py         # Test without CARLA
│   │
│   ├── traditional/          # OpenCV lane detection
│   │   └── cv_lane_detector.py
│   │
│   ├── deep_learning/        # CNN lane detection
│   │   └── lane_net.py
│   │
│   └── utils/                # Utilities
│       ├── lane_analyzer.py
│       └── visualizer.py
│
├── DEVCONTAINER_SETUP.md     # 📘 Dev Container guide (READ THIS!)
├── README_CORRECTED.md       # This file
├── QUICK_START.md            # Quick start guide
├── requirements.txt          # Python dependencies
└── docker-compose.yml        # Docker Compose config
```

---

## 🔧 Configuration

### Option A: Command Line (Quick)

```bash
python main.py --host 192.168.1.100 --port 2000 --method cv
```

### Option B: Config File (Persistent)

Edit `lane_detection/config.yaml`:

```yaml
carla:
  host: "192.168.1.100"  # Your Linux machine IP
  port: 2000

camera:
  width: 800
  height: 600

system:
  detection_method: "cv"  # or "dl"
```

Then just run:
```bash
python main.py
```

### Option C: Environment Variables

```bash
export CARLA_HOST=192.168.1.100
export CARLA_PORT=2000
python main.py --host $CARLA_HOST --port $CARLA_PORT
```

---

## 🧪 Testing

### Test 1: Without CARLA (Always works)

```bash
cd lane_detection
python test_setup.py
```

This tests CV detector, analyzer, and visualizer using synthetic images.

### Test 2: CARLA Module Import

```bash
python -c "import carla; print('✅ CARLA client works!')"
```

### Test 3: Connection to Remote CARLA

```bash
python test_carla_connection.py --host <LINUX_IP> --port 2000
```

### Test 4: Full Lane Detection

```bash
python main.py --method cv --host <LINUX_IP> --port 2000
```

---

## 🐛 Troubleshooting

### "Cannot import carla"

**Inside Dev Container:**
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Should include /opt/carla/...

# Try manual import
python -c "import sys; sys.path.append('/opt/carla/PythonAPI/carla'); import carla"
```

**Fix:** Rebuild container
```
Cmd+Shift+P → "Dev Containers: Rebuild Container"
```

### "Connection refused" to CARLA

**Check from Mac (outside container):**
```bash
# Can you reach the Linux machine?
ping 192.168.1.100

# Is port 2000 open?
nc -zv 192.168.1.100 2000
```

**Check on Linux machine:**
```bash
# Is CARLA running?
ps aux | grep Carla

# Is it listening on port 2000?
netstat -tuln | grep 2000

# Check firewall
sudo ufw status
```

### Slow Performance

This is expected - x86_64 emulation on ARM is slower.

**Optimizations:**
```bash
# Reduce resolution
python main.py --width 640 --height 480

# On Linux CARLA server, use low quality
./CarlaUE4.sh -quality-level=Low

# Allocate more resources to Docker
# Docker Desktop → Settings → Resources
```

---

## 📊 Performance Comparison

| Setup | Client | Server | Speed | Complexity |
|-------|--------|--------|-------|------------|
| **Dev Container + Remote** | M1 Docker | Linux | 🟡🟡🟡 | 🟢 Easy |
| Manual Docker + Remote | M1 Docker | Linux | 🟡🟡🟡 | 🟡 Medium |
| Remote SSH | Linux | Linux | 🟢🟢🟢 | 🟡 Medium |
| Native (Windows/Linux) | Native | Same | 🟢🟢🟢🟢 | 🟢 Easy |

---

## 🎯 Daily Workflow

### Morning:

**On Linux machine:**
```bash
screen -r carla  # Reattach to CARLA if needed
# Or start fresh:
./CarlaUE4.sh
```

**On M1 Mac:**
```bash
cd /Users/donghyun/All/seame/ads_ld
code .
# Click "Reopen in Container" when prompted
```

### Working:

```bash
# Edit code in VSCode (automatically synced to container)
# Run in integrated terminal:
cd lane_detection
python main.py --host <LINUX_IP> --port 2000
```

### Evening:

- Close VSCode (container stops automatically)
- Linux CARLA can keep running (in screen)

---

## 🚗 Moving to PiRacer

Good news! For PiRacer:
- ✅ No CARLA needed
- ✅ Uses real camera
- ✅ Can run directly on Raspberry Pi (ARM64!)
- ✅ Just replace `CARLAInterface` with `PiRacerCamera`

---

## 📚 Documentation

| File | Content |
|------|---------|
| **[DEVCONTAINER_SETUP.md](DEVCONTAINER_SETUP.md)** | Complete Dev Container guide |
| **[QUICK_START.md](QUICK_START.md)** | General quick start |
| **[lane_detection/README.md](lane_detection/README.md)** | Technical documentation |
| **[requirements.txt](requirements.txt)** | Python dependencies |

---

## ✅ Checklist

**Before you start:**
- [ ] Docker Desktop installed on M1 Mac
- [ ] Rosetta 2 enabled in Docker
- [ ] Dev Containers extension in VSCode
- [ ] CARLA running on Linux machine
- [ ] Linux machine IP address known
- [ ] Port 2000 accessible

**First time setup:**
- [ ] Open in Dev Container (builds automatically)
- [ ] Test: `python -c "import carla"`
- [ ] Test: `python test_carla_connection.py --host <IP>`
- [ ] Run: `python main.py --host <IP>`

---

## 🆘 Getting Help

1. **Dev Container issues**: See [DEVCONTAINER_SETUP.md](DEVCONTAINER_SETUP.md)
2. **Connection issues**: Check firewall and network
3. **Performance issues**: Try Remote SSH instead

---

## Summary

✅ **What you need to understand:**

1. **M1 Mac can't run CARLA natively** (neither client nor server)
2. **Solution**:
   - CARLA **client** runs in Docker on M1 (x86_64 emulation)
   - CARLA **server** runs on remote Linux machine (native)
   - They communicate over network
3. **Dev Container** makes this seamless in VSCode
4. **Performance** is acceptable for development
5. **PiRacer** won't need any of this! 🎉

Ready to start? Open [DEVCONTAINER_SETUP.md](DEVCONTAINER_SETUP.md)!
