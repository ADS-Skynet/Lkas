# Dev Container Setup Guide (Recommended for M1/M2 Mac)

## The Correct Architecture

**What you actually need:**
```
┌──────────────────────────────────────────────────────┐
│  Your M1 Mac                                         │
│                                                       │
│  ┌─────────────────────────────────────────────────┐│
│  │ VSCode with Dev Container Extension             ││
│  │                                                  ││
│  │  ┌────────────────────────────────────────────┐ ││
│  │  │ Docker Container (linux/amd64)             │ ││
│  │  │                                             │ ││
│  │  │  ✅ Python 3.10 (x86_64)                   │ ││
│  │  │  ✅ CARLA Python client (x86_64 .whl)      │ ││
│  │  │  ✅ All your lane detection code           │ ││
│  │  │  ✅ OpenCV, PyTorch, etc.                  │ ││
│  │  │                                             │ ││
│  │  │  Your code runs here ──────────────────┐   │ ││
│  │  └────────────────────────────────────────│───┘ ││
│  └───────────────────────────────────────────│─────┘│
└────────────────────────────────────────────┼───────┘
                                              │
                                              │ TCP/IP
                                              │ port 2000
                                              ↓
                        ┌────────────────────────────────┐
                        │  Remote Linux Machine          │
                        │                                 │
                        │  🖥️  CARLA Simulator Server    │
                        │  📡 Port 2000                   │
                        │                                 │
                        └────────────────────────────────┘
```

## Why Dev Container is Perfect for This

✅ **Solves the M1 Problem**:
- Runs x86_64 Python + CARLA client in Docker
- No need to compile anything
- Uses CARLA's official x86_64 Python wheels

✅ **Better than Manual Docker**:
- VSCode integration (edit code seamlessly)
- Debugger works
- Extensions work
- Git integration
- Terminal inside container

✅ **Portable**:
- Same environment for whole team
- Works on M1, M2, Intel Mac, Linux, Windows
- Configuration in code (`.devcontainer/`)

---

## Prerequisites

### 1. Install Docker Desktop
```bash
# Download from https://www.docker.com/products/docker-desktop/
```

### 2. Enable Rosetta 2 in Docker (Important!)
1. Open Docker Desktop
2. Settings → Features in development → Beta Features
3. ✅ Enable "Use Rosetta for x86/amd64 emulation on Apple Silicon"
4. Apply & Restart

### 3. Install VSCode Extension
In VSCode, install:
- **Dev Containers** (ms-vscode-remote.remote-containers)

---

## Quick Start

### Step 1: Open in Dev Container

1. **Open VSCode** in your project:
   ```bash
   cd /Users/donghyun/All/seame/ads_ld
   code .
   ```

2. **Open Command Palette** (`Cmd+Shift+P`)

3. **Select**: `Dev Containers: Reopen in Container`

4. **Wait** for container to build (first time: ~10-15 minutes)
   - Downloads base image
   - Installs dependencies
   - Downloads CARLA Python client

5. **You're in!** 🎉
   - VSCode is now running inside the container
   - Terminal is inside container
   - All code edits are synced

### Step 2: Verify CARLA Client Works

In the VSCode terminal (inside container):
```bash
# Test CARLA module
python -c "import carla; print('✅ CARLA client ready!')"
```

### Step 3: Connect to Remote CARLA Server

**Get the IP of your Linux machine running CARLA:**
```bash
# On the Linux machine:
hostname -I
# Example output: 192.168.1.100
```

**Test connection from Dev Container:**
```bash
cd lane_detection
python test_carla_connection.py --host 192.168.1.100 --port 2000
```

### Step 4: Run Lane Detection

```bash
cd lane_detection
python main.py --method cv --host 192.168.1.100 --port 2000
```

---

## How to Use Dev Container Daily

### Starting Work
```bash
# 1. Open project
cd /Users/donghyun/All/seame/ads_ld
code .

# 2. VSCode will auto-detect and ask: "Reopen in Container?"
#    Click "Reopen in Container"

# 3. Start coding!
```

### While Working
- **Edit files**: Just edit normally in VSCode
- **Run Python**: Use integrated terminal
- **Debug**: Use VSCode debugger (F5)
- **Git**: Use VSCode Git features

### Stopping
- Just close VSCode
- Container stops automatically
- Or: `Cmd+Shift+P` → `Dev Containers: Reopen Locally`

---

## Dev Container Configuration

The setup is defined in `.devcontainer/`:

### `devcontainer.json` - Main Config
```json
{
  "name": "Lane Detection - CARLA Client",
  "build": { "dockerfile": "Dockerfile" },
  "runArgs": ["--platform=linux/amd64"],  // Force x86_64
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", ...]
    }
  }
}
```

### `Dockerfile` - Container Definition
- Base: `python:3.10-bullseye` (x86_64)
- Installs: CARLA client, OpenCV, PyTorch, etc.
- CARLA downloaded from official releases

---

## Updating the Container

If you modify `.devcontainer/` files:

1. `Cmd+Shift+P` → `Dev Containers: Rebuild Container`
2. Wait for rebuild
3. Continue working

---

## Troubleshooting

### "Failed to build container"
```bash
# Check Docker is running
docker info

# Check Docker Desktop settings
# Make sure Rosetta 2 is enabled
```

### "Cannot import carla"
```bash
# Inside container terminal:
echo $PYTHONPATH
# Should include: /opt/carla/...

# Test manually:
python -c "import sys; sys.path.append('/opt/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'); import carla"
```

### "Connection refused to CARLA server"
```bash
# Make sure Linux machine CARLA is running
# Make sure firewall allows port 2000

# Test from Mac (outside container):
nc -zv 192.168.1.100 2000

# Test from inside container:
ping 192.168.1.100
nc -zv 192.168.1.100 2000
```

### Slow performance
This is normal - it's running x86_64 on ARM via emulation.

**Optimizations:**
- Use lower resolution: `--width 640 --height 480`
- Reduce quality on CARLA server side
- Close other apps
- Allocate more resources to Docker (Settings → Resources)

---

## Remote CARLA Server Setup

On your **Linux machine** (not Mac):

### 1. Start CARLA Server
```bash
cd /path/to/CARLA
./CarlaUE4.sh
```

### 2. Allow Network Access
```bash
# Check firewall
sudo ufw status

# Allow port 2000 if needed
sudo ufw allow 2000/tcp
```

### 3. Get Server IP
```bash
hostname -I
# Note the IP address (e.g., 192.168.1.100)
```

### 4. Keep CARLA Running
Use `screen` or `tmux` to keep it running:
```bash
# Install screen
sudo apt install screen

# Start in screen
screen -S carla
./CarlaUE4.sh

# Detach: Ctrl+A, then D
# Reattach: screen -r carla
```

---

## Connecting from Dev Container

### Using Command Line Args
```bash
python main.py --host 192.168.1.100 --port 2000
```

### Using Config File
Edit `lane_detection/config.yaml`:
```yaml
carla:
  host: "192.168.1.100"  # Your Linux machine IP
  port: 2000
```

Then:
```bash
python main.py  # Uses config file
```

### Environment Variable
```bash
export CARLA_HOST=192.168.1.100
export CARLA_PORT=2000
python main.py --host $CARLA_HOST --port $CARLA_PORT
```

---

## Testing Without CARLA Server

You can still test components without CARLA:

```bash
cd lane_detection
python test_setup.py  # Tests CV detector, analyzer, visualizer
```

This uses synthetic images and doesn't need CARLA!

---

## Performance Comparison

| Setup | CARLA Client | Performance | Complexity |
|-------|-------------|-------------|------------|
| **Dev Container (Recommended)** | ✅ x86_64 in Docker | 🟡 Medium (emulation) | 🟢 Easy |
| Manual Docker | ✅ x86_64 in Docker | 🟡 Medium (emulation) | 🟡 Medium |
| Native M1 | ❌ Can't install | - | - |
| Remote Dev (SSH) | ✅ Native on Linux | 🟢 Fast | 🟡 Medium |

---

## Advanced: Remote Development (Alternative)

If Dev Container is too slow, use **Remote SSH**:

### Setup
1. Install: **Remote - SSH** extension in VSCode
2. Configure SSH to Linux machine
3. Open VSCode remotely
4. Develop directly on Linux (native performance!)

### Pros/Cons
- ✅ Native performance (no emulation)
- ✅ Direct access to CARLA server
- ❌ Need SSH access
- ❌ Need to install dependencies on Linux machine

---

## Files Reference

```
.devcontainer/
├── devcontainer.json   # VSCode Dev Container config
└── Dockerfile          # Container image definition

lane_detection/
├── config.yaml         # Can set remote CARLA host here
├── main.py            # Use --host flag
└── test_carla_connection.py  # Test remote connection
```

---

## Checklist

Before starting:
- [ ] Docker Desktop installed
- [ ] Rosetta 2 enabled in Docker
- [ ] Dev Containers extension installed in VSCode
- [ ] Linux machine IP address known
- [ ] CARLA running on Linux machine
- [ ] Port 2000 accessible

To verify setup:
- [ ] Container builds successfully
- [ ] `import carla` works
- [ ] Can ping Linux machine from container
- [ ] `test_carla_connection.py` succeeds
- [ ] Lane detection runs

---

## Quick Commands

```bash
# Open in container
code /Users/donghyun/All/seame/ads_ld

# Inside container terminal:
python -c "import carla"  # Test CARLA
python test_carla_connection.py --host <IP> --port 2000  # Test connection
python main.py --method cv --host <IP> --port 2000  # Run

# Rebuild container (after changes)
# Cmd+Shift+P → "Dev Containers: Rebuild Container"
```

---

## Summary

1. **Dev Container** runs x86_64 Python + CARLA client on your M1 Mac
2. **CARLA server** runs on remote Linux machine
3. **Connection** happens over network (TCP/IP)
4. **VSCode** makes it seamless - feels like local development

This is the **best solution** for your M1 Mac setup! 🎉
