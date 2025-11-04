# CARLA Setup Guide for macOS M1/M2 (Apple Silicon)

## The Problem

CARLA simulator does not provide pre-built binaries for macOS ARM64 (M1/M2). The issues are:

1. ❌ No `.whl` files for macOS ARM64
2. ❌ CARLA Python API source code (`.cpp` files) requires compilation
3. ❌ Compiling CARLA on M1 requires complex Unreal Engine setup
4. ❌ Native build is not officially supported

## The Solution: Docker with Rosetta 2

Run CARLA server in Docker (x86_64 with Rosetta 2 emulation) and connect to it from your M1 Mac's Python environment.

---

## Setup Instructions

### Step 1: Install Docker Desktop for Mac

1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
2. Install and start Docker Desktop

### Step 2: Enable Rosetta 2 in Docker

1. Open **Docker Desktop**
2. Go to **Settings** (⚙️ icon)
3. Navigate to **Features in development** → **Beta Features**
4. Enable **"Use Rosetta for x86/amd64 emulation on Apple Silicon"**
5. Click **Apply & Restart**

### Step 3: Pull CARLA Docker Image

```bash
# Pull CARLA 0.9.15 (latest stable)
docker pull carlasim/carla:0.9.15

# Or use a specific version
# docker pull carlasim/carla:0.9.13
```

This will take some time (~6-8 GB download).

### Step 4: Run CARLA Server in Docker

```bash
# Run CARLA server with display support
docker run \
  --rm \
  --platform linux/amd64 \
  -p 2000-2002:2000-2002 \
  -e DISPLAY= \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen
```

**Explanation:**
- `--platform linux/amd64`: Force x86_64 architecture (runs via Rosetta)
- `-p 2000-2002:2000-2002`: Forward CARLA ports
- `-RenderOffScreen`: Run without GUI (headless mode)
- `--rm`: Auto-remove container when stopped

**Alternative with lower quality for better performance:**
```bash
docker run \
  --rm \
  --platform linux/amd64 \
  -p 2000-2002:2000-2002 \
  carlasim/carla:0.9.15 \
  ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

### Step 5: Install CARLA Python Client on Your Mac

The Python client can run natively on M1! You just need the matching version:

```bash
# For CARLA 0.9.15
pip install carla==0.9.15

# If that fails, try installing from source
cd /Users/donghyun/All/seame/ads_ld
git clone https://github.com/carla-simulator/carla.git carla-client
cd carla-client
git checkout 0.9.15  # Match your Docker version

# The Python client is pure Python and should work
export PYTHONPATH="${PYTHONPATH}:${PWD}/PythonAPI/carla"
```

**OR** if the pip package doesn't work, download the egg file:

```bash
# Download the Python API egg (pure Python, works on M1)
cd /Users/donghyun/All/seame/ads_ld
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
export PYTHONPATH=$PYTHONPATH:/Users/donghyun/All/seame/ads_ld/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

### Step 6: Test Connection

In a new terminal (while Docker container is running):

```bash
cd /Users/donghyun/All/seame/ads_ld/lane_detection
python test_carla_connection.py
```

---

## Quick Start Commands

### Terminal 1: Start CARLA Server
```bash
docker run --rm --platform linux/amd64 -p 2000-2002:2000-2002 \
  carlasim/carla:0.9.15 \
  ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

### Terminal 2: Run Lane Detection
```bash
cd /Users/donghyun/All/seame/ads_ld/lane_detection
python main.py --method cv --host localhost --port 2000
```

---

## Docker Management Commands

### Start CARLA (save as alias)
```bash
# Add to ~/.zshrc or ~/.bashrc
alias carla-start='docker run --rm --platform linux/amd64 -p 2000-2002:2000-2002 carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -quality-level=Low'
```

### Check if CARLA is running
```bash
docker ps
```

### Stop CARLA
```bash
# Press Ctrl+C in the terminal running Docker
# Or:
docker stop $(docker ps -q --filter ancestor=carlasim/carla:0.9.15)
```

### Remove CARLA image (to free space)
```bash
docker rmi carlasim/carla:0.9.15
```

---

## Advanced: Docker Compose Setup

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  carla:
    image: carlasim/carla:0.9.15
    platform: linux/amd64
    ports:
      - "2000:2000"
      - "2001:2001"
      - "2002:2002"
    command: ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
    deploy:
      resources:
        limits:
          memory: 8G
```

Then use:
```bash
# Start
docker-compose up

# Stop
docker-compose down
```

---

## Performance Considerations

### Expected Performance on M1
- ⚠️ **Slower than native**: Rosetta 2 emulation adds overhead
- 🔋 **Higher battery usage**: Emulation is CPU-intensive
- 💾 **Memory usage**: ~4-8GB RAM for Docker + CARLA
- 🌡️ **Heat**: M1 will run warm under load

### Optimization Tips

1. **Use lower quality settings**:
   ```bash
   -quality-level=Low
   ```

2. **Reduce resolution** in your Python code:
   ```python
   # In main.py
   python main.py --width 640 --height 480
   ```

3. **Limit FPS**:
   ```bash
   -benchmark -fps=20
   ```

4. **Run headless** (already default with `-RenderOffScreen`)

---

## Troubleshooting

### Error: "Cannot connect to Docker daemon"
```bash
# Make sure Docker Desktop is running
open -a Docker
```

### Error: "CARLA server not responding"
```bash
# Check if container is running
docker ps

# Check Docker logs
docker logs $(docker ps -q --filter ancestor=carlasim/carla:0.9.15)
```

### Error: "Port 2000 already in use"
```bash
# Find what's using the port
lsof -i :2000

# Or use different ports
docker run -p 3000:2000 ...
# Then connect with: python main.py --port 3000
```

### Error: "No module named 'carla'"
```bash
# Try the PYTHONPATH approach
export PYTHONPATH="${PYTHONPATH}:/path/to/carla/PythonAPI/carla"

# Or install the .egg file directly
easy_install /path/to/carla-0.9.15-py3.7-linux-x86_64.egg
```

### Slow Performance
- Close other applications
- Ensure Docker has enough resources (Settings → Resources)
- Use lower quality settings
- Reduce image resolution

---

## Alternative: Remote Linux Machine

If Docker performance is too slow:

1. **Set up CARLA on a Linux machine** (lab computer, cloud VM, etc.)
2. **Connect remotely** from your M1 Mac:
   ```python
   # In your code, just change the host
   python main.py --host <remote-ip> --port 2000
   ```

This gives you native CARLA performance while developing on M1!

---

## Verification Checklist

- [ ] Docker Desktop installed and running
- [ ] Rosetta 2 enabled in Docker settings
- [ ] CARLA Docker image pulled
- [ ] CARLA server starts without errors
- [ ] Python can import carla module
- [ ] Can connect to CARLA from Python
- [ ] Lane detection runs successfully

---

## Next Steps

1. ✅ Get Docker + CARLA running
2. ✅ Test connection with `test_carla_connection.py`
3. ✅ Run lane detection: `python main.py`
4. 🚀 Start developing!

For the PiRacer deployment, you won't need Docker since it will use real cameras!
