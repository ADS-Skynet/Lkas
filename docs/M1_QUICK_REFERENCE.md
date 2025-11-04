# M1/M2 Mac Quick Reference

## TL;DR - Get Started in 3 Steps

### 1. Install Docker Desktop
- Download from [docker.com](https://www.docker.com/products/docker-desktop/)
- Enable **Rosetta 2** in Settings → Beta Features

### 2. Start CARLA Server
```bash
cd /Users/donghyun/All/seame/ads_ld
./start_carla.sh
```

### 3. Run Lane Detection (in new terminal)
```bash
cd /Users/donghyun/All/seame/ads_ld/lane_detection
python test_carla_connection.py  # Test first
python main.py --method cv        # Then run lane detection
```

---

## Why Docker?

❌ **What doesn't work on M1:**
- Native CARLA (no ARM64 build)
- CARLA Python .whl files (x86_64 only)
- Building from source (complex, unsupported)

✅ **What works:**
- CARLA server in Docker (Rosetta 2 emulation)
- Python client runs natively on M1
- Connection over localhost

---

## Daily Workflow

### Morning Setup
```bash
# Terminal 1: Start CARLA
cd /Users/donghyun/All/seame/ads_ld
./start_carla.sh

# Terminal 2: Develop
cd lane_detection
python main.py --method cv
```

### Using Docker Compose (Alternative)
```bash
# Start (runs in background)
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Common Commands

### Check CARLA Status
```bash
# Is Docker running?
docker info

# Is CARLA container running?
docker ps

# View CARLA logs
docker logs $(docker ps -q --filter ancestor=carlasim/carla:0.9.15)
```

### Stop CARLA
```bash
# If started with start_carla.sh
Ctrl + C

# If started with docker-compose
docker-compose down

# Force stop
docker stop carla-server
```

### Clean Up Docker
```bash
# Remove stopped containers
docker container prune

# Remove CARLA image (to save space)
docker rmi carlasim/carla:0.9.15

# Free up all unused Docker resources
docker system prune -a
```

---

## Troubleshooting

### "Cannot connect to CARLA"
```bash
# 1. Check if Docker is running
open -a Docker

# 2. Check if CARLA container is running
docker ps

# 3. Test connection
cd lane_detection
python test_carla_connection.py
```

### "CARLA is slow"
```bash
# Use lower quality
./start_carla.sh Low

# Or edit docker-compose.yml:
# -quality-level=Low
# -benchmark -fps=15
```

### "Port 2000 already in use"
```bash
# Find what's using it
lsof -i :2000

# Kill it
kill -9 <PID>

# Or use different port
docker run -p 3000:2000 ...
python main.py --port 3000
```

### "No module named 'carla'"
```bash
# Try installing
pip install carla==0.9.15

# If that fails, use the source you already have
export PYTHONPATH="${PYTHONPATH}:/Users/donghyun/All/seame/ads_ld/carla"
```

---

## Performance Tips

### Battery Life
- Close other apps
- Use Low quality setting
- Lower FPS: `-fps=15`

### Speed
- Reduce resolution: `python main.py --width 640 --height 480`
- Use headless mode (already default)
- Allocate more RAM to Docker (Settings → Resources)

### Heat
- Use external cooling
- Run quality-level=Low
- Limit CPU cores in docker-compose.yml

---

## File Reference

| File | Purpose |
|------|---------|
| `MACOS_M1_SETUP.md` | Detailed setup guide |
| `start_carla.sh` | Quick start CARLA |
| `docker-compose.yml` | Docker Compose config |
| `test_carla_connection.py` | Test CARLA connection |
| `test_setup.py` | Test without CARLA |

---

## Alternative: Remote CARLA

If Docker is too slow:

1. **Run CARLA on a Linux machine** (lab computer, cloud VM)
2. **Connect from your M1 Mac:**
   ```bash
   python main.py --host <remote-ip> --port 2000
   ```

This gives native CARLA performance!

---

## Next Steps After CARLA Works

1. ✅ Test with `test_carla_connection.py`
2. ✅ Run lane detection: `python main.py`
3. 🎨 Tune detection parameters
4. 📊 Collect training data for DL model
5. 🚗 Eventually: Adapt for PiRacer (no Docker needed!)

---

## Help

- Full setup: [MACOS_M1_SETUP.md](MACOS_M1_SETUP.md)
- Quick start: [QUICK_START.md](QUICK_START.md)
- Module docs: [lane_detection/README.md](lane_detection/README.md)
