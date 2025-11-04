# Visualization Guide 🎨

## The Problem

You mentioned the camera view disappeared when using XQuartz. This is because:
1. Distributed mode doesn't send debug images over network (too heavy)
2. OpenCV's X11 forwarding has issues with XQuartz/remote setups
3. Docker/headless environments need different solutions

## The Solution

**Three visualization backends** with auto-detection!

```
┌─────────────────────────────────────────────────────────┐
│          VISUALIZATION OPTIONS                          │
└─────────────────────────────────────────────────────────┘

1. OpenCV (cv2)      - Default, works with local X11
2. Pygame            - Better X11 forwarding support
3. Web Browser       - Best for remote/Docker! ⭐
4. None (headless)   - No visualization
```

---

## Quick Start

### Option 1: Auto-Select (Recommended)
```bash
python main_distributed_v2.py --detector-url tcp://localhost:5555
# Automatically picks best viewer for your environment
```

### Option 2: Web Viewer (Best for Remote!) ⭐
```bash
# Start with web viewer
python main_distributed_v2.py --viewer web --detector-url tcp://localhost:5555

# Then open in browser:
http://localhost:8080
```

### Option 3: Pygame (Better X11)
```bash
# Install pygame first
pip install pygame

# Use pygame viewer
python main_distributed_v2.py --viewer pygame --detector-url tcp://localhost:5555
```

### Option 4: Headless (No Display)
```bash
python main_distributed_v2.py --viewer none --detector-url tcp://localhost:5555
```

---

## Detailed Comparison

| Viewer | Best For | Pros | Cons | XQuartz Support |
|--------|---------|------|------|-----------------|
| **OpenCV** | Local development | Fast, default | X11 issues | ⚠️ Poor |
| **Pygame** | X11 forwarding | Better X11 | Need pygame | ✅ Good |
| **Web** | Remote/Docker | No X11 needed! | Port forwarding | ✅ Excellent |
| **None** | Headless servers | Lightweight | No visualization | N/A |

---

## Web Viewer (Recommended for Remote) 🌐

### Why Web Viewer is Best for You:

```
Traditional (OpenCV):
   Your Machine ──X11 over SSH──> Remote Server ❌ Slow/Broken

Web Viewer:
   Your Browser ──HTTP──> Remote Server ✅ Fast & Reliable
```

### Usage:

```bash
# Terminal 1: Detection server
python detection_server.py --method cv --port 5555

# Terminal 2: CARLA client with web viewer
python main_distributed_v2.py \
    --viewer web \
    --web-port 8080 \
    --detector-url tcp://localhost:5555

# Terminal 3: Open browser
firefox http://localhost:8080
# Or from your local machine:
firefox http://your-server-ip:8080
```

### SSH Tunneling:

If running on remote server:
```bash
# On your local machine:
ssh -L 8080:localhost:8080 user@remote-server

# Then access:
http://localhost:8080
```

---

## Installation

### For OpenCV (default):
```bash
pip install opencv-python
```

### For Pygame:
```bash
pip install pygame
```

### For Web Viewer:
```bash
# No additional install needed!
# Uses built-in Python http.server
```

---

## Environment-Specific Recommendations

### Local Development (Linux/Mac with X11)
```bash
python main_distributed_v2.py --viewer opencv --detector-url tcp://localhost:5555
```

### Remote via SSH (XQuartz, VNC)
```bash
# Option A: Pygame (better X11 forwarding)
python main_distributed_v2.py --viewer pygame --detector-url tcp://localhost:5555

# Option B: Web (no X11 needed!)
python main_distributed_v2.py --viewer web --detector-url tcp://localhost:5555
```

### Docker Container
```bash
# Web viewer is perfect for Docker!
docker run -p 8080:8080 -p 5555:5555 your-image \
    python main_distributed_v2.py \
    --viewer web \
    --web-port 8080 \
    --detector-url tcp://localhost:5555

# Access from host:
http://localhost:8080
```

### WSL (Windows Subsystem for Linux)
```bash
# Auto-detection will choose web viewer
python main_distributed_v2.py --viewer auto --detector-url tcp://localhost:5555

# Or explicitly:
python main_distributed_v2.py --viewer web --detector-url tcp://localhost:5555
```

### Headless Server (No Display)
```bash
python main_distributed_v2.py --viewer none --detector-url tcp://localhost:5555
```

---

## Auto-Detection Logic

The `--viewer auto` mode automatically selects:

```python
if no DISPLAY environment variable:
    → Use Web Viewer

elif WSL detected:
    → Use Web Viewer

elif pygame available:
    → Use Pygame

else:
    → Use OpenCV
```

---

## Command Reference

### Full Command Options:

```bash
python main_distributed_v2.py [OPTIONS]

Visualization Options:
  --viewer {auto,opencv,pygame,web,none}
                        Visualization backend (default: auto)
  --web-port PORT      Port for web viewer (default: 8080)
  --no-display         Disable visualization entirely

CARLA Options:
  --host HOST          CARLA server host
  --port PORT          CARLA server port
  --spawn-point INDEX  Vehicle spawn point

Detection Options:
  --detector-url URL   Detection server URL
  --detector-timeout MS  Request timeout (default: 1000ms)

Examples:
  # Auto-select viewer
  python main_distributed_v2.py --detector-url tcp://localhost:5555

  # Web viewer on custom port
  python main_distributed_v2.py --viewer web --web-port 9000 --detector-url tcp://localhost:5555

  # Pygame with remote CARLA
  python main_distributed_v2.py --viewer pygame --host 192.168.1.100 --detector-url tcp://localhost:5555

  # Headless mode
  python main_distributed_v2.py --viewer none --detector-url tcp://localhost:5555
```

---

## Troubleshooting

### "Cannot open display"
```bash
# Use web viewer instead of OpenCV
python main_distributed_v2.py --viewer web --detector-url tcp://localhost:5555
```

### "No module named 'pygame'"
```bash
pip install pygame
# Or use web viewer:
python main_distributed_v2.py --viewer web --detector-url tcp://localhost:5555
```

### Web viewer not accessible from browser
```bash
# Check firewall
sudo ufw allow 8080/tcp

# Or use SSH tunnel
ssh -L 8080:localhost:8080 user@server
```

### Slow/laggy visualization
```bash
# Web viewer has lower FPS than OpenCV
# For better performance, use pygame locally:
python main_distributed_v2.py --viewer pygame --detector-url tcp://localhost:5555
```

---

## Performance Comparison

| Viewer | FPS | Latency | CPU Usage | Network |
|--------|-----|---------|-----------|---------|
| OpenCV | 30+ | Very Low | Low | Local only |
| Pygame | 30+ | Very Low | Low | Local only |
| Web | 15-30 | Low | Medium | HTTP stream |
| None | N/A | N/A | Minimal | N/A |

---

## Migration Guide

### From Old main_distributed.py:

**Before:**
```bash
python main_distributed.py --detector-url tcp://localhost:5555
# Uses OpenCV only, XQuartz issues
```

**After:**
```bash
# Auto-detect best viewer
python main_distributed_v2.py --detector-url tcp://localhost:5555

# Or explicitly use web viewer
python main_distributed_v2.py --viewer web --detector-url tcp://localhost:5555
```

---

## Summary

**For Your XQuartz Setup, Use:**

```bash
# Recommended: Web Viewer (no X11 needed!)
python main_distributed_v2.py \
    --viewer web \
    --web-port 8080 \
    --detector-url tcp://localhost:5555

# Then open in browser:
http://localhost:8080
```

**Benefits:**
✅ No XQuartz issues
✅ Works in browser
✅ SSH tunnel friendly
✅ Docker compatible
✅ Beautiful web UI

Enjoy smooth visualization! 🎨🚗
