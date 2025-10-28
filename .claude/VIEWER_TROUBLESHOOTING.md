# Visualization Viewer Troubleshooting

## Your Issue: Pygame GLX Error

**Error Message:**
```
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
X Error of failed request:  GLXBadContext
  Major opcode of failed request:  149 (GLX)
  Minor opcode of failed request:  6 (X_GLXIsDirect)
```

**Root Cause:** Pygame tries to use OpenGL/GLX hardware acceleration, which isn't available in Docker/remote environments without proper GPU drivers.

---

## Solutions (In Order of Recommendation)

### 🥇 Solution 1: Use Web Viewer (BEST for Docker/Remote)

**Why:** Works without X11, OpenGL, or any display drivers. View in your browser!

```bash
python simulation/main.py --viewer web --web-port 8080
```

Then open: `http://localhost:8080` in your browser

**Pros:**
- ✅ No X11 required
- ✅ No OpenGL required
- ✅ Works in any environment (Docker, WSL, remote)
- ✅ Easy to share (just forward port)
- ✅ Multiple viewers possible

**Cons:**
- ❌ Slight latency (~100ms)
- ❌ Requires browser

---

### 🥈 Solution 2: Use OpenCV Viewer (Software Rendering)

**Why:** OpenCV uses software rendering by default, no OpenGL needed

```bash
python simulation/main.py --viewer opencv
```

**Pros:**
- ✅ No OpenGL required
- ✅ Simple and reliable
- ✅ Works with basic X11

**Cons:**
- ❌ Requires X11 forwarding
- ❌ Can be slow on large images

---

### 🥉 Solution 3: Auto-Select (Now Fixed)

**Why:** Automatically picks best viewer for your environment

```bash
python simulation/main.py --viewer auto
```

**What it does now:**
1. Detects Docker → Web viewer
2. Detects no DISPLAY → Web viewer
3. Detects WSL → Web viewer
4. Otherwise → OpenCV viewer (more stable than pygame)

---

### 🔧 Solution 4: Fix Pygame (If You Really Want It)

#### Option A: Install Mesa Software Rendering
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libgl1-mesa-dri mesa-utils
```

#### Option B: Set Environment Variables
```bash
export SDL_VIDEODRIVER=x11
export SDL_RENDER_DRIVER=software
export LIBGL_ALWAYS_SOFTWARE=1

python simulation/main.py --viewer pygame
```

#### Option C: Use Our Fixed Pygame
We've updated pygame_viewer.py to use software rendering automatically. Try again:
```bash
python simulation/main.py --viewer pygame
```

If it still fails, fall back to OpenCV or Web viewer.

---

## Environment Detection

### Check Your Environment

**Are you in Docker?**
```bash
ls /.dockerenv  # If exists, you're in Docker
```

**Do you have X11?**
```bash
echo $DISPLAY  # Should show something like :0 or localhost:10.0
```

**Do you have OpenGL?**
```bash
glxinfo | grep "OpenGL"  # Should show OpenGL info
```

---

## Viewer Comparison

| Viewer | X11 Needed | OpenGL Needed | Docker-Friendly | Performance | Best For |
|--------|------------|---------------|-----------------|-------------|----------|
| **Web** | ❌ No | ❌ No | ✅ Yes | Good | Remote, Docker, WSL |
| **OpenCV** | ✅ Yes | ❌ No | ⚠️ Maybe | Good | Local X11 |
| **Pygame** | ✅ Yes | ⚠️ Maybe | ❌ No | Best | Native Linux with GPU |
| **None** | ❌ No | ❌ No | ✅ Yes | N/A | Headless testing |

---

## Recommended Usage by Environment

### Docker/Container
```bash
python simulation/main.py --viewer web
```

### WSL2 (Windows Subsystem for Linux)
```bash
python simulation/main.py --viewer web
```

### Remote SSH with X11 Forwarding
```bash
# Option 1: Web viewer (easier)
python simulation/main.py --viewer web

# Option 2: OpenCV with X11 (requires: ssh -X)
python simulation/main.py --viewer opencv
```

### Native Linux Desktop
```bash
# Auto-select will pick opencv
python simulation/main.py --viewer auto
```

### Headless (No Display)
```bash
python simulation/main.py --viewer none
```

---

## Fixes Applied

### 1. Pygame Viewer Improvements
**File:** `/workspaces/ads_ld/simulation/ui/pygame_viewer.py`

**Changes:**
- Force software rendering: `SDL_RENDER_DRIVER=software`
- Use software surface: `pygame.SWSURFACE`
- Better error handling with fallback suggestion

### 2. Auto-Selection Improvements
**File:** `/workspaces/ads_ld/simulation/integration/visualization.py`

**Changes:**
- Detect Docker/container
- Check for OpenGL availability
- Prefer OpenCV over Pygame (more stable)
- Better error handling for pygame failures

### 3. Fallback Chain
Now if pygame fails, it automatically falls back to OpenCV:

```python
try:
    viewer = PygameViewer()
except ImportError:
    print("Pygame not installed, using OpenCV")
    viewer = OpenCVViewer()
except Exception as e:
    print(f"Pygame failed: {e}, using OpenCV")
    viewer = OpenCVViewer()
```

---

## Quick Test

### Test Web Viewer
```bash
python simulation/main.py --viewer web --no-sync --base-throttle 0.3
```
Open: http://localhost:8080

### Test OpenCV Viewer
```bash
python simulation/main.py --viewer opencv --no-sync --base-throttle 0.3
```

### Test Pygame (Fixed)
```bash
python simulation/main.py --viewer pygame --no-sync --base-throttle 0.3
```

---

## For Your Specific Case

Based on your error, you're likely in a **Docker/remote environment without OpenGL**.

### Recommended Command:
```bash
python simulation/main.py --viewer web --web-port 8080 --base-throttle 0.3
```

Then open `http://localhost:8080` in your browser!

### Alternative (If X11 works):
```bash
python simulation/main.py --viewer opencv --base-throttle 0.3
```

---

## Still Having Issues?

### Check These:

1. **DISPLAY variable set?**
   ```bash
   echo $DISPLAY
   ```

2. **X11 forwarding working?**
   ```bash
   xclock  # Should show a clock window
   ```

3. **Port 8080 available? (for web viewer)**
   ```bash
   netstat -tuln | grep 8080
   ```

4. **Firewall blocking?**
   ```bash
   # Check if you can access localhost:8080
   curl http://localhost:8080
   ```

---

## Summary

**Your Issue:** Pygame + Docker/remote = GLX error ❌

**Quick Fix:** Use web viewer ✅
```bash
python simulation/main.py --viewer web
```

**Why:** Web viewer doesn't need X11, OpenGL, or any display drivers. It just works! 🚀
