# PyProject.toml Updates Summary

**Date:** 2025-11-03

## Changes Made

### 1. Added New Entry Point: `zmq-viewer`

**File:** `pyproject.toml`

```toml
[project.scripts]
lane-detection = "detection.run:main"
simulation = "simulation.run:main"
zmq-viewer = "viewer.zmq_web_viewer:main"  # NEW!
```

**Usage:**
```bash
# After pip install -e .
zmq-viewer --vehicle tcp://localhost:5557 --port 8080
```

### 2. Added New Package: `viewer`

**File:** `pyproject.toml`

```toml
[tool.setuptools]
packages = ["detection", "simulation", "decision", "viewer", "utils"]
#                                                    ^^^^^^^^ NEW!
```

### 3. Created Viewer Module Files

```
viewer/
├── __init__.py              # Package exports
├── zmq_web_viewer.py        # Main implementation
└── README.md                # Documentation
```

---

## Installation

### Fresh Install

```bash
# Clone/pull latest changes
git pull

# Reinstall package (picks up new entry points)
pip install -e .
```

### Verify Installation

```bash
# Check installed entry points
which zmq-viewer
# Should show: /path/to/venv/bin/zmq-viewer

# Test help
zmq-viewer --help
```

---

## Available Commands

After installation, you have **3 command-line tools**:

### 1. `lane-detection`

**Purpose:** Standalone detection server

**Usage:**
```bash
lane-detection --method cv --port 5556
lane-detection --method dl --gpu 0 --port 5555
```

### 2. `simulation`

**Purpose:** Main CARLA simulation

**Usage:**
```bash
# Classic mode (with local viewer)
simulation --viewer web --web-port 8080

# Production mode (with ZMQ broadcasting)
simulation --viewer none --zmq-broadcast
```

### 3. `zmq-viewer` (NEW!)

**Purpose:** Remote web viewer (runs on laptop)

**Usage:**
```bash
# Connect to local simulation
zmq-viewer --vehicle tcp://localhost:5557 --port 8080

# Connect to remote vehicle
zmq-viewer --vehicle tcp://192.168.1.100:5557 --port 8080
```

---

## Complete Workflow

### Development (Same Machine)

```bash
# Terminal 1: CARLA
./CarlaUE4.sh

# Terminal 2: Detection
lane-detection --method cv --port 5556

# Terminal 3: Simulation
simulation \
    --detector-url tcp://localhost:5556 \
    --viewer none \
    --zmq-broadcast

# Terminal 4: Viewer
zmq-viewer --port 8080

# Browser: http://localhost:8080
```

### Production (Vehicle + Laptop)

**On Vehicle:**
```bash
# Terminal 1: Detection
lane-detection --method dl --gpu 0

# Terminal 2: Control
simulation \
    --detector-url tcp://localhost:5556 \
    --viewer none \
    --zmq-broadcast \
    --broadcast-url tcp://*:5557
```

**On Laptop:**
```bash
zmq-viewer --vehicle tcp://192.168.1.100:5557 --port 8080
```

---

## Package Structure

```
seame-ads/
├── pyproject.toml
│
├── detection/
│   ├── run.py  ──────────► lane-detection command
│   └── ...
│
├── simulation/
│   ├── run.py  ──────────► simulation command
│   ├── integration/
│   │   ├── zmq_broadcast.py     # Pub-sub broadcasting
│   │   ├── shared_memory.py     # Ultra-low latency
│   │   └── communication.py     # Req-rep (existing)
│   └── ...
│
├── viewer/
│   ├── zmq_web_viewer.py  ►  zmq-viewer command (NEW!)
│   ├── __init__.py
│   └── README.md
│
├── decision/
│   └── ...
│
└── utils/
    └── ...
```

---

## Dependencies

All required dependencies are already in `pyproject.toml`:

- ✅ `pyzmq>=25.1.0` - ZMQ communication
- ✅ `numpy>=1.24.0` - Arrays
- ✅ `opencv-python>=4.8.0` - Image processing
- ✅ (All other existing deps)

**No new dependencies needed!**

---

## Migration from Old Code

### Before (Direct Python)

```bash
# Old way
python viewer/zmq_web_viewer.py --vehicle tcp://localhost:5557
```

### After (Entry Point)

```bash
# New way (cleaner!)
zmq-viewer --vehicle tcp://localhost:5557
```

Both work, but entry point is:
- ✅ Shorter
- ✅ Always available after install
- ✅ Works from any directory

---

## Troubleshooting

### Problem: `zmq-viewer: command not found`

**Solution:** Reinstall package
```bash
pip install -e .
```

### Problem: `ModuleNotFoundError: No module named 'viewer'`

**Solution:** Ensure viewer module exists
```bash
ls viewer/
# Should show: __init__.py, zmq_web_viewer.py, README.md
```

### Problem: Old entry points still pointing to wrong location

**Solution:** Force reinstall
```bash
pip uninstall seame-ads
pip install -e .
```

---

## Testing

### Test Entry Points

```bash
# Test all commands are available
which lane-detection
which simulation
which zmq-viewer

# Test help for each
lane-detection --help
simulation --help
zmq-viewer --help
```

### Test Functionality

```bash
# Start CARLA
./CarlaUE4.sh &

# Start detection
lane-detection --method cv --port 5556 &

# Start simulation
simulation --viewer none --zmq-broadcast &

# Start viewer
zmq-viewer --port 8080

# Open browser
firefox http://localhost:8080
```

---

## Documentation References

- [New Architecture Guide](NEW_ARCHITECTURE.md)
- [Quick Start Guide](QUICKSTART_NEW_ARCHITECTURE.md)
- [Viewer README](../viewer/README.md)
- [Main README](../README.md)

---

## Summary

✅ **Added:** `zmq-viewer` command-line entry point
✅ **Added:** `viewer` package to setuptools
✅ **Updated:** Main README with new commands
✅ **Created:** Viewer module documentation

**Everything is ready for production deployment!** 🚀
