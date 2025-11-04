# Shared Memory Quickstart Guide

**Ultra-Low Latency Lane Detection with Shared Memory IPC**

## Overview

This guide shows you how to run the lane detection system with **shared memory** instead of ZMQ for ultra-low latency communication (~0.001ms vs ~2ms).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Same Machine                             │
│                                                             │
│  ┌──────────────┐   Shared Memory    ┌─────────────────┐  │
│  │  Simulation/ │◄──────────────────►│   Detection     │  │
│  │  Camera      │    ~0.001ms !      │   Process       │  │
│  │              │                    │                 │  │
│  │  Writes:     │                    │   Reads:        │  │
│  │  - Images    │                    │   - Images      │  │
│  │              │                    │                 │  │
│  │  Reads:      │                    │   Writes:       │  │
│  │  - Detections│                    │   - Detections  │  │
│  └──────────────┘                    └─────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ **2000x faster** than ZMQ TCP (0.001ms vs 2ms)
- ✅ **Zero-copy** memory access
- ✅ **Perfect for real-time** control loops
- ✅ **Process isolation** (reliability)

**Limitations:**
- ⚠️ **Same machine only** (both processes must be on same computer)
- ⚠️ Cannot run detection on remote GPU server

---

## Quick Start

### Start Both Processes (Any Order!)

**Good news:** Both processes can now be started in **any order**! The system will automatically retry and wait for the other process.

#### Option 1: Start Detection Server First

```bash
# Terminal 1: Start detection server with shared memory
lane-detection --method cv --shared-memory

# Terminal 2: Start CARLA simulation (will connect automatically)
simulation --shared-memory --viewer web
```

#### Option 2: Start Simulation First (Works Too!)

```bash
# Terminal 1: Start simulation (will wait for detection server)
simulation --shared-memory --viewer web

# Terminal 2: Start detection server (simulation will connect automatically)
lane-detection --method cv --shared-memory
```

#### Option 3: Start Both Simultaneously

```bash
# Start both at the same time - they'll find each other!
lane-detection --method cv --shared-memory & simulation --shared-memory --viewer web
```

### What You'll See

**Detection Server Output:**
```
============================================================
Lane Detection Server
============================================================

Initializing CV detector...
✓ Detector ready: CV Lane Detector

Using SHARED MEMORY mode (ultra-low latency ~0.001ms)
  Image input: camera_feed
  Detection output: detection_results

Connecting to image shared memory 'camera_feed'...
  Waiting for shared memory 'camera_feed' (attempt 1/20)...
  Waiting for shared memory 'camera_feed' (attempt 2/20)...
✓ Connected to shared memory: camera_feed

Creating detection shared memory 'detection_results'...
✓ Created detection shared memory: detection_results (88 bytes)

============================================================
Server initialized successfully!
============================================================
```

**Simulation Output:**
```
============================================================
DISTRIBUTED LANE KEEPING SYSTEM - ENHANCED
============================================================
CARLA Server: localhost:2000
Detection Mode: SHARED MEMORY (ultra-low latency ~0.001ms)
  Image output: camera_feed
  Detection input: detection_results
Camera: 800x600
============================================================

[5/5] Setting up detection communication...
Setting up shared memory communication...
(Both processes can start in any order - will retry if needed)

Creating shared memory image writer...
✓ Created shared memory: camera_feed (1440000 bytes)

Connecting to detection results...
  Waiting for detection shared memory 'detection_results' (attempt 1/20)...
  Waiting for detection shared memory 'detection_results' (attempt 2/20)...
✓ Connected to detection shared memory: detection_results

✓ Shared memory communication ready
```

### Step 3: (Optional) Remote Viewer

If you want to monitor from a laptop while using shared memory for detection:

```bash
# Terminal 3: Remote web viewer (on laptop or same machine)
python viewer/zmq_web_viewer.py --vehicle tcp://localhost:5557
```

Then open browser: http://localhost:8080

---

## Command Reference

### Detection Server

```bash
lane-detection [OPTIONS]

# Shared Memory Options
--shared-memory              Use shared memory mode (ultra-low latency)
--image-shm-name NAME        Shared memory name for images (default: camera_feed)
--detection-shm-name NAME    Shared memory name for detections (default: detection_results)

# General Options
--method {cv,dl}            Detection method (default: cv)
--config PATH               Configuration file path
--gpu ID                    GPU device ID (for DL method)

# ZMQ Mode Options (ignored in shared memory mode)
--port PORT                 Port to listen on (default: 5556)
--host HOST                 Host to bind to (default: *)
```

### Simulation

```bash
simulation [OPTIONS]

# Shared Memory Options
--shared-memory              Use shared memory for detection
--image-shm-name NAME        Shared memory name for images (default: camera_feed)
--detection-shm-name NAME    Shared memory name for detections (default: detection_results)

# ZMQ Detection Options (used if not --shared-memory)
--detector-url URL           Detection server URL (default: tcp://localhost:5556)
--detector-timeout MS        Detection timeout in ms (default: 1000)

# Viewer Options
--viewer {auto,opencv,pygame,web,none}  Viewer type (default: auto)
--web-port PORT              Port for web viewer (default: 8080)

# ZMQ Broadcasting Options (for remote viewer)
--zmq-broadcast              Enable ZMQ broadcasting for remote viewer
--broadcast-url URL          ZMQ broadcast URL (default: tcp://*:5557)
--broadcast-images           Also broadcast raw images (HIGH BANDWIDTH!)
```

---

## Performance Comparison

### Latency Breakdown (Shared Memory vs ZMQ)

| Component | ZMQ Mode | Shared Memory Mode | Improvement |
|-----------|----------|-------------------|-------------|
| Camera → Detection | **~2ms** | **~0.001ms** | **2000x faster** |
| Detection Processing | ~15ms | ~15ms | Same |
| Detection → Control | **~2ms** | **~0.001ms** | **2000x faster** |
| **Total E2E** | **~19ms** | **~15ms** | **21% faster** |

### When to Use Each Mode

**Use Shared Memory When:**
- ✅ Both processes on same machine
- ✅ Need ultra-low latency (<1ms)
- ✅ Real-time control critical
- ✅ Running on vehicle (Raspberry Pi, Jetson)

**Use ZMQ When:**
- ✅ Detection server on different machine (remote GPU)
- ✅ Need network communication
- ✅ Testing/debugging remotely
- ✅ Multiple clients need detection

---

## Troubleshooting

### Error: "Shared memory not found after 20 attempts"

**Problem:** The other process is not running at all.

**Solution:**
1. Make sure BOTH processes are running:
   ```bash
   # Check if detection server is running
   ps aux | grep lane-detection

   # Check if simulation is running
   ps aux | grep simulation
   ```
2. Start the missing process
3. The waiting process will automatically connect once both are running

### Error: "Failed to create shared memory" or "Permission denied"

**Problem:** Old shared memory blocks from previous crash or permission issues.

**Solution:** Clean up old shared memory:
```bash
# List shared memory blocks
ls -la /dev/shm/

# Remove old blocks (if they exist)
rm /dev/shm/camera_feed
rm /dev/shm/detection_results

# If permission denied, use sudo
sudo rm /dev/shm/camera_feed /dev/shm/detection_results
```

### Performance Not Improving

**Problem:** Bottleneck elsewhere (not communication).

**Check:**
1. Use `--latency` flag to see breakdown:
   ```bash
   simulation --shared-memory --latency
   ```
2. Look for bottleneck in latency report:
   - If "Detection Processing" is highest → CPU/GPU bottleneck
   - If "Network Round-trip" is highest → Check detection server

### Processes Not Communicating

**Problem:** Name mismatch between processes.

**Solution:**
Ensure both use the same shared memory names:
```bash
# Both must use the same names
lane-detection --shared-memory --image-shm-name my_camera --detection-shm-name my_results
simulation --shared-memory --image-shm-name my_camera --detection-shm-name my_results
```

**Note:** Start order doesn't matter - they will wait for each other!

---

## Advanced: Hybrid Mode (Shared Memory + ZMQ Broadcast)

For the **best of both worlds**: ultra-low latency detection + remote monitoring.

```bash
# Terminal 1: Detection server (shared memory)
lane-detection --method cv --shared-memory

# Terminal 2: Simulation (shared memory + ZMQ broadcast)
simulation --shared-memory --viewer none --zmq-broadcast

# Terminal 3: Remote viewer (laptop)
python viewer/zmq_web_viewer.py --vehicle tcp://vehicle-ip:5557
```

**Architecture:**
```
Vehicle (Raspberry Pi)
┌────────────────────────────────────┐
│  Camera ←→ Detection (shared mem)  │ ← Ultra-fast!
│      ↓                             │
│    ZMQ Broadcast ──────────────────┼──→ Laptop Viewer
└────────────────────────────────────┘
```

**Performance:**
- Camera → Detection: 0.001ms (shared memory)
- Vehicle → Laptop: 5ms (ZMQ, non-blocking)
- No impact on control loop!

---

## Integration with Existing Code

### Python API

```python
from simulation.integration.shared_memory_detection import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionClient,
)

# Write images
image_writer = SharedMemoryImageChannel(
    name="camera_feed",
    shape=(600, 800, 3),
    create=True
)
image_writer.write(image, timestamp=time.time(), frame_id=0)

# Read detections
detector = SharedMemoryDetectionClient(
    detection_shm_name="detection_results"
)
detection = detector.get_detection(timeout=1.0)
```

---

## Next Steps

1. **Test Performance:**
   ```bash
   simulation --shared-memory --latency
   ```
   Compare latency reports with ZMQ mode.

2. **Real Vehicle Deployment:**
   - Install on Raspberry Pi / Jetson
   - Use shared memory for critical path
   - Add ZMQ broadcast for monitoring

3. **Optimize Further:**
   - Increase detection FPS (faster GPU)
   - Reduce image resolution (if acceptable)
   - Use multiple buffers (double-buffering)

---

## Summary

**Shared memory mode gives you:**
- 🚀 **2000x faster** communication
- 🎯 **Perfect for real-time** control
- 💪 **Production-ready** for real vehicles
- 🔧 **Easy to use** (just add `--shared-memory`)

**Try it now:**
```bash
lane-detection --method cv --shared-memory
simulation --shared-memory --viewer web
```

Enjoy ultra-low latency lane detection! 🏎️💨
