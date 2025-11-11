# Autonomous Driving Skynet - Lane Keeping System

**Real-time lane keeping assist system with WebSocket-powered monitoring and live parameter tuning for CARLA simulator.**

## 🌟 Features

- **🚀 WebSocket Real-Time Streaming** - Binary frame transmission with 50-300ms latency
- **🎮 Live Parameter Tuning** - Adjust detection and PID parameters on-the-fly
- **📡 ZMQ Broker Architecture** - Distributed communication between modules
- **🔄 Multiple Detection Methods** - OpenCV (CV), YOLO, YOLO-Seg
- **🎯 PID Controller** - Smooth steering with configurable gains
- **🌐 Remote Web Viewer** - Monitor from any browser, no X11 required
- **⚡ Low Latency** - Optimized performance with frame rate limiting
- **🔧 Production Ready** - Process isolation, fault tolerance, comprehensive logging

## 📦 Installation

### Prerequisites
- Python 3.10+
- CARLA 0.9.15+ simulator
- GPU (optional, for YOLO detection)

### Quick Install

```bash
# Clone repository
git clone <repository-url>
cd ads_skynet

# Install package with all dependencies
pip install -e .

# Verify installation
lkas --help
simulation --help
viewer --help
```

This installs the `ads-skynet` package with three main entry points:
- `lkas` - Lane Keeping Assist System (detection + decision + broker)
- `simulation` - CARLA simulation orchestrator
- `viewer` - WebSocket-powered web viewer

## 🚀 Quick Start

### Full System Setup

```bash
# Terminal 1: Start CARLA simulator
cd ~/carla
./CarlaUE4.sh

# Terminal 2: Start LKAS (detection + decision + ZMQ broker)
cd ~/ads_skynet
lkas --method cv --broadcast

# Terminal 3: Start simulation (connects to LKAS via ZMQ)
simulation --broadcast

# Terminal 4: Start web viewer (optional, for monitoring)
viewer

# Open browser: http://localhost:8080
```

### What You'll See

**Web Viewer Dashboard:**
- 🎥 Live video stream with lane overlays
- 📊 Real-time FPS and latency metrics
- 🎛️ Interactive parameter sliders
- 🔘 Control buttons (Pause/Resume/Respawn)
- 🟢 Connection status indicator

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA Simulator                          │
│                     (UE4 Engine)                            │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
         Camera Frames             Vehicle Control
               │                          │
┌──────────────┴──────────────────────────┴───────────────────┐
│              Simulation Orchestrator                        │
│  • Spawns vehicle & camera                                  │
│  • Sends frames to LKAS (ZMQ port 5560)                     │
│  • Receives steering from LKAS (ZMQ port 5563)              │
│  • Publishes status to LKAS Broker (ZMQ port 5562)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ZMQ Communication (ports 5560-5563)
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                   LKAS Module (ZMQ Broker)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │  Detection   │→  │   Decision   │→  │   Actuator   │   │
│  │   (Vision)   │   │    (PID)     │   │  (Steering)  │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                                       │           │
│         └───────────────┬───────────────────────┘           │
│                         │                                   │
│                   ZMQ Broker Hub                            │
│              (Coordinates all modules)                      │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ZMQ Broadcasting (ports 5557-5559)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    Frames           Actions          Parameters
  (port 5557)     (port 5558)      (port 5559)
        │                 │                 │
┌───────┴─────────────────┴─────────────────┴─────────────────┐
│                   Viewer Process                            │
│  • Receives data via ZMQ                                    │
│  • Renders overlays on laptop                               │
│  • Serves WebSocket (binary frames, ~50-100ms latency)     │
│  • HTTP server (port 8080) + WebSocket (port 8081)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    WebSocket
                  (Binary JPEG Frames)
                          │
              ┌───────────┴───────────┐
              │   Web Browser         │
              │  (localhost:8080)     │
              │  • Live video         │
              │  • Parameter tuning   │
              │  • Control buttons    │
              └───────────────────────┘
```

### Data Flow

1. **CARLA** → Camera frames → **Simulation**
2. **Simulation** → Frames via ZMQ → **LKAS Detection**
3. **LKAS Detection** → Lane data → **LKAS Decision**
4. **LKAS Decision** → Steering commands → **Simulation**
5. **Simulation** → Vehicle control → **CARLA**
6. **LKAS Broker** → Broadcasts all data → **Viewer**
7. **Viewer** → WebSocket binary frames → **Browser**
8. **Browser** → Actions/Parameters → **Viewer** → **LKAS Broker**

## 📁 Project Structure

```
ads_skynet/
├── pyproject.toml              # Package configuration & dependencies
├── config.yaml                 # System configuration (auto-loaded)
├── README.md                   # This file
│
├── src/
│   ├── lkas/                   # Lane Keeping Assist System
│   │   ├── run.py              # Main LKAS entry point
│   │   ├── orchestrator.py     # LKAS pipeline coordinator
│   │   │
│   │   ├── detection/          # Lane detection (CV, YOLO)
│   │   │   ├── core/
│   │   │   │   ├── detector.py       # Detection interface
│   │   │   │   ├── config.py         # Configuration management
│   │   │   │   └── models.py         # Lane data models
│   │   │   ├── cv/                   # OpenCV detector
│   │   │   ├── yolo/                 # YOLO detector
│   │   │   └── preprocessing/        # ROI masking, etc.
│   │   │
│   │   ├── decision/           # Steering control
│   │   │   ├── controller.py   # PID controller
│   │   │   └── metrics.py      # Control metrics
│   │   │
│   │   └── integration/        # Communication
│   │       └── zmq/            # ZMQ broker & messaging
│   │           ├── broker.py         # Main ZMQ broker
│   │           ├── broadcaster.py    # Data broadcasting
│   │           └── messages.py       # Message protocols
│   │
│   ├── simulation/             # CARLA Simulation
│   │   ├── run.py              # Simulation entry point
│   │   ├── orchestrator.py     # System coordinator
│   │   │
│   │   ├── carla_api/          # CARLA interface
│   │   │   ├── connection.py   # CARLA connection
│   │   │   ├── vehicle.py      # Vehicle control
│   │   │   └── camera.py       # Camera sensors
│   │   │
│   │   ├── integration/        # LKAS integration
│   │   │   └── zmq_broadcast.py      # ZMQ publishers/subscribers
│   │   │
│   │   └── utils/              # Utilities
│   │       └── visualizer.py   # Overlay rendering
│   │
│   └── viewer/                 # WebSocket Web Viewer
│       ├── run.py              # Viewer entry point
│       ├── frontend.html       # Web interface (HTML/CSS/JS)
│       ├── test_websocket.py   # WebSocket testing tool
│       └── README.md           # Viewer documentation
│
└── docs/                       # Additional documentation
```

## 🎯 Module Responsibilities

### LKAS Module (`src/lkas/`)
- **Detection:** Processes camera frames, detects lane markings
- **Decision:** Analyzes lanes, computes steering via PID controller
- **ZMQ Broker:** Coordinates all communication between modules
- **Broadcasting:** Publishes data to viewer for monitoring

**Entry point:** `lkas --method cv --broadcast`

### Simulation Module (`src/simulation/`)
- **CARLA Integration:** Connects to simulator, spawns vehicle
- **Camera Management:** Sets up sensors, captures frames
- **ZMQ Communication:** Sends frames to LKAS, receives steering
- **Status Publishing:** Broadcasts vehicle telemetry

**Entry point:** `simulation --broadcast`

### Viewer Module (`src/viewer/`)
- **ZMQ Subscription:** Receives data from LKAS broker
- **Rendering:** Draws lane overlays and HUD on laptop
- **WebSocket Server:** Streams binary frames to browser
- **Web Interface:** Provides monitoring and control dashboard

**Entry point:** `viewer`

## ⚙️ Configuration

### config.yaml

The system loads `config.yaml` from the project root:

```yaml
# CARLA Connection
carla:
  host: localhost
  port: 2000
  timeout: 10.0

  vehicle:
    model: vehicle.tesla.model3
    spawn_point: 0  # or null for random

  camera:
    width: 640
    height: 480
    fov: 110

# Detection Parameters
detection:
  method: cv  # cv, yolo, yolo-seg
  cv:
    canny_low: 50
    canny_high: 150
    hough_threshold: 50
    hough_min_line_len: 40
    smoothing_factor: 0.7

# PID Control Parameters
decision:
  kp: 0.5             # Proportional gain
  kd: 0.1             # Derivative gain
  throttle_base: 0.14
  throttle_min: 0.05
  steer_threshold: 0.15

# ZMQ Ports
zmq:
  # LKAS Broker ports
  broker:
    detection_input_port: 5560    # Receive frames from sim
    decision_output_port: 5563    # Send steering to sim
    viewer_data_port: 5557        # Broadcast to viewer
    viewer_action_port: 5558      # Receive actions from viewer
    parameter_update_port: 5559   # Receive parameters from viewer

  # Simulation ports
  simulation:
    detection_output_port: 5560   # Send frames to LKAS
    decision_input_port: 5563     # Receive steering from LKAS
    status_publish_port: 5562     # Publish status to broker

# Viewer Configuration
visualization:
  web_port: 8080  # HTTP server (WebSocket will be port+1)
```

### Custom Configuration

```bash
# Use project root config.yaml (default)
lkas --method cv --broadcast

# Use custom config
lkas --config /path/to/custom-config.yaml --method cv --broadcast

# Override specific settings
simulation --broadcast --spawn-id 123
viewer --port 9090
```

## 🎮 Web Interface Features

### Live Video Stream
- Real-time lane detection overlays
- Vehicle telemetry HUD (speed, steering, position)
- FPS and latency monitoring
- Connection status indicator

### Interactive Controls
- **🔄 Respawn Vehicle** - Reset to spawn point
- **⏸ Pause / ▶ Resume** - Control simulation
- **Keyboard:** `R` for respawn, `Space` for pause/resume

### Live Parameter Tuning

**Detection Parameters (adjustable in real-time):**
- Canny Low Threshold (1-150)
- Canny High Threshold (50-255)
- Hough Threshold (1-150)
- Hough Min Line Length (10-150)
- Smoothing Factor (0-1)

**Decision Parameters (PID tuning):**
- Kp - Proportional gain (0-2)
- Kd - Derivative gain (0-1)
- Base Throttle (0-0.5)
- Min Throttle (0-0.2)
- Steer Threshold (0-0.5)

**All changes apply instantly without restarting!**

## 📊 Performance

### Typical Latencies
- **CARLA → Simulation:** ~5ms
- **Simulation → LKAS:** ~5-10ms (ZMQ)
- **LKAS Detection:** 5-15ms (CV), 20-40ms (YOLO)
- **LKAS Decision:** <1ms
- **LKAS → Simulation:** ~5-10ms (ZMQ)
- **LKAS → Viewer:** ~5ms (ZMQ)
- **Viewer → Browser:** 50-100ms (WebSocket + rendering)
- **End-to-End (CARLA → Browser):** ~100-200ms

### Optimization Tips

1. **Reduce camera resolution:**
   ```yaml
   camera:
     width: 640
     height: 480
   ```

2. **Adjust WebSocket frame rate:**
   ```python
   # In viewer/run.py
   self.ws_frame_interval = 1.0 / 30.0  # 30 FPS (default)
   ```

3. **Lower JPEG quality:**
   ```python
   # In viewer/run.py
   cv2.imencode('.jpg', ..., [cv2.IMWRITE_JPEG_QUALITY, 70])
   ```

4. **Use OpenCV for detection:**
   ```bash
   lkas --method cv  # Faster than YOLO
   ```

## 🔧 Development

### Testing WebSocket Connection

```bash
# Terminal 1: Start viewer
viewer

# Terminal 2: Test WebSocket
python3 src/viewer/test_websocket.py

# Expected output:
# ✓ Connected to ws://localhost:8081
# ✓ Sent test message
# ✓ Received frame (binary)
# ✓ Received status (JSON)
```

### Debugging

**Enable verbose logging:**
```bash
lkas --method cv --broadcast --verbose
simulation --broadcast --verbose
viewer --verbose
```

**Check ZMQ ports:**
```bash
ss -tlnp | grep '555[7-9]\|556[0-3]'
```

**Monitor WebSocket:**
```bash
# Check WebSocket server
ss -tlnp | grep 8081

# Browser console (F12)
# Check connection status and frame reception
```

### Adding Custom Detection Method

```python
# In lkas/detection/<method>/detector.py
from lkas.detection.core.detector import LaneDetector

class MyDetector(LaneDetector):
    def detect(self, image):
        # Your detection logic
        left_lane = (x1, y1, x2, y2)
        right_lane = (x1, y1, x2, y2)
        return left_lane, right_lane

# Register in lkas/detection/core/detector.py
DETECTORS = {
    'cv': CVDetector,
    'yolo': YOLODetector,
    'my_method': MyDetector,
}

# Use it
lkas --method my_method --broadcast
```

## 🐛 Troubleshooting

### CARLA Connection Failed
```
Error: Could not connect to CARLA
```
**Fix:**
- Ensure CARLA is running: `./CarlaUE4.sh`
- Check host/port in config.yaml
- Verify firewall settings

### WebSocket Not Connecting
```
Connection: Disconnected (red)
```
**Fix:**
```bash
# Check viewer is running
ps aux | grep viewer

# Verify WebSocket server started
# Should see: ✓ WebSocket server started on port 8081

# Test connection
python3 src/viewer/test_websocket.py

# Check firewall
sudo ufw allow 8081
```

### High Latency (>1 second)
```
Latency: 5000+ms
```
**Fix:**
- Reduce JPEG quality (viewer/run.py, line ~290)
- Lower camera resolution (config.yaml)
- Reduce frame rate limit (viewer/run.py, line ~104)

### Parameters Not Updating
```
Slider moves but behavior doesn't change
```
**Fix:**
- Check LKAS is running with `--broadcast` flag
- Verify parameter port: `ss -tlnp | grep 5559`
- Check browser console for errors

## 📚 Documentation

Detailed documentation for each module:

| Module | Documentation |
|--------|---------------|
| **LKAS** | [src/lkas/README.md](src/lkas/README.md) |
| **Simulation** | [src/simulation/README.md](src/simulation/README.md) |
| **Viewer** | [src/viewer/README.md](src/viewer/README.md) |

## 🎓 Key Technologies

- **CARLA Simulator** - Realistic autonomous driving environment
- **OpenCV** - Computer vision for lane detection
- **ZMQ (ZeroMQ)** - High-performance distributed messaging
- **WebSocket** - Real-time bidirectional browser communication
- **Python asyncio** - Asynchronous WebSocket server
- **PID Controller** - Smooth vehicle steering control

## 🏆 Highlights

### WebSocket Improvements
- ✅ **Binary frames** - No base64 overhead (33% size reduction!)
- ✅ **Frame rate limiting** - 30 FPS max prevents flooding
- ✅ **Instant reconnection** - Auto-reconnect on disconnect
- ✅ **Low latency** - ~50-100ms browser latency
- ✅ **Efficient encoding** - JPEG quality balanced for speed

### Architecture Benefits
- ✅ **Modular design** - Clean separation of concerns
- ✅ **Process isolation** - Independent module lifecycles
- ✅ **ZMQ broker** - Centralized communication hub
- ✅ **Distributed ready** - Run modules on different machines
- ✅ **Live tuning** - Adjust parameters without restart

### Production Ready
- ✅ **Error handling** - Graceful degradation
- ✅ **Comprehensive logging** - Detailed diagnostics
- ✅ **Performance monitoring** - Real-time metrics
- ✅ **Fault tolerance** - Auto-reconnection and retry logic
- ✅ **Testing tools** - WebSocket test client included

## 📦 Package Information

- **Name:** `ads-skynet`
- **Version:** 0.1.0
- **Python:** 3.10+
- **License:** See LICENSE file

## 🚀 Getting Started

Ready to run? Follow the [Quick Start](#-quick-start) guide above!

**New to the project?** Start with these steps:
1. Install package: `pip install -e .`
2. Start CARLA: `./CarlaUE4.sh`
3. Start LKAS: `lkas --method cv --broadcast`
4. Start simulation: `simulation --broadcast`
5. Start viewer: `viewer`
6. Open browser: http://localhost:8080

**Questions?** Check the [Troubleshooting](#-troubleshooting) section or module-specific READMEs.

---

**Built with ❤️ for autonomous driving education and research**
