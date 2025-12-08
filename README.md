# LKAS (Lane Keeping Assist System) Module

**Real-time computer vision pipeline for lane detection and vehicle steering control.**

## Overview

The LKAS module processes camera frames from the simulation/vehicle, detects lane markings, and computes steering commands to keep the vehicle centered in its lane. It communicates with the simulation via shared memory or ZMQ, and broadcasts data to the viewer for monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LKAS Module                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  Shared Memory  ┌─────────────┐  Shared Memory│
│  │  Detection  │ ───────────────►│  Decision   │───────────────►│
│  │   Server    │   (~0.01ms)     │   Server    │   (~0.01ms)   │
│  │   (CV/DL)   │                 │  (PD/PID)   │               │
│  └─────┬───────┘                 └──────┬──────┘               │
│        │                                │                       │
│        │ ZMQ Parameters                 │ ZMQ Parameters        │
│        │                                │                       │
│        └────────────┬───────────────────┘                       │
│                     │                                           │
│              ┌──────▼────────┐                                  │
│              │  LKAS Broker  │                                  │
│              │ (Integration) │                                  │
│              └──────┬────────┘                                  │
│                     │                                           │
└─────────────────────┼───────────────────────────────────────────┘
                      │
       ┌──────────────┼──────────────┬───────────────────┐
       │              │              │                   │
   Simulation      Viewer     Parameter Updates     Action Requests
   (Vehicle)    (Monitoring)   (Live Tuning)        (Pause/Resume)
```

## Features

### Detection Module ([src/detection/](src/detection/))
- **Multi-method support:**
  - `cv`: Classical OpenCV (Canny + Hough Transform)
  - `dl`: Deep learning-based detection (PyTorch)
  - Pluggable architecture for easy extension
- **Lane processing:**
  - Edge detection with Canny algorithm
  - Line detection via Hough Transform
  - Lane clustering and extrapolation
  - Temporal smoothing for stability
- **Real-time optimization:**
  - ROI (Region of Interest) masking
  - Configurable preprocessing pipeline
  - Adaptive thresholding
- **Performance:** 5-15ms (CV), 15-30ms (DL)

See [Detection Module README](src/detection/README.md) for details.

### Decision Module ([src/decision/](src/decision/))
- **Multi-controller support:**
  - `pd`: Proportional-Derivative control (fast, stateless)
  - `pid`: Proportional-Integral-Derivative control (zero steady-state error)
  - `mpc`: Model Predictive Control (optimal, future)
- **Lane analysis:**
  - Lateral offset calculation (pixels, meters, normalized)
  - Heading angle estimation
  - Lane departure detection (CENTERED, DRIFT, DEPARTURE)
- **Adaptive throttle:**
  - Automatic speed reduction in turns
  - Configurable throttle policy
- **Safety features:**
  - Emergency braking when no lanes detected
  - Configurable control limits
- **Performance:** <2ms total decision latency

See [Decision Module README](src/decision/README.md) for details.

### Integration Module ([src/integration/](src/integration/))
- **Shared Memory IPC:**
  - Ultra-low latency (~0.01ms) for local processes
  - Zero-copy image and detection transfer
  - ImageChannel, DetectionChannel, ControlChannel
- **ZMQ Communication:**
  - Network-capable broadcasting to viewers
  - Real-time parameter updates (port 5559 → 5560)
  - Action requests (pause, resume, respawn)
  - Vehicle telemetry streaming
- **LKASBroker:**
  - Central coordination hub
  - Routes parameters to detection/decision servers
  - Broadcasts data to viewers (port 5557)
  - Handles viewer actions (port 5558)

See [Integration Module README](src/integration/README.md) for details.

### Utils Module ([src/utils/](src/utils/))
- **Terminal display:**
  - Persistent footer with live stats (FPS, frame count, latency)
  - Ordered logging from concurrent processes
  - ANSI-based structured output
- **Helper utilities:**
  - FPS formatting
  - Progress bars

See [Utils Module README](src/utils/README.md) for details.

## Quick Start

### Basic Usage

```bash
# Start LKAS with OpenCV method
lkas --method cv --broadcast

# Start LKAS with YOLO method
lkas --method yolo --broadcast

# Start with custom config
lkas --config path/to/config.yaml --method cv --broadcast
```

### Full System Setup

```bash
# Terminal 1: Start CARLA simulator
./CarlaUE4.sh

# Terminal 2: Start LKAS
lkas --method cv --broadcast

# Terminal 3: Start simulation
simulation --broadcast

# Terminal 4: Start viewer (optional)
viewer
```

## Configuration

Configuration is loaded from `config.yaml` in the project root.

### Detection Parameters

```yaml
detection:
  method: cv  # cv, yolo, yolo-seg
  cv:
    canny_low: 50
    canny_high: 150
    hough_threshold: 50
    hough_min_line_len: 40
    smoothing_factor: 0.7
  roi:
    top_trim: 0.55
    bottom_trim: 0.1
    side_trim: 0.1
```

### Decision Parameters

```yaml
decision:
  method: pd        # pd, pid, or mpc
  kp: 0.5           # Proportional gain
  ki: 0.01          # Integral gain (PID only)
  kd: 0.1           # Derivative gain

  # Adaptive throttle policy
  throttle_base: 0.15
  throttle_min: 0.05
  steer_threshold: 0.15
  steer_max: 0.70

  # Lane analyzer
  drift_threshold: 0.15
  departure_threshold: 0.35
  lane_width_meters: 3.7
```

### Integration Configuration

```yaml
# Shared Memory (ultra-low latency IPC)
shared_memory:
  image_channel: "camera_feed"
  detection_channel: "lane_detection"
  control_channel: "vehicle_control"

# ZMQ (network-capable communication)
zmq:
  broker:
    viewer_data_port: 5557          # Broadcast to viewers
    viewer_action_port: 5558        # Receive actions from viewers
    parameter_update_port: 5559     # Receive parameter updates
    server_parameter_port: 5560     # Forward parameters to servers
    simulation_action_port: 5561    # Forward actions to simulation
    simulation_status_port: 5562    # Receive vehicle status
```

## Module Structure

```
lkas/
├── src/
│   ├── detection/               # Lane detection module
│   │   ├── core/
│   │   │   ├── interfaces.py   # LaneDetector ABC
│   │   │   ├── factory.py      # DetectorFactory
│   │   │   └── __init__.py
│   │   ├── method/
│   │   │   ├── computer_vision/  # CV-based detection
│   │   │   └── deep_learning/    # DL-based detection
│   │   ├── detector.py         # LaneDetection wrapper
│   │   ├── client.py           # DetectionClient
│   │   ├── server.py           # DetectionServer
│   │   ├── run.py              # CLI entrypoint
│   │   └── README.md           # Detection docs
│   │
│   ├── decision/                # Steering control module
│   │   ├── core/
│   │   │   ├── interfaces.py   # SteeringController ABC
│   │   │   ├── factory.py      # ControllerFactory
│   │   │   └── __init__.py
│   │   ├── method/
│   │   │   ├── pd_controller.py   # PD control
│   │   │   ├── pid_controller.py  # PID control
│   │   │   └── mpc_controller.py  # MPC control
│   │   ├── controller.py       # DecisionController
│   │   ├── lane_analyzer.py    # LaneAnalyzer
│   │   ├── client.py           # DecisionClient
│   │   ├── server.py           # DecisionServer
│   │   ├── run.py              # CLI entrypoint
│   │   └── README.md           # Decision docs
│   │
│   ├── integration/             # Communication layer
│   │   ├── shared_memory/      # Ultra-low latency IPC
│   │   │   ├── channels.py     # Image/Detection/Control channels
│   │   │   ├── messages.py     # Message definitions
│   │   │   └── __init__.py
│   │   ├── zmq/                # Network messaging
│   │   │   ├── broker.py       # LKASBroker
│   │   │   ├── broadcaster.py  # Data broadcasting
│   │   │   ├── messages.py     # ZMQ message protocols
│   │   │   ├── README.md       # ZMQ docs
│   │   │   └── __init__.py
│   │   └── README.md           # Integration docs
│   │
│   ├── utils/                   # Utilities
│   │   ├── terminal.py         # Terminal display
│   │   ├── README.md           # Utils docs
│   │   └── __init__.py
│   │
│   ├── process.py              # Process management
│   ├── server.py               # LKASServer (orchestrator)
│   ├── client.py               # LKASClient
│   ├── run.py                  # Main CLI entrypoint
│   └── __init__.py
│
├── ARCHITECTURE.md              # Architecture documentation
└── README.md                    # This file
```

## Performance

### Typical Performance Metrics

- **Detection latency:** 5-15ms (CV), 15-30ms (DL)
- **Decision latency:** <2ms
- **Shared memory transfer:** ~0.01ms per channel
- **ZMQ broadcast:** 0.5-2ms (localhost), 5-50ms (network)
- **End-to-end latency:** 10-50ms
- **Target FPS:** 30+ FPS

### Latency Breakdown

```
Camera → Detection:  ~0.01ms  (Shared Memory)
Detection Process:   5-30ms   (CV/DL algorithm)
Detection → Decision: ~0.01ms (Shared Memory)
Decision Process:    <2ms     (PD/PID control)
Decision → Control:  ~0.01ms  (Shared Memory)
──────────────────────────────
Total:               ~10-50ms
```

### Performance Tips

1. **Use OpenCV for low latency:**
   ```bash
   lkas --method cv
   ```

2. **Optimize ROI settings:**
   ```yaml
   roi:
     top_trim: 0.6  # Ignore more of top
     side_trim: 0.2  # Focus on center
   ```

3. **Tune smoothing factor:**
   ```yaml
   detection:
     cv:
       smoothing_factor: 0.8  # Higher = smoother but more lag
   ```

## Live Parameter Tuning

The viewer provides real-time parameter adjustment via WebSocket:

- **Detection parameters:** Canny thresholds, Hough parameters, smoothing
- **Decision parameters:** PID gains, throttle, steering threshold
- **Changes apply immediately** without restarting the system

## Communication Protocols

### Frame Input (from Simulation)
```python
# Simulation → LKAS (port 5560)
{
    "image": <numpy_array>,  # Camera frame
    "timestamp": 1234567890.123
}
```

### Steering Output (to Simulation)
```python
# LKAS → Simulation (port 5563)
{
    "steering": 0.25,        # -1.0 to 1.0
    "throttle": 0.14,        # 0.0 to 1.0
    "brake": 0.0             # 0.0 to 1.0
}
```

### Data Broadcast (to Viewer)
```python
# LKAS → Viewer (port 5557)
# Frame
{"type": "frame", "image": <bytes>}

# Detection
{"type": "detection", "left_lane": {...}, "right_lane": {...}}

# Vehicle state
{"type": "state", "speed": 8.0, "steering": 0.1, ...}
```

## Development

### Adding a New Detection Method

1. Create detector class in `detection/method/<method>/`:
```python
from lkas.detection.core.base import BaseDetector

class MyDetector(BaseDetector):
    def detect(self, image):
        # Implement detection logic
        return left_lane, right_lane
```

2. Register in `detection/core/factory.py`:
```python
# Add to DetectorFactory.create() method
if method == 'my_method':
    from lkas.detection.method.my_method import MyDetector
    return MyDetector(self.config)
```

3. Use it:
```bash
lkas --method my_method --broadcast
```

### Debugging

Enable verbose logging:
```bash
lkas --method cv --broadcast --verbose
```

Check ZMQ ports:
```bash
ss -tlnp | grep '555[7-9]\|556[0-3]'
```

## Troubleshooting

### LKAS not receiving frames
- Check simulation is running with `--broadcast`
- Verify ZMQ ports are not in use
- Check `config.yaml` port configuration

### Poor detection quality
- Adjust Canny thresholds (canny_low, canny_high)
- Tune Hough parameters (hough_threshold, hough_min_line_len)
- Modify ROI settings to focus on lane area

### Steering too sensitive/sluggish
- Adjust PID gains (Kp for responsiveness, Kd for stability)
- Modify steer_threshold for deadzone
- Check smoothing_factor (higher = smoother)

## References

### Module Documentation
- [Detection Module](src/detection/README.md) - Lane detection algorithms and usage
- [Decision Module](src/decision/README.md) - Steering control and lane analysis
- [Integration Module](src/integration/README.md) - Communication infrastructure
  - [ZMQ Integration](src/integration/zmq/README.md) - ZMQ broker details
- [Utils Module](src/utils/README.md) - Terminal display and utilities

### Architecture
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and data flow

### External
- [CARLA Simulator](https://carla.org/) - Autonomous driving simulator
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [ZeroMQ](https://zeromq.org/) - High-performance messaging library
