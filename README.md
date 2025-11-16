# LKAS (Lane Keeping Assist System) Module

**Real-time computer vision pipeline for lane detection and vehicle steering control.**

## Overview

The LKAS module processes camera frames from the simulation/vehicle, detects lane markings, and computes steering commands to keep the vehicle centered in its lane. It communicates with the simulation via shared memory or ZMQ, and broadcasts data to the viewer for monitoring.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     LKAS Module                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │  Detection  │ -> │  Decision   │ -> │ Actuator   │  │
│  │  (Vision)   │    │   (PID)     │    │ (Steering) │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
│         │                   │                   │        │
│         └───────────────────┴───────────────────┘        │
│                         │                                │
│                    ZMQ Broker                            │
│                  (Coordination)                          │
│                         │                                │
└─────────────────────────┼────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    Simulation        Viewer       Parameter Updates
    (Vehicle)      (Monitoring)    (Live Tuning)
```

## Features

### Detection (Computer Vision)
- **Multi-method support:**
  - `cv`: Classical OpenCV (Canny + Hough Transform)
  - `yolo`: Deep learning-based detection (YOLOv8)
  - `yolo-seg`: Semantic segmentation-based detection
- **Lane processing:**
  - Edge detection with Canny algorithm
  - Line detection via Hough Transform
  - Lane clustering and extrapolation
  - Temporal smoothing for stability
- **Real-time optimization:**
  - ROI (Region of Interest) masking
  - Configurable preprocessing pipeline
  - Adaptive thresholding

### Decision (Control)
- **PID Controller:**
  - Proportional (Kp) + Derivative (Kd) control
  - Lateral offset and heading angle calculation
  - Smooth steering output with thresholding
- **Safety features:**
  - Lane departure detection
  - Emergency centering logic
  - Configurable control limits

### ZMQ Broker (Integration Hub)
- **Bidirectional communication:**
  - Receives frames from simulation (port 5560)
  - Sends steering commands to simulation (port 5563)
- **Data broadcasting:**
  - Publishes frames, detections, and vehicle state to viewer (port 5557)
  - Receives actions from viewer (port 5558)
  - Handles parameter updates (port 5559)
- **Process coordination:**
  - Manages LKAS pipeline lifecycle
  - Handles pause/resume/respawn actions
  - Synchronizes data between modules

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
  kp: 0.5           # Proportional gain
  kd: 0.1           # Derivative gain
  throttle_base: 0.14
  throttle_min: 0.05
  steer_threshold: 0.15
```

### ZMQ Configuration

```yaml
zmq:
  broker:
    # Simulation connection
    detection_input_port: 5560   # Receive frames from sim
    decision_output_port: 5563   # Send steering to sim

    # Viewer broadcasting
    viewer_data_port: 5557       # Broadcast data to viewer
    viewer_action_port: 5558     # Receive actions from viewer
    parameter_update_port: 5559  # Receive parameter updates
```

## Module Structure

```
lkas/
├── run.py                        # Main entry point
│
├── detection/                    # Lane detection
│   ├── core/
│   │   ├── detector.py          # Detection interface
│   │   ├── config.py            # Configuration management
│   │   └── models.py            # Data models
│   ├── cv/
│   │   └── detector.py          # OpenCV-based detector
│   ├── yolo/
│   │   ├── detector.py          # YOLO-based detector
│   │   └── model_loader.py      # Model loading utilities
│   └── preprocessing/
│       └── roi.py               # ROI masking
│
├── decision/                     # Steering control
│   ├── controller.py            # PID controller
│   └── metrics.py               # Control metrics
│
├── integration/                  # Communication layer
│   ├── shared_memory/           # Shared memory IPC (legacy)
│   │   └── channels.py
│   └── zmq/                     # ZMQ-based communication
│       ├── broker.py            # Main ZMQ broker
│       ├── broadcaster.py       # Data broadcasting
│       └── messages.py          # Message protocols
│
└── orchestrator.py              # LKAS pipeline coordinator
```

## Performance

### Typical Performance Metrics

- **Detection latency:** 5-15ms (OpenCV), 20-40ms (YOLO)
- **Decision latency:** <1ms
- **End-to-end latency:** 10-50ms
- **Target FPS:** 30+ FPS

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

1. Create detector class in `detection/<method>/`:
```python
from detection.core.detector import LaneDetector

class MyDetector(LaneDetector):
    def detect(self, image):
        # Implement detection logic
        return left_lane, right_lane
```

2. Register in `detection/core/detector.py`:
```python
DETECTORS = {
    'cv': CVDetector,
    'yolo': YOLODetector,
    'my_method': MyDetector,
}
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

- [Shared Memory Communication](integration/shared_memory/README.md)
- [ZMQ Integration](integration/zmq/README.md)
- [Detection Methods Comparison](detection/README.md)
