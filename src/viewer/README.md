# Viewer Module

**Remote web-based viewer for monitoring autonomous vehicles.**

## Overview

The viewer module runs on your **laptop** (not the vehicle!) and:
- ✅ Receives data from vehicle via ZMQ
- ✅ Draws overlays (lanes, HUD, metrics)
- ✅ Serves web interface for browser viewing
- ✅ Sends commands back to vehicle (respawn, pause)

## Why Separate Viewer?

**Problem with old approach:**
- ❌ Rendering runs on vehicle CPU
- ❌ Heavy overlay drawing impacts control loop
- ❌ Not suitable for resource-constrained devices

**Benefits of new approach:**
- ✅ **Vehicle stays lightweight** - No rendering on vehicle!
- ✅ **Rich visualizations** - Draw complex overlays on laptop
- ✅ **Remote monitoring** - Monitor from any machine on network
- ✅ **Multiple viewers** - Multiple laptops can connect

## Quick Start

### 1. Start vehicle/simulation (with broadcasting)

```bash
python simulation/run.py \
    --viewer none \
    --zmq-broadcast \
    --detector-url tcp://localhost:5556
```

### 2. Start viewer (on laptop)

```bash
# Using entry point (after pip install -e .)
zmq-viewer --vehicle tcp://vehicle-ip:5557 --port 8080

# Or directly
python viewer/zmq_web_viewer.py \
    --vehicle tcp://localhost:5557 \
    --port 8080
```

### 3. Open browser

```
http://localhost:8080
```

## Usage

### Command-line Options

```bash
zmq-viewer \
    --vehicle tcp://192.168.1.100:5557 \  # Vehicle data URL
    --actions tcp://192.168.1.100:5558 \  # Actions URL
    --port 8080                            # HTTP port
```

### Network Setup

**Same machine (development):**
```bash
zmq-viewer --vehicle tcp://localhost:5557
```

**Remote vehicle (production):**
```bash
# Find vehicle IP
# On vehicle: hostname -I

# Connect from laptop
zmq-viewer --vehicle tcp://192.168.1.100:5557
```

## Architecture

```
┌─ VEHICLE ─────────────────────┐
│                               │
│  Camera → Detection           │
│           ↓                   │
│  Control Decision             │
│           ↓                   │
│  ZMQ Broadcaster ─────────────┼──┐
│  :5557 (frames, detections)   │  │
│  :5558 (receive actions)      │  │
└───────────────────────────────┘  │
                                   │ Network
┌─ LAPTOP ──────────────────────┐  │
│                               │  │
│  ZMQ Subscriber ◄─────────────┼──┘
│           ↓                   │
│  Draw Overlays (HEAVY!)       │
│           ↓                   │
│  HTTP Server → Browser        │
│           ↑                   │
│  Actions (respawn, pause)     │
│                               │
└───────────────────────────────┘
```

## Features

### Data Reception
- **Frames**: Raw RGB images from vehicle camera
- **Detections**: Lane detection results (left/right lanes)
- **State**: Vehicle telemetry (steering, speed, throttle, brake)

### Visualization
- **Lane overlays**: Left/right lane lines with confidence
- **HUD**: Speed, steering, metrics
- **Performance**: Detection latency, FPS
- **Status**: Vehicle state, alerts

### Actions
- **Respawn**: Reset vehicle position
- **Pause**: Freeze simulation
- **Resume**: Continue simulation

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Latency | ~5-10ms | Network + rendering |
| FPS | 30 FPS | Matches vehicle broadcast rate |
| CPU (vehicle) | **0%** | No rendering on vehicle! |
| CPU (laptop) | ~20% | Drawing overlays |
| Network | ~500 KB/s | JPEG compressed frames |

## Troubleshooting

### Problem: No frames received

**Check:** Is broadcasting enabled on vehicle?
```bash
# Vehicle should show:
✓ ZMQ broadcaster started on tcp://*:5557
```

**Fix:** Add `--zmq-broadcast` flag to simulation

### Problem: Connection refused

**Check:** Firewall or wrong IP
```bash
# Test connection
telnet vehicle-ip 5557
```

**Fix:** Use correct IP address, check firewall

### Problem: Black screen in browser

**Check:** Is vehicle sending frames?
```bash
# Viewer terminal should show:
[Subscriber] Receiving 30.0 FPS | Frame 123
```

**Fix:** Check if simulation is running and camera is active

### Problem: Actions don't work

**Check:** Is action subscriber registered?
```bash
# Vehicle should show:
✓ ZMQ action subscriber registered
  Actions: respawn, pause, resume
```

**Fix:** Ensure `--zmq-broadcast` flag is used

## Development

### Adding New Overlays

Edit `_render_frame()` in `zmq_web_viewer.py`:

```python
def _render_frame(self):
    output = self.latest_frame.copy()

    # Your custom overlay here!
    cv2.putText(output, "Custom Text", (10, 150), ...)

    self.rendered_frame = output
```

### Adding New Actions

1. Register action in vehicle (`simulation/run.py`):
```python
def handle_custom_action():
    print("Custom action!")
    return True

action_subscriber.register_action('custom', handle_custom_action)
```

2. Add button in web viewer HTML:
```html
<button onclick="sendAction('custom')">Custom Action</button>
```

## Files

- `zmq_web_viewer.py` - Main viewer implementation
- `__init__.py` - Package exports
- `README.md` - This file

## See Also

- [Architecture Guide](../.docs/NEW_ARCHITECTURE.md)
- [Quick Start Guide](../.docs/QUICKSTART_NEW_ARCHITECTURE.md)
- [ZMQ Broadcasting Module](../simulation/integration/zmq_broadcast.py)

## License

See main project LICENSE file.
