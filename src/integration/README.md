# Integration Module

**High-performance inter-process communication for LKAS.**

## Overview

The Integration module provides communication infrastructure for the LKAS system, supporting both ultra-low latency shared memory IPC and network-based ZMQ messaging. It enables distributed processing across multiple processes and machines while maintaining real-time performance.

### Key Features

- **Dual Communication Modes**:
  - **Shared Memory**: Ultra-low latency (<0.01ms) for local processes
  - **ZMQ**: Network-capable messaging for remote viewers and control
- **Zero-Copy Data Transfer**: Efficient image and detection data sharing
- **Type-Safe Messages**: Structured dataclass-based protocols
- **Real-time Broadcasting**: Live video, detection, and telemetry streaming
- **Parameter Updates**: Dynamic configuration without restart
- **Action Handling**: Remote control commands (pause, resume, respawn)

## Architecture

```
integration/
├── shared_memory/              # Ultra-low latency local IPC
│   ├── channels.py            # Shared memory channels
│   ├── messages.py            # Message type definitions
│   └── __init__.py
│
└── zmq/                        # Network-capable messaging
    ├── broker.py              # LKASBroker (coordination hub)
    ├── broadcaster.py         # Data broadcasting to viewers
    ├── messages.py            # ZMQ message protocols
    ├── README.md              # Detailed ZMQ documentation
    └── __init__.py
```

## Communication Modes

### 1. Shared Memory (Local IPC)

**Ultra-low latency communication** between LKAS processes on the same machine.

#### Architecture

```
┌──────────────┐        Shared Memory        ┌─────────────────┐
│ Simulation   │ ───────────────────────────►│ Detection       │
│              │  Image Channel (~0.01ms)    │ Server          │
└──────────────┘                             └────────┬────────┘
                                                      │
                                                      │ Detection Channel
                                                      │ (~0.01ms)
                                                      ▼
                                             ┌─────────────────┐
                                             │ Decision        │
                                             │ Server          │
                                             └────────┬────────┘
                                                      │
                                                      │ Control Channel
                                                      │ (~0.01ms)
                                                      ▼
┌──────────────┐                             ┌─────────────────┐
│ Simulation   │◄────────────────────────────│ LKAS Broker     │
└──────────────┘  Control Commands           └─────────────────┘
```

#### Channels

**ImageChannel** - Camera frames
```python
from lkas.integration.shared_memory import ImageChannel

# Writer (Simulation)
channel = ImageChannel(name="camera_feed", create=True,
                      width=800, height=600, channels=3)
channel.write(image_array, frame_id=1)

# Reader (Detection)
channel = ImageChannel(name="camera_feed", create=False)
msg = channel.read()
if msg:
    print(f"Frame {msg.frame_id}: {msg.image.shape}")
```

**DetectionChannel** - Lane detection results
```python
from lkas.integration.shared_memory import DetectionChannel
from lkas.integration.shared_memory.messages import LaneMessage

# Writer (Detection)
channel = DetectionChannel(name="lane_detection", create=True)
left_lane = LaneMessage(x1=200, y1=600, x2=350, y2=360, confidence=0.95)
right_lane = LaneMessage(x1=600, y1=600, x2=450, y2=360, confidence=0.92)
channel.write(left_lane, right_lane, frame_id=1, processing_time_ms=8.5)

# Reader (Decision)
channel = DetectionChannel(name="lane_detection", create=False)
msg = channel.read()
if msg and msg.left_lane:
    print(f"Left lane: ({msg.left_lane.x1}, {msg.left_lane.y1})")
```

**ControlChannel** - Steering commands
```python
from lkas.integration.shared_memory import ControlChannel
from lkas.integration.shared_memory.messages import ControlMode

# Writer (Decision)
channel = ControlChannel(name="vehicle_control", create=True)
channel.write(
    steering=0.125,
    throttle=0.15,
    brake=0.0,
    mode=ControlMode.LANE_KEEPING,
    lateral_offset=-0.05,
    heading_angle=2.3
)

# Reader (Simulation)
channel = ControlChannel(name="vehicle_control", create=False)
msg = channel.read()
if msg:
    vehicle.apply_control(msg.steering, msg.throttle, msg.brake)
```

#### Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Image write (800×600 RGB) | ~0.01ms | 100,000+ FPS theoretical |
| Detection write | ~0.005ms | 200,000+ FPS theoretical |
| Control write | ~0.003ms | 300,000+ FPS theoretical |

**Real-world end-to-end:** ~10-50ms (dominated by detection/decision computation)

### 2. ZMQ (Network Messaging)

**Network-capable communication** for broadcasting to remote viewers and parameter updates.

#### Architecture

```
                        ┌─────────────────────────────────┐
                        │       LKASBroker                │
                        │   (LKAS Main Process)           │
                        │                                 │
                        │  • Routes parameters            │
                        │  • Routes actions               │
                        │  • Broadcasts data              │
                        └──────┬────────────┬─────────────┘
                               │            │
        ┌──────────────────────┘            └────────────────────┐
        │                                                         │
        ▼                                                         ▼
┌──────────────────┐                                   ┌──────────────────┐
│ Detection Server │                                   │   Viewer(s)      │
│                  │                                   │                  │
│ ParameterClient  │                                   │ • Video stream   │
│ receives updates │                                   │ • Detection viz  │
└──────────────────┘                                   │ • Telemetry      │
        │                                              │ • Controls       │
        ▼                                              └──────────────────┘
┌──────────────────┐
│ Decision Server  │
│                  │
│ ParameterClient  │
│ receives updates │
└──────────────────┘
```

#### Port Allocation

| Port | Direction | Purpose |
|------|-----------|---------|
| 5557 | LKAS → Viewers | Broadcast frames/detection/state |
| 5558 | Viewer → LKAS | Action requests (pause, resume, etc.) |
| 5559 | Viewer → LKAS | Parameter updates |
| 5560 | LKAS → Servers | Forward parameters to servers |
| 5561 | LKAS → Simulation | Forward actions to simulation |
| 5562 | Simulation → LKAS | Vehicle status from simulation |

#### LKASBroker

Central coordination hub for ZMQ communication.

```python
from lkas.integration.zmq import LKASBroker

# Initialize broker
broker = LKASBroker()

# Register action handlers (optional, actions auto-forwarded to simulation)
broker.register_action('pause', lambda: print("Paused"))
broker.register_action('resume', lambda: print("Resumed"))

# Main loop
while running:
    # Poll for messages (non-blocking)
    broker.poll()

    # Broadcast data from shared memory
    image_msg = image_channel.read()
    if image_msg:
        broker.broadcast_frame(image_msg.image, image_msg.frame_id)

    detection_msg = detection_channel.read()
    if detection_msg:
        detection_data = {
            'left_lane': {...} if detection_msg.left_lane else None,
            'right_lane': {...} if detection_msg.right_lane else None,
            'processing_time_ms': detection_msg.processing_time_ms,
            'frame_id': detection_msg.frame_id,
        }
        broker.broadcast_detection(detection_data, detection_msg.frame_id)

# Cleanup
broker.close()
```

#### ParameterClient (Server Side)

Receive parameter updates in detection/decision servers.

```python
from lkas.integration.zmq import ParameterClient

class DetectionServer:
    def __init__(self):
        # Create parameter client for this category
        self.param_client = ParameterClient(category='detection')
        self.param_client.register_callback(self._on_parameter_update)

    def _on_parameter_update(self, param_name: str, value: float):
        # Update detector parameter
        if hasattr(self.detector, 'update_parameter'):
            self.detector.update_parameter(param_name, value)
            print(f"Updated {param_name} = {value}")

    def run(self):
        while self.running:
            # Poll for updates (non-blocking)
            self.param_client.poll()

            # Do detection work
            # ...

    def stop(self):
        self.param_client.close()
```

## Message Protocols

### Shared Memory Messages

#### ImageMessage
```python
@dataclass
class ImageMessage:
    image: np.ndarray       # RGB image (H, W, 3)
    frame_id: int           # Sequential frame number
    timestamp: float        # Capture time (seconds)
```

#### DetectionMessage
```python
@dataclass
class DetectionMessage:
    left_lane: LaneMessage | None
    right_lane: LaneMessage | None
    frame_id: int
    timestamp: float
    processing_time_ms: float

@dataclass
class LaneMessage:
    x1: int                 # Start point (bottom)
    y1: int
    x2: int                 # End point (top)
    y2: int
    confidence: float       # Detection confidence [0, 1]
```

#### ControlMessage
```python
@dataclass
class ControlMessage:
    steering: float                    # [-1, 1]
    throttle: float                    # [0, 1]
    brake: float                       # [0, 1]
    mode: ControlMode                  # LANE_KEEPING, MANUAL, etc.
    lateral_offset: float | None       # Normalized offset from center
    lateral_offset_meters: float | None
    heading_angle: float | None        # Degrees
    lane_width_pixels: float | None
    departure_status: str | None       # CENTERED, DRIFT, DEPARTURE
```

### ZMQ Messages

#### Parameter Update (Viewer → LKAS → Servers)
```json
{
    "category": "detection",
    "name": "canny_low",
    "value": 60.0
}
```

#### Action Request (Viewer → LKAS)
```json
{
    "action": "pause"
}
```

#### Frame Broadcast (LKAS → Viewers)
```json
{
    "type": "frame",
    "frame_id": 1234,
    "image": "<base64_or_bytes>",
    "timestamp": 1234567890.123
}
```

#### Detection Broadcast (LKAS → Viewers)
```json
{
    "type": "detection",
    "frame_id": 1234,
    "left_lane": {"x1": 200, "y1": 600, "x2": 350, "y2": 360, "confidence": 0.95},
    "right_lane": {"x1": 600, "y1": 600, "x2": 450, "y2": 360, "confidence": 0.92},
    "processing_time_ms": 8.5
}
```

#### Vehicle State Broadcast (LKAS → Viewers)
```json
{
    "type": "state",
    "speed": 8.5,
    "steering": 0.125,
    "throttle": 0.15,
    "lateral_offset": -0.05,
    "heading_angle": 2.3,
    "departure_status": "centered"
}
```

## Usage Examples

### Complete LKAS System

```python
from lkas.integration.shared_memory import ImageChannel, DetectionChannel, ControlChannel
from lkas.integration.zmq import LKASBroker

# Initialize shared memory channels
image_channel = ImageChannel("camera_feed", create=False)
detection_channel = DetectionChannel("lane_detection", create=False)
control_channel = ControlChannel("vehicle_control", create=False)

# Initialize ZMQ broker (optional, for broadcasting)
broker = LKASBroker() if args.broadcast else None

# Main loop
while running:
    # Poll ZMQ for parameter updates and actions
    if broker:
        broker.poll()

    # Read and broadcast camera frames
    image_msg = image_channel.read()
    if image_msg and broker:
        broker.broadcast_frame(image_msg.image, image_msg.frame_id)

    # Read and broadcast detection results
    detection_msg = detection_channel.read()
    if detection_msg and broker:
        detection_data = {
            'left_lane': asdict(detection_msg.left_lane) if detection_msg.left_lane else None,
            'right_lane': asdict(detection_msg.right_lane) if detection_msg.right_lane else None,
            'processing_time_ms': detection_msg.processing_time_ms,
            'frame_id': detection_msg.frame_id,
        }
        broker.broadcast_detection(detection_data, detection_msg.frame_id)

    # Read control commands (for monitoring/display)
    control_msg = control_channel.read()

# Cleanup
if broker:
    broker.close()
```

### Detection Server with Both Communication Modes

```python
from lkas.integration.shared_memory import ImageChannel, DetectionChannel
from lkas.integration.zmq import ParameterClient

class DetectionServer:
    def __init__(self):
        # Shared memory I/O
        self.image_channel = ImageChannel("camera_feed", create=False)
        self.detection_channel = DetectionChannel("lane_detection", create=True)

        # ZMQ parameter updates
        self.param_client = ParameterClient(category='detection')
        self.param_client.register_callback(self._on_parameter_update)

        # Detector
        self.detector = create_detector()

    def _on_parameter_update(self, param_name: str, value: float):
        self.detector.update_parameter(param_name, value)

    def run(self):
        while self.running:
            # Check for parameter updates
            self.param_client.poll()

            # Read image from shared memory
            image_msg = self.image_channel.read()
            if not image_msg:
                continue

            # Detect lanes
            result = self.detector.detect(image_msg.image)

            # Write to shared memory
            left_lane = LaneMessage(...) if result.left_lane else None
            right_lane = LaneMessage(...) if result.right_lane else None
            self.detection_channel.write(
                left_lane, right_lane,
                frame_id=image_msg.frame_id,
                processing_time_ms=result.processing_time_ms
            )

    def stop(self):
        self.param_client.close()
        self.image_channel.unlink()
        self.detection_channel.unlink()
```

## Running LKAS with Broadcasting

```bash
# 1. Start CARLA
./CarlaUE4.sh

# 2. Start LKAS with broadcasting (enables ZMQ broker)
lkas --method cv --broadcast

# 3. Start simulation with broadcasting (sends vehicle status to LKAS)
simulation --broadcast

# 4. Start viewer (connects to LKAS broker)
viewer

# Open browser: http://localhost:8080
```

**Data Flow:**
- Simulation → LKAS (shared memory) → Viewer (ZMQ): Camera frames
- Detection → LKAS (shared memory) → Viewer (ZMQ): Lane detections
- Decision → LKAS (shared memory) → Viewer (ZMQ): Control commands
- Simulation → LKAS (ZMQ port 5562) → Viewer (ZMQ): Vehicle telemetry
- Viewer → LKAS (ZMQ port 5559) → Servers (ZMQ port 5560): Parameter updates
- Viewer → LKAS (ZMQ port 5558) → Simulation (ZMQ port 5561): Actions

## Design Principles

1. **Zero-Copy Where Possible**: Shared memory avoids data serialization
2. **Non-Blocking Operations**: All poll() calls use timeouts
3. **Clean Separation**: Shared memory for latency-critical paths, ZMQ for monitoring
4. **Type Safety**: Dataclass-based messages with validation
5. **Encapsulation**: Clean APIs hide implementation details
6. **Graceful Degradation**: System works without ZMQ broadcasting

## Performance Comparison

| Communication | Latency | Bandwidth | Use Case |
|---------------|---------|-----------|----------|
| Shared Memory | ~0.01ms | 10+ GB/s | Local processes, latency-critical |
| ZMQ (TCP localhost) | ~0.5-2ms | 1-5 GB/s | Local monitoring, parameter updates |
| ZMQ (TCP network) | ~5-50ms | 100 MB/s - 1 GB/s | Remote viewers, telemetry |

## Troubleshooting

### Shared Memory Errors

**"No such file or directory"**
- Channel not created yet - ensure writer creates channel first
- Channel was unlinked - restart the creating process

**"File exists"**
- Previous channel not cleaned up - run cleanup script or reboot

**Stale data**
- Writer not running - check process status
- Check `ready` flag - may need to clear

### ZMQ Errors

**"Address already in use"**
- Port conflict - check with `ss -tlnp | grep 555[7-9]`
- Kill existing process or change port

**No data received**
- Check firewall settings
- Verify ports match in config
- Ensure broker is running

**High latency**
- ZMQ messages buffered - reduce HWM (High Water Mark)
- Network congestion - check bandwidth usage
- Use shared memory for latency-critical paths

## Related Documentation

- [ZMQ Integration Details](zmq/README.md) - In-depth ZMQ architecture
- [Shared Memory Messages](shared_memory/messages.py) - Message definitions
- [Decision Module](../decision/README.md) - Control command generation
- [Detection Module](../detection/README.md) - Lane detection output

## Version

Current version: 0.1.0 (see [\_\_init\_\_.py](__init__.py))
