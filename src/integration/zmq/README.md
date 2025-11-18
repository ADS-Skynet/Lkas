# LKAS ZMQ Integration

Clean, encapsulated ZMQ communication layer for LKAS.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LKAS Main Process                              │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                        LKASBroker                               │     │
│  │                                                                  │     │
│  │  ROUTING:                                                        │     │
│  │  • Receives parameters from viewer → forwards to servers        │     │
│  │  • Receives actions from viewer → forwards to simulation        │     │
│  │  • Receives vehicle status from simulation                      │     │
│  │                                                                  │     │
│  │  BROADCASTING (to viewers):                                      │     │
│  │  • Frames (from shared memory)                                  │     │
│  │  • Detection results (from shared memory)                       │     │
│  │  • Vehicle status (from simulation via ZMQ)                     │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│    ↓ params (ZMQ)    ↓ params (ZMQ)     ↑ frames/detection (SHM)       │
│  ┌──────────────┐  ┌──────────────┐                                     │
│  │  Detection   │  │   Decision   │                                     │
│  │  SubProcess  │  │  SubProcess  │                                     │
│  └──────────────┘  └──────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────┘
         ↑                    ↑                  │
         │                    │                  │ actions (ZMQ)
         │  params (ZMQ)      │                  ↓
         │                    │         ┌─────────────────┐
         └────────────────────┘         │   Simulation    │
                                        │                 │
                                        │  Sends vehicle  │
                                        │  status via ZMQ │
                                        └─────────────────┘
                                               │
                                               │ status (ZMQ)
                                               ↓
                                          LKAS Broker
```

## Communication Flow

### 1. Parameter Updates: Viewer → Servers

```
Viewer                  LKAS Broker                Detection/Decision
  │                          │                            │
  │  parameter update        │                            │
  ├─────────────────────────►│                            │
  │  tcp://*:5559            │                            │
  │                          │  forward to category       │
  │                          ├───────────────────────────►│
  │                          │  tcp://*:5560              │
  │                          │                            │
  │                          │                     ParameterClient
  │                          │                     .poll() receives
```

### 2. Action Requests: Viewer → LKAS

```
Viewer                  LKAS Broker
  │                          │
  │  action request          │
  ├─────────────────────────►│
  │  tcp://*:5558            │
  │                          │
  │                    route to handler
  │                    (pause/resume/etc)
```

### 3. Broadcasting: LKAS Broker → Viewers

```
LKAS Broker                    Viewer
  │                              │
  │  frame (from SHM)            │
  ├─────────────────────────────►│
  │  tcp://*:5557                │
  │                              │
  │  detection (from SHM)        │
  ├─────────────────────────────►│
  │                              │
  │  vehicle status (from sim)   │
  ├─────────────────────────────►│
  │                              │
```

### 4. Vehicle Status: Simulation → LKAS Broker

```
Simulation              LKAS Broker
  │                          │
  │  vehicle status          │
  ├─────────────────────────►│
  │  tcp://localhost:5562    │
  │                          │
```

## Port Allocation

| Port | Direction          | Purpose                              |
|------|--------------------|--------------------------------------|
| 5557 | LKAS → Viewers     | Broadcast frames/detection/state     |
| 5558 | Viewer → LKAS      | Action requests from viewer          |
| 5559 | Viewer → LKAS      | Parameter updates from viewer        |
| 5560 | LKAS → Servers     | Forward parameters to servers        |
| 5561 | LKAS → Simulation  | Forward actions to simulation        |
| 5562 | Simulation → LKAS  | Vehicle status from simulation       |

## Usage

### LKAS Main Process (Broker Side)

```python
from lkas.integration.zmq import LKASBroker

# Initialize broker with --broadcast flag
if args.broadcast:
    broker = LKASBroker()

    # Optional: Register local action handlers
    # (Actions are also forwarded to simulation automatically)
    broker.register_action('pause', on_pause)
    broker.register_action('resume', on_resume)
else:
    broker = None

# Main loop
while running:
    # Poll for incoming messages (non-blocking)
    # This handles:
    # - Parameter updates from viewer → forwards to detection/decision servers
    # - Action requests from viewer → forwards to simulation + local handlers
    # - Vehicle status from simulation → forwards to viewers
    if broker:
        broker.poll()

    # Read subprocess outputs
    read_detection_output()
    read_decision_output()

    # Broadcast frames and detection data from shared memory
    if broker:
        # Read from shared memory channels
        image_msg = image_channel.read()
        detection_msg = detection_channel.read()

        if image_msg:
            broker.broadcast_frame(image_msg.image, image_msg.frame_id)

        if detection_msg:
            # Convert to viewer format
            detection_data = {
                'left_lane': {...} if detection_msg.left_lane else None,
                'right_lane': {...} if detection_msg.right_lane else None,
                'processing_time_ms': detection_msg.processing_time_ms,
                'frame_id': detection_msg.frame_id,
            }
            broker.broadcast_detection(detection_data, detection_msg.frame_id)

# Cleanup
if broker:
    broker.close()
```

### Detection/Decision Servers (Client Side)

```python
from lkas.integration.zmq import ParameterClient

class DetectionServer:
    def __init__(self):
        # Create parameter client
        self.param_client = ParameterClient(category='detection')
        self.param_client.register_callback(self._on_parameter_update)

    def _on_parameter_update(self, param_name: str, value: float):
        # Update detector parameters
        if hasattr(self.detector, 'update_parameter'):
            self.detector.update_parameter(param_name, value)

    def run(self):
        while self.running:
            # Poll for parameter updates (non-blocking)
            self.param_client.poll()

            # Do your detection work
            # ...

    def stop(self):
        self.param_client.close()
```

## Key Design Principles

### 1. Clean Separation of Concerns

- **Broker runs in LKAS main process**: Manages all ZMQ communication
- **Clients run in subprocesses**: Simple, focused API for receiving updates
- **No cross-dependencies**: LKAS doesn't import from simulation

### 2. Easy to Use

**Broker Side (LKAS):**
```python
broker = LKASBroker()  # One line initialization
broker.poll()          # Non-blocking poll in main loop
broker.close()         # Clean shutdown
```

**Client Side (Servers):**
```python
client = ParameterClient(category='detection')
client.register_callback(on_update)
client.poll()  # Non-blocking
client.close()
```

### 3. Non-Blocking Everything

- All `poll()` calls are non-blocking with timeouts
- Main loops never block on ZMQ operations
- Subprocesses continue working even if viewer disconnects

### 4. Fire-and-Forget Broadcasting

- Broadcaster doesn't wait for viewers
- High Water Mark (HWM) set to 10 to drop old frames
- Real-time data, not buffered history

### 5. Type Safety

- Structured message types: `VehicleState`, `ParameterUpdate`, `ActionRequest`
- Clear, documented interfaces
- Easy to extend

## Running LKAS with Broadcasting

```bash
# 1. Start CARLA
./CarlaUE4.sh

# 2. Start LKAS with broadcasting enabled (starts ZMQ broker)
lkas --method cv --broadcast

# 3. Start simulation with broadcasting (sends vehicle status to LKAS broker)
simulation --broadcast

# 4. Start viewer (connects to LKAS broker)
viewer

# Open browser: http://localhost:8080

# The viewer can now:
# - See live video stream with lane detection overlay
# - View vehicle state in real-time (speed, steering, position)
# - Adjust detection/decision parameters (Canny, Hough, Kp, Kd, etc.)
# - Send actions (pause, resume, respawn)

# Data flow:
# - Simulation → LKAS broker (port 5562) → Viewer (vehicle status)
# - LKAS (shared memory) → LKAS broker → Viewer (frames, detection)
# - Viewer → LKAS broker (port 5559) → Detection/Decision servers (parameters)
# - Viewer → LKAS broker (port 5558) → Simulation (actions)
```

## Migration from Simulation-based Broker

### Before (Simulation-based)

```python
# Broker running in simulation process
from simulation.integration.zmq_broadcast import ParameterSubscriber

# Detection server depends on simulation module
self.param_subscriber = ParameterSubscriber(connect_url=url)
self.param_subscriber.register_callback('detection', callback)
```

### After (LKAS-based)

```python
# Broker running in LKAS process
from lkas.integration.zmq import LKASBroker

broker = LKASBroker()  # Clean, self-contained

# Detection server uses clean API
from lkas.integration.zmq import ParameterClient

self.param_client = ParameterClient(category='detection')
self.param_client.register_callback(callback)
```

## Benefits

1. **Better Architecture**: Broker runs in the LKAS wrapper process (makes sense!)
2. **Clean APIs**: Easy for users to integrate ZMQ without knowing details
3. **No Cross-Dependencies**: LKAS is independent of simulation
4. **Encapsulation**: All ZMQ logic in one module
5. **Easy Testing**: Mock broker/client for unit tests
6. **Maintainability**: Changes to ZMQ don't affect other code

## Files

- `__init__.py`: Public API exports
- `broker.py`: LKASBroker (runs in LKAS main process)
- `client.py`: ParameterClient (used by detection/decision servers)
- `broadcaster.py`: VehicleBroadcaster (state/frame publishing)
- `messages.py`: Message type definitions
- `README.md`: This file

---

**Clean. Simple. Powerful.** 🚀
