# Simulation Module

**CARLA simulator integration for testing Lane Keeping Assist System in a virtual environment.**

## Overview

The simulation module bridges CARLA simulator with the LKAS pipeline, providing a realistic testing environment for autonomous driving algorithms. It manages vehicle spawning, camera sensors, and bidirectional communication with LKAS via ZMQ.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CARLA Simulator                       │
│                    (UE4 Engine)                         │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
       Vehicle Control              Camera Frames
             │                            │
┌────────────┴────────────────────────────┴───────────────┐
│              Simulation Orchestrator                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────┐        ┌──────────────────────┐   │
│  │ CARLA Manager  │        │  ZMQ Communication   │   │
│  │ - Vehicle      │   <->  │  - Send frames       │   │
│  │ - Camera       │        │  - Receive steering  │   │
│  │ - World        │        │  - Publish status    │   │
│  └────────────────┘        └──────────────────────┘   │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
          ZMQ Port    ZMQ Port    ZMQ Port
            5560        5563        5562
              │           │           │
        Send Frames  Receive     Publish Status
         to LKAS     Steering     to Broker
```

## Features

### CARLA Integration
- **Connection management:**
  - Automatic connection with retry logic
  - Graceful disconnection handling
  - Health monitoring
- **Vehicle spawning:**
  - Configurable vehicle models
  - Spawn point selection
  - Autopilot toggle
- **Camera sensors:**
  - RGB camera setup
  - Configurable resolution and FOV
  - Real-time frame capture
- **World management:**
  - Weather control
  - Time of day settings
  - Traffic and pedestrian spawning

### LKAS Communication (ZMQ)
- **Frame transmission:**
  - Sends camera frames to LKAS (port 5560)
  - High-frequency updates (30+ FPS)
  - Efficient serialization
- **Steering reception:**
  - Receives steering commands from LKAS (port 5563)
  - Applies control to CARLA vehicle
  - Low-latency actuation
- **Status broadcasting:**
  - Publishes vehicle state to LKAS broker (port 5562)
  - Speed, position, rotation data
  - Real-time telemetry

### Action Handling
- **Remote control from Viewer:**
  - **Pause:** Freeze simulation
  - **Resume:** Continue simulation
  - **Respawn:** Reset vehicle to spawn point
- Actions received from LKAS broker (port 5561)

### Performance Monitoring
- **Real-time metrics:**
  - Simulation FPS
  - Frame processing latency
  - LKAS response time
- **Comprehensive logging:**
  - Timestamped events
  - Error tracking
  - Performance diagnostics

## Quick Start

### Prerequisites

```bash
# Ensure CARLA is installed and running
cd ~/carla
./CarlaUE4.sh
```

### Basic Usage

```bash
# Start simulation with broadcasting
simulation --broadcast

# Start with custom spawn point
simulation --broadcast --spawn-id 123

# Start with specific vehicle model
simulation --config my_config.yaml --broadcast
```

### Full System Setup

```bash
# Terminal 1: CARLA simulator
cd ~/carla
./CarlaUE4.sh

# Terminal 2: LKAS with ZMQ broker
cd ~/ads_skynet
lkas --method cv --broadcast

# Terminal 3: Simulation
simulation --broadcast

# Terminal 4: Viewer (optional)
viewer

# Open browser: http://localhost:8080
```

## Configuration

Configuration is in `config.yaml` at project root.

### CARLA Settings

```yaml
carla:
  host: localhost
  port: 2000
  timeout: 10.0

  vehicle:
    model: vehicle.tesla.model3
    spawn_point: 0  # or null for random
    enable_autopilot: false

  camera:
    width: 640
    height: 480
    fov: 110
    x: 1.5
    y: 0.0
    z: 1.4
    pitch: 0
    yaw: 0
    roll: 0

  world:
    town: Town03
    weather: ClearNoon
```

### ZMQ Configuration

```yaml
zmq:
  simulation:
    # LKAS communication
    detection_output_port: 5560    # Send frames to LKAS
    decision_input_port: 5563      # Receive steering from LKAS

    # Broker communication
    status_publish_port: 5562      # Publish status to LKAS broker
    action_subscribe_port: 5561    # Receive actions from broker
```

## Module Structure

```
simulation/
├── run.py                        # Main entry point
├── orchestrator.py               # System coordinator
│
├── carla_api/                    # CARLA interface
│   ├── connection.py             # Connection management
│   ├── vehicle.py                # Vehicle control
│   ├── camera.py                 # Camera sensors
│   └── world.py                  # World management
│
├── integration/                  # Communication layer
│   ├── zmq_broadcast.py          # ZMQ publishers/subscribers
│   └── shared_memory/            # Legacy IPC (deprecated)
│       └── channels.py
│
├── utils/                        # Utilities
│   ├── visualizer.py             # Overlay rendering
│   └── metrics.py                # Performance tracking
│
└── constants.py                  # Configuration constants
```

## Communication Protocols

### Frame Output (to LKAS)
```python
# Simulation → LKAS (port 5560)
{
    "topic": "frame",
    "image": <numpy_array>,      # (H, W, 3) uint8
    "timestamp": 1234567890.123,
    "metadata": {
        "width": 640,
        "height": 480
    }
}
```

### Steering Input (from LKAS)
```python
# LKAS → Simulation (port 5563)
{
    "steering": 0.25,   # -1.0 (full left) to 1.0 (full right)
    "throttle": 0.14,   # 0.0 to 1.0
    "brake": 0.0        # 0.0 to 1.0
}
```

### Status Broadcast (to LKAS Broker)
```python
# Simulation → LKAS Broker (port 5562)
{
    "type": "state",
    "speed_kmh": 8.0,
    "position": {"x": 10.0, "y": 20.0, "z": 0.5},
    "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
    "steering": 0.1,
    "throttle": 0.14,
    "timestamp": 1234567890.123
}
```

### Action Reception (from LKAS Broker)
```python
# LKAS Broker → Simulation (port 5561)
{
    "action": "pause"    # pause, resume, respawn
}
```

## Performance

### Typical Metrics

- **Simulation FPS:** 30-60 FPS (depends on CARLA settings)
- **Frame transmission latency:** <10ms
- **Steering application latency:** <5ms
- **End-to-end latency:** 20-50ms (CARLA → LKAS → CARLA)

### Performance Tips

1. **Reduce camera resolution:**
   ```yaml
   carla:
     camera:
       width: 640
       height: 480
   ```

2. **Use simpler vehicle models:**
   ```yaml
   carla:
     vehicle:
       model: vehicle.tesla.model3  # Simpler than heavy trucks
   ```

3. **Optimize CARLA settings:**
   ```bash
   # Lower graphics quality in CARLA settings
   # Disable unnecessary actors (traffic, pedestrians)
   ```

## Development

### Adding Custom Vehicle Spawning

```python
# In carla_api/vehicle.py
def spawn_custom_vehicle(world, model_name, spawn_point):
    blueprint = world.get_blueprint_library().find(model_name)
    vehicle = world.spawn_actor(blueprint, spawn_point)
    return vehicle
```

### Custom Camera Configuration

```python
# In carla_api/camera.py
def setup_multi_camera(vehicle):
    # Front camera
    front_cam = setup_camera(vehicle, x=1.5, z=1.4)

    # Side cameras
    left_cam = setup_camera(vehicle, x=0.0, y=-1.0, z=1.4)
    right_cam = setup_camera(vehicle, x=0.0, y=1.0, z=1.4)

    return [front_cam, left_cam, right_cam]
```

### Debugging

Enable verbose logging:
```bash
simulation --broadcast --verbose
```

Monitor ZMQ ports:
```bash
# Check if ports are listening
ss -tlnp | grep '556[0-3]'

# Monitor ZMQ traffic (requires tcpdump)
sudo tcpdump -i lo -n port 5560
```

## Troubleshooting

### Cannot connect to CARLA
```
Error: Failed to connect to CARLA at localhost:2000
```
**Fix:**
- Ensure CARLA is running: `./CarlaUE4.sh`
- Check CARLA port in config: `carla.port`
- Verify firewall settings

### Vehicle not responding to steering
```
Warning: Steering commands not applied
```
**Fix:**
- Ensure LKAS is running and connected
- Check ZMQ port 5563 is available
- Verify `--broadcast` flag is used
- Check steering values are in range [-1.0, 1.0]

### Low FPS / Performance issues
```
Warning: Simulation FPS below 20
```
**Fix:**
- Reduce camera resolution in config
- Lower CARLA graphics quality
- Close unnecessary applications
- Use simpler map (Town01 instead of Town10HD)

### ZMQ communication errors
```
Error: ZMQ socket bind failed
```
**Fix:**
- Kill processes using the ports:
  ```bash
  lsof -ti:5560 | xargs kill -9
  lsof -ti:5563 | xargs kill -9
  ```
- Check port configuration in `config.yaml`
- Ensure no duplicate simulation instances

## Advanced Usage

### Multi-Vehicle Simulation

```python
# Custom orchestrator setup
orchestrator = SimulationOrchestrator(config)
orchestrator.spawn_vehicle("vehicle.tesla.model3", spawn_id=0)
orchestrator.spawn_vehicle("vehicle.audi.a2", spawn_id=50)
```

### Custom Weather Conditions

```yaml
carla:
  world:
    weather:
      cloudiness: 80.0
      precipitation: 30.0
      sun_altitude_angle: 70.0
```

### Recording and Replay

```bash
# Record simulation data
simulation --broadcast --record output.log

# Replay recorded data
simulation --replay output.log
```

## Integration with Other Modules

### With LKAS
```
Simulation → (port 5560) → LKAS (Detection)
LKAS (Decision) → (port 5563) → Simulation
Simulation → (port 5562) → LKAS Broker
```

### With Viewer
```
Simulation → LKAS Broker → (port 5557) → Viewer
Viewer → (port 5558) → LKAS Broker → (port 5561) → Simulation
```

## References

- [CARLA Documentation](https://carla.readthedocs.io/)
- [ZMQ Integration Guide](integration/README.md)
- [LKAS Module](../lkas/README.md)
- [Viewer Module](../viewer/README.md)
