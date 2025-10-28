# Modular Lane Keeping System Architecture

## Overview

The system has been refactored into **three independent modules** that communicate through well-defined message interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│           (Coordinates all three modules)                    │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    CARLA     │───>│  DETECTION   │───>│  DECISION    │
│   MODULE     │    │   MODULE     │    │   MODULE     │
└──────────────┘    └──────────────┘    └──────────────┘
         ▲                                        │
         └────────────────────────────────────────┘
```

## Module Descriptions

### 1. CARLA Module (`modules/carla_module/`)

**Responsibility**: Interface with CARLA simulator

**Components**:
- `connection.py`: CARLA server connection management
- `vehicle.py`: Vehicle spawning, control, and state management
- `sensors.py`: Camera sensor setup and image capture

**Key Features**:
- Connects to CARLA simulator
- Spawns and manages vehicle
- Captures camera images
- Applies control commands to vehicle

**Interface**:
- **Output**: `ImageMessage` (RGB images to Detection Module)
- **Input**: `ControlMessage` (steering, throttle, brake from Decision Module)

---

### 2. Detection Module (`modules/detection_module/`)

**Responsibility**: Process images and detect lane markings

**Components**:
- `detector.py`: Main detection interface that wraps CV/DL methods
- Uses existing detection algorithms from `method/computer_vision/` and `method/deep_learning/`

**Key Features**:
- Standalone lane detection
- Supports multiple detection methods (CV, DL)
- Returns structured lane information
- No dependency on CARLA or control logic

**Interface**:
- **Input**: `ImageMessage` (from CARLA Module)
- **Output**: `DetectionMessage` (lane lines to Decision Module)

---

### 3. Decision Module (`modules/decision_module/`)

**Responsibility**: Analyze lanes and generate control commands

**Components**:
- `analyzer.py`: Lane geometry analysis and metrics computation
- `controller.py`: PD controller for steering + decision logic

**Key Features**:
- Analyzes lane geometry
- Computes vehicle position relative to lanes
- Generates steering commands using PD control
- Manages throttle and brake
- No dependency on CARLA or detection internals

**Interface**:
- **Input**: `DetectionMessage` (from Detection Module)
- **Output**: `ControlMessage` (to CARLA Module)

---

### 4. Integration Layer (`integration/`)

**Responsibility**: Coordinate the three modules

**Components**:
- `messages.py`: Data structures for inter-module communication
- `orchestrator.py`: Main coordinator that runs the control loop

**Key Features**:
- Defines message formats between modules
- Coordinates data flow: CARLA → Detection → Decision → CARLA
- Manages system lifecycle (init, run, shutdown)
- Tracks performance metrics

---

## Message Flow

### Step-by-Step Execution:

1. **CARLA Module** captures image from camera sensor
   ```
   ImageMessage {
     image: np.ndarray (RGB)
     timestamp: float
     frame_id: int
   }
   ```

2. **Detection Module** processes image and detects lanes
   ```
   DetectionMessage {
     left_lane: LaneMessage (x1, y1, x2, y2)
     right_lane: LaneMessage (x1, y1, x2, y2)
     processing_time_ms: float
     debug_image: np.ndarray (optional)
   }
   ```

3. **Decision Module** analyzes lanes and computes control
   ```
   ControlMessage {
     steering: float [-1, 1]
     throttle: float [0, 1]
     brake: float [0, 1]
     mode: ControlMode
     lateral_offset: float (diagnostic)
     heading_angle: float (diagnostic)
   }
   ```

4. **CARLA Module** applies control to vehicle
   - Actuates steering, throttle, brake
   - Cycle repeats

---

## Directory Structure

```
lane_detection/
├── modules/
│   ├── carla_module/           # CARLA interaction
│   │   ├── __init__.py
│   │   ├── connection.py       # Server connection
│   │   ├── vehicle.py          # Vehicle management
│   │   └── sensors.py          # Camera sensor
│   │
│   ├── decision_module/        # Decision & Control
│   │   ├── __init__.py
│   │   ├── analyzer.py         # Lane analysis
│   │   └── controller.py       # Control generation
│   │
│   └── detection_module/       # Lane Detection
│       ├── __init__.py
│       └── detector.py         # Detection interface
│
├── integration/
│   ├── __init__.py
│   ├── messages.py             # Message definitions
│   └── orchestrator.py         # System coordinator
│
├── main_modular.py             # NEW entry point
├── main.py                     # OLD entry point (still works)
│
├── method/                     # Detection algorithms
│   ├── computer_vision/        # CV-based detection
│   └── deep_learning/          # DL-based detection
│
├── core/                       # Shared infrastructure
├── processing/                 # Processing utilities
└── utils/                      # Utility functions
```

---

## Usage

### Running the Modular System

```bash
# Start CARLA server first
./CarlaUE4.sh

# Run the modular system (CV method)
python main_modular.py --method cv

# Run with DL method
python main_modular.py --method dl

# Run without visualization
python main_modular.py --method cv --no-display

# Connect to remote CARLA
python main_modular.py --host 192.168.1.100 --port 2000
```

### Command Line Options

- `--method`: Detection method (`cv` or `dl`)
- `--config`: Path to config file (default: `config.yaml`)
- `--host`: CARLA server host (default: `localhost`)
- `--port`: CARLA server port (default: `2000`)
- `--spawn-point`: Vehicle spawn point index
- `--no-display`: Disable visualization window
- `--no-autopilot`: Disable CARLA autopilot (use pure lane keeping control)

---

## Benefits of Modular Architecture

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Changes to one module don't affect others
- Easier to understand and maintain

### 2. **Independent Development**
- Modules can be developed and tested independently
- Different teams can work on different modules
- Easy to swap implementations (e.g., different detection algorithms)

### 3. **Reusability**
- Detection module can be used in other projects
- Decision module can work with different simulators
- CARLA module can be reused for other tasks

### 4. **Testability**
- Each module can be unit tested independently
- Mock messages for integration testing
- No need for full CARLA setup to test individual modules

### 5. **Flexibility**
- Easy to add new detection methods
- Can swap out control algorithms
- Support multiple simulators (not just CARLA)

---

## Example: Using Modules Independently

### Using Detection Module Standalone

```python
from modules.detection_module import LaneDetectionModule
from core.config import ConfigManager
from integration.messages import ImageMessage
import cv2

# Load config and create detector
config = ConfigManager.load('config.yaml')
detector = LaneDetectionModule(config, method='cv')

# Process an image
image = cv2.imread('test_image.jpg')
image_msg = ImageMessage(image=image, timestamp=0.0, frame_id=0)

detection = detector.process_image(image_msg)

print(f"Left lane: {detection.left_lane}")
print(f"Right lane: {detection.right_lane}")
```

### Using Decision Module Standalone

```python
from modules.decision_module import DecisionController
from integration.messages import DetectionMessage, LaneMessage

# Create controller
controller = DecisionController(image_width=800, image_height=600)

# Create mock detection
detection = DetectionMessage(
    left_lane=LaneMessage(x1=200, y1=600, x2=350, y2=400, confidence=0.9),
    right_lane=LaneMessage(x1=600, y1=600, x2=450, y2=400, confidence=0.9),
    processing_time_ms=15.0,
    frame_id=0,
    timestamp=0.0
)

# Get control command
control = controller.process_detection(detection)
print(f"Steering: {control.steering}")
print(f"Throttle: {control.throttle}")
```

---

## Migration from Old Architecture

The old architecture (`main.py`) still works and is kept for compatibility.

**Key differences**:

| Aspect | Old Architecture | New Modular Architecture |
|--------|-----------------|-------------------------|
| Entry point | `main.py` | `main_modular.py` |
| Structure | Monolithic class | Three independent modules |
| Communication | Direct method calls | Message passing |
| Testability | Coupled | Independent modules |
| Reusability | Limited | High |

**Recommendation**: Use `main_modular.py` for new development.

---

## Future Enhancements

With the modular architecture, the following enhancements are easier:

1. **Multiple Detection Methods**: Run CV and DL in parallel, compare results
2. **Advanced Control**: Add MPC, LQR controllers alongside PD
3. **Multi-Simulator Support**: Add support for other simulators (AirSim, LGSVL)
4. **Sensor Fusion**: Add LIDAR, radar sensors to Detection Module
5. **Cloud Processing**: Run Detection Module on remote GPU server
6. **Logging & Replay**: Record messages for offline analysis
7. **ROS Integration**: Adapt to ROS message format

---

## Development Guidelines

### Adding a New Detection Method

1. Create detector class implementing `LaneDetector` interface
2. Register in `core/factory.py`
3. Detection Module automatically supports it
4. No changes needed to Decision or CARLA modules

### Adding a New Control Algorithm

1. Create controller in `modules/decision_module/`
2. Modify `DecisionController` to use it
3. Interface remains the same (receives `DetectionMessage`, returns `ControlMessage`)
4. No changes needed to Detection or CARLA modules

### Supporting a New Simulator

1. Create new module in `modules/` (e.g., `airsim_module/`)
2. Implement same interface (provide `ImageMessage`, accept `ControlMessage`)
3. Update orchestrator to use new module
4. Detection and Decision modules work without changes

---

## Summary

The modular architecture provides:
- ✅ **Clear separation** of CARLA, Detection, and Decision logic
- ✅ **Well-defined interfaces** using message classes
- ✅ **Independent modules** that can be developed and tested separately
- ✅ **Easy extensibility** for new features and algorithms
- ✅ **Better maintainability** with single-responsibility modules

Use `main_modular.py` to run the new architecture!
