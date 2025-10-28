# Quick Start Guide - Modular Lane Keeping System

## What's New?

Your lane keeping system has been refactored into **three independent modules**:

1. **CARLA Module** - Handles simulator, vehicle, and sensors
2. **Detection Module** - Processes images and detects lanes
3. **Decision Module** - Analyzes lanes and generates control commands

## Running the System

### 1. Start CARLA (in a separate terminal)
```bash
./CarlaUE4.sh
```

### 2. Run the Modular System
```bash
cd /workspaces/ads_ld/lane_detection

# Run with Computer Vision detection
python main_modular.py --method cv

# Run with Deep Learning detection
python main_modular.py --method dl
```

### 3. Options
```bash
# Run without display
python main_modular.py --method cv --no-display

# Connect to remote CARLA
python main_modular.py --host 192.168.1.100 --port 2000

# Specific spawn point
python main_modular.py --spawn-point 5

# Disable autopilot (use pure lane keeping control)
python main_modular.py --no-autopilot
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                            │
│            (integration/orchestrator.py)                  │
└──────────────────────────────────────────────────────────┘
    │ coordinates           │                │
    ▼                       ▼                ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   CARLA     │───>│  DETECTION   │───>│  DECISION    │
│   MODULE    │    │    MODULE    │    │    MODULE    │
│             │<───┼──────────────┘    │              │
└─────────────┘    │  Sends:           └──────────────┘
                   │  LaneMessage           │
Provides:          │                        │ Generates:
- ImageMessage     └────────────────────────┘ ControlMessage
- Vehicle control

Data Flow:
1. CARLA captures image
2. Detection detects lanes
3. Decision generates steering
4. CARLA applies control
```

## Module Details

### CARLA Module (`modules/carla_module/`)
- **connection.py**: Connects to CARLA server
- **vehicle.py**: Spawns and controls vehicle
- **sensors.py**: Camera sensor management

### Detection Module (`modules/detection_module/`)
- **detector.py**: Wraps CV/DL detection methods
- Returns lane lines with confidence scores
- Completely independent from CARLA

### Decision Module (`modules/decision_module/`)
- **analyzer.py**: Analyzes lane geometry
- **controller.py**: PD controller + decision logic
- Generates steering/throttle/brake commands

### Integration Layer (`integration/`)
- **messages.py**: Data structures for communication
- **orchestrator.py**: Coordinates the three modules

## Key Files

| File | Purpose |
|------|---------|
| `main_modular.py` | **NEW** entry point (recommended) |
| `main.py` | OLD entry point (still works) |
| `MODULAR_ARCHITECTURE.md` | Detailed architecture documentation |
| `integration/messages.py` | Message definitions |
| `integration/orchestrator.py` | System coordinator |

## Benefits

✅ **Independent Modules**: Each can be developed/tested separately
✅ **Clear Interfaces**: Well-defined message passing
✅ **Easy Testing**: Mock messages for unit tests
✅ **Reusable**: Detection module works without CARLA
✅ **Extensible**: Easy to add new algorithms

## Examples

### Use Detection Module Standalone
```python
from modules.detection_module import LaneDetectionModule
from core.config import ConfigManager
import cv2

config = ConfigManager.load('config.yaml')
detector = LaneDetectionModule(config, method='cv')

image = cv2.imread('test.jpg')
# ... process image
```

### Use Decision Module Standalone
```python
from modules.decision_module import DecisionController

controller = DecisionController(800, 600, kp=0.5, kd=0.1)
# ... generate control from detection
```

## Troubleshooting

**Import errors?**
```bash
# Make sure you're in the lane_detection directory
cd /workspaces/ads_ld/lane_detection
python main_modular.py
```

**CARLA connection fails?**
```bash
# Check if CARLA is running
./CarlaUE4.sh

# Try different port
python main_modular.py --port 2000
```

## Next Steps

1. Read `MODULAR_ARCHITECTURE.md` for detailed documentation
2. Try running with different detection methods
3. Experiment with controller gains (edit `config.yaml`)
4. Add your own detection or control algorithms

Enjoy the new modular architecture! 🚗
