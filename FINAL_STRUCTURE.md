# Final Project Structure

Date: 2025-10-27

## ✅ Completed Restructuring

Your project now has clean, simple module names at the root level.

## 📁 Final Directory Structure

```
ads_ld/
├── simulation/          ⭐ CARLA simulator integration
│   ├── __init__.py
│   ├── connection.py    # CARLA connection management
│   ├── vehicle.py       # Vehicle control
│   └── sensors.py       # Camera sensors
│
├── decision/            ⭐ Control decisions & analysis
│   ├── __init__.py
│   ├── analyzer.py      # Lane position analysis
│   └── controller.py    # PD controller for steering
│
└── detection/           ⭐ Lane detection & orchestration
    ├── main_modular.py           # Single-process entry point
    ├── main_distributed_v2.py    # Distributed system with web viewer
    ├── detection_server.py       # Standalone detection server
    ├── config.yaml               # Configuration
    │
    ├── core/                     # Core abstractions
    ├── detection_module/         # Detection wrapper
    ├── integration/              # Orchestration & communication
    ├── method/                   # CV & DL implementations
    ├── processing/               # Frame processing
    ├── ui/                       # Visualization
    ├── utils/                    # Utilities
    └── tests/                    # Tests
```

## 🎯 Module Purposes

### 1. `simulation/` - CARLA Integration
**Purpose:** Pure CARLA simulator interface
**Responsibilities:**
- Connect to CARLA server
- Spawn and control vehicles
- Manage camera sensors
- No business logic - just CARLA API wrapper

**Key Classes:**
- `CARLAConnection` - Server connection
- `VehicleManager` - Vehicle control
- `CameraSensor` - Camera management

### 2. `decision/` - Control Decisions
**Purpose:** Analyze lanes and generate control commands
**Responsibilities:**
- Analyze lane position relative to vehicle
- Calculate lateral offset and heading error
- Generate steering commands via PD controller
- Detect lane departures

**Key Classes:**
- `LaneAnalyzer` - Position analysis
- `DecisionController` - Control logic

### 3. `detection/` - Lane Detection & Orchestration
**Purpose:** Detect lanes and coordinate all modules
**Responsibilities:**
- Detect lanes using CV or DL methods
- Orchestrate data flow between modules
- Provide entry points for different architectures
- Handle distributed communication

**Key Files:**
- `main_modular.py` - Single-process architecture
- `main_distributed_v2.py` - Multi-process architecture
- `detection_server.py` - Standalone detection service

## 📦 Import Examples

### From Outside the Modules

```python
# Import simulation module
from simulation import CARLAConnection, VehicleManager, CameraSensor

# Import decision module
from decision import DecisionController, LaneAnalyzer

# Import detection components
from detection.core.config import ConfigManager
from detection.integration.orchestrator import SystemOrchestrator
from detection.method.computer_vision import CVLaneDetector
```

### Module Dependencies

```
simulation/     (Independent - only depends on CARLA package)
    ↓
detection/      (Uses simulation and decision)
    ↓
decision/       (Uses detection.core for data models)
```

## 🚀 Running the System

### Single-Process Mode

```bash
cd detection
python main_modular.py --method cv --host localhost --port 2000
```

### Distributed Mode (with Web Viewer)

```bash
# Terminal 1: Detection server
cd detection
python detection_server.py --method cv --port 5555

# Terminal 2: CARLA client
cd detection
python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --viewer web \
  --web-port 8080

# Open browser: http://localhost:8080
```

### With Remote CARLA Server

```bash
cd detection
python main_modular.py \
  --method cv \
  --host 192.168.1.100 \
  --port 2000
```

## ✅ Why These Names?

### `simulation/` (not `carla/`)
- ✅ Avoids conflict with CARLA Python package
- ✅ More generic - could swap simulators later
- ✅ Clear purpose - simulation integration

### `decision/` (not `decision_module/`)
- ✅ Short and clean
- ✅ Clearly describes function
- ✅ Easy to type

### `detection/` (not `lane_detection/`)
- ✅ Short and clean
- ✅ Main focus of the project
- ✅ Contains orchestration logic

## 🔄 Data Flow

```
┌──────────────┐
│ simulation/  │ ──→ Camera image
└──────────────┘
       ↓
┌──────────────┐
│ detection/   │ ──→ Lane lines
└──────────────┘
       ↓
┌──────────────┐
│ decision/    │ ──→ Steering command
└──────────────┘
       ↓
┌──────────────┐
│ simulation/  │ ──→ Apply control
└──────────────┘
```

## 📊 Module Independence

### Can be used independently:

1. **`simulation/`** - Use in any CARLA project
   ```python
   from simulation import CARLAConnection
   conn = CARLAConnection("localhost", 2000)
   ```

2. **`decision/`** - Use with any lane detection system
   ```python
   from decision import LaneAnalyzer
   analyzer = LaneAnalyzer(width=800, height=600)
   ```

3. **`detection/`** - Complete lane keeping system
   ```bash
   python detection/main_modular.py
   ```

## 🎓 Benefits of This Structure

1. **Simple Names** - Easy to remember and type
2. **Clear Separation** - Each module has distinct purpose
3. **No Naming Conflicts** - Avoids CARLA package conflict
4. **Reusable** - Each module can be used independently
5. **Scalable** - Easy to add new modules at root level

## 🧪 Verified Working

All imports tested and verified:
```
✓ simulation module imports correctly
✓ decision module imports correctly
✓ detection.main_modular imports correctly
✓ detection.main_distributed_v2 imports correctly
✓ detection.detection_server imports correctly
```

## 📝 Quick Reference

| Module | Purpose | Entry Point |
|--------|---------|-------------|
| `simulation/` | CARLA interface | - |
| `decision/` | Control logic | - |
| `detection/` | Detection & orchestration | `main_modular.py`, `main_distributed_v2.py` |

## 🎉 Ready to Use!

Your project structure is clean, simple, and production-ready!

Run from the detection directory:
```bash
cd detection
python main_distributed_v2.py --viewer opencv --detector-url tcp://localhost:5556
```
