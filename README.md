# Autonomous Driving Lane Keeping System

A modular, production-ready lane keeping system for CARLA simulator with clean separation of concerns.

## 🌟 Features

- **Clean 3-Module Architecture**: Simulation, Detection, Decision
- **Dual Detection Methods**: Computer Vision (OpenCV) and Deep Learning (PyTorch CNN)
- **Distributed System**: Run detection on remote GPU servers
- **Multiple Visualization Options**: OpenCV, Pygame, and Web viewer (no X11 needed!)
- **Production Ready**: Process isolation, ZMQ communication, fault tolerance
- **Modern Python Package**: `pyproject.toml`, editable install, entry point scripts

## 📦 Installation

### Prerequisites
- Python 3.10+
- CARLA 0.9.15+ simulator
- GPU (optional, for deep learning detection)

### Install Package

```bash
# Clone repository
git clone <repository-url>
cd seame-ads

# Install in editable mode with all dependencies
pip install -e .

# Or install with optional development tools
pip install -e ".[dev]"

# Or install everything (dev + training tools)
pip install -e ".[all]"
```

This installs the package as `seame-ads` with three command-line entry points:
- `simulation` - Main CARLA simulation
- `lane-detection` - Standalone detection server
- `viewer` - Remote web viewer (NEW: production mode)

## 🚀 Quick Start

### Option 1: Development Mode (Classic)

```bash
# Terminal 1: Start CARLA server
./CarlaUE4.sh

# Terminal 2: Start detection server (using installed entry point)
lane-detection --method cv --port 5556

# Terminal 3: Start CARLA simulation with web viewer
simulation --detector-url tcp://localhost:5556 --viewer web --web-port 8080

# Open browser: http://localhost:8080
```

### Option 2: Production Mode (NEW! Recommended)

**Better for real vehicles - separates rendering to laptop:**

```bash
# Terminal 1: Start CARLA server
./CarlaUE4.sh

# Terminal 2: Start detection server
lane-detection --method cv --port 5556

# Terminal 3: Start simulation with ZMQ broadcasting
simulation \
    --detector-url tcp://localhost:5556 \
    --viewer none \
    --broadcast detection-only

# Terminal 4: Start remote viewer (on laptop)
viewer --vehicle tcp://localhost:5557 --port 8080

# Open browser: http://localhost:8080
```

**Broadcast Modes:**
- `--broadcast none` - No broadcasting (default)
- `--broadcast detection-only` - Production mode (~9 KB/s, recommended for vehicles)
- `--broadcast with-images` - Development mode (~1.5 MB/s, includes raw images)

**Benefits:**
- ✅ Vehicle/sim CPU stays lightweight (no rendering!)
- ✅ Rich overlays drawn on laptop
- ✅ Remote monitoring capable
- ✅ Multiple viewers can connect

**Alternative (without entry points):**
```bash
# Terminal 2
python -m detection.detection --method cv --port 5556

# Terminal 3
python -m simulation.simulation --detector-url tcp://localhost:5556 --viewer web
```

## 📁 Project Structure

```
seame-ads/
├── pyproject.toml           # 📦 Package configuration & dependencies
├── config.yaml              # ⚙️ System configuration (auto-loaded from project root)
│
├── simulation/              ⭐ CARLA simulation & orchestration
│   ├── simulation.py        # Main entry point (installed as 'simulation' command)
│   ├── __init__.py          # Package exports
│   │
│   ├── connection.py        # CARLA connection
│   ├── vehicle.py           # Vehicle control
│   ├── sensors.py           # Camera sensors
│   │
│   ├── integration/         # System orchestration
│   │   ├── distributed_orchestrator.py  # Multi-process orchestrator
│   │   ├── communication.py           # ZMQ communication (req-rep)
│   │   ├── zmq_broadcast.py          # NEW: ZMQ broadcasting (pub-sub)
│   │   ├── shared_memory.py          # NEW: Shared memory (ultra-low latency)
│   │   ├── messages.py                # Message protocols
│   │   └── visualization.py           # Visualization manager
│   │
│   ├── processing/          # Frame processing
│   │   ├── frame_processor.py  # Processing pipeline
│   │   ├── pd_controller.py    # PD controller
│   │   └── metrics_logger.py   # Performance metrics
│   │
│   ├── ui/                  # User interface
│   │   ├── web_viewer.py    # Web-based viewer (no X11!)
│   │   ├── pygame_viewer.py  # Pygame viewer
│   │   ├── keyboard_handler.py  # Keyboard controls
│   │   └── video_recorder.py    # Video recording
│   │
│   └── utils/               # Utilities
│       ├── lane_analyzer.py     # Lane analysis
│       ├── visualizer.py        # Visualization helpers
│       └── spectator_overlay.py  # CARLA spectator overlay
│
├── detection/               ⭐ Pure lane detection
│   ├── detection.py         # Standalone server (installed as 'lane-detection' command)
│   ├── __init__.py          # Package exports
│   │
│   ├── core/                # Core abstractions
│   │   ├── interfaces.py    # Abstract base classes
│   │   ├── models.py        # Data models (Lane, Metrics)
│   │   ├── config.py        # Configuration management
│   │   └── factory.py       # Factory pattern
│   │
│   ├── detection_module/    # Detection wrapper
│   │   └── detector.py      # Detection module
│   │
│   ├── method/              # Detection implementations
│   │   ├── computer_vision/      # OpenCV-based
│   │   │   └── cv_lane_detector.py
│   │   └── deep_learning/        # CNN-based
│   │       ├── lane_net.py
│   │       └── lane_net_base.py
│   │
│   └── tests/               # Test suite
│       ├── test_connection.py
│       └── test_setup.py
│
├── decision/                ⭐ Control decisions
│   ├── analyzer.py          # Lane position analysis
│   └── controller.py        # PD control logic
│
├── viewer/                  ⭐ NEW: Remote web viewer
│   ├── run.py               # ZMQ-based viewer (installed as 'viewer' command)
│   ├── __init__.py          # Package exports
│   └── README.md            # Viewer documentation
│
└── .docs/                   # Documentation
    ├── START_HERE.md
    ├── QUICK_START.md
    ├── ARCHITECTURE_DECISION.md
    └── ...
```

## 🎯 Architecture

### Clean 3-Module Separation

```
┌──────────────────────────────────────────────────────────────┐
│                    simulation/                               │
│              (CARLA Orchestration Layer)                     │
│  • Runs CARLA simulation                                     │
│  • Coordinates modules                                       │
│  • Provides entry points                                     │
└──────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌────────────────┐  ┌───────────────────┐  ┌──────────────────┐
│  simulation/   │  │   detection/      │  │    decision/     │
│  (CARLA API)   │  │(Lane Detection)   │  │ (Control Logic)  │
│                │  │                   │  │                  │
│ • Connection   │  │ • CV Detection    │  │ • Lane Analysis  │
│ • Vehicle      │  │ • DL Detection    │  │ • PD Controller  │
│ • Sensors      │  │ • Pure algorithms │  │ • Steering       │
└────────────────┘  └───────────────────┘  └──────────────────┘
```

### Module Responsibilities

**`simulation/`** - CARLA Integration & Orchestration
- Connects to CARLA simulator
- Manages vehicles and sensors
- Orchestrates data flow between modules
- **Contains:** main entry points, orchestrators, UI

**`detection/`** - Pure Lane Detection
- Detects lanes from images (CV or DL)
- No CARLA dependencies
- Can run as standalone service
- **Contains:** detection algorithms, detection server

**`decision/`** - Control Decisions
- Analyzes lane position
- Generates steering commands
- PD control logic
- **Contains:** analyzer, controller

## 🎮 Usage

### Basic Usage (Local)

```bash
# Terminal 1: Start detection server
lane-detection --method cv --port 5556

# Terminal 2: Start CARLA simulation with web viewer
simulation \
  --detector-url tcp://localhost:5556 \
  --viewer web \
  --web-port 8080
```

### Remote CARLA Server

```bash
# Terminal 1: Detection server (on GPU machine)
lane-detection --method cv --port 5556

# Terminal 2: CARLA simulation (on CARLA machine)
simulation \
  --detector-url tcp://gpu-server-ip:5556 \
  --host <CARLA_HOST> \
  --port 2000 \
  --viewer web \
  --web-port 8080
```

### Deep Learning Detection

```bash
# Terminal 1: DL detection server (requires GPU)
lane-detection --method dl --port 5556 --gpu 0

# Terminal 2: CARLA simulation
simulation --detector-url tcp://localhost:5556 --viewer web
```

### Viewer Options

```bash
# Auto-detect best viewer (default)
simulation --detector-url tcp://localhost:5556 --viewer auto

# Web viewer (works in Docker, no X11 needed)
simulation --detector-url tcp://localhost:5556 --viewer web --web-port 8080

# OpenCV window (requires X11)
simulation --detector-url tcp://localhost:5556 --viewer opencv

# Pygame window
simulation --detector-url tcp://localhost:5556 --viewer pygame

# No visualization (headless)
simulation --detector-url tcp://localhost:5556 --no-display
```

## 🔧 Configuration

The system automatically loads `config.yaml` from the project root. You can also specify a custom config:

```bash
# Use project root config.yaml (default)
simulation

# Use custom config
simulation --config /path/to/custom-config.yaml

# Use built-in defaults (no file)
simulation --config default
```

### Configuration File Structure

Edit `config.yaml` in the project root:

```yaml
# CARLA Connection
carla:
  host: "localhost"
  port: 2000
  vehicle_type: "vehicle.tesla.model3"

# Camera Settings
camera:
  width: 800
  height: 600
  fov: 90.0
  position:
    x: 2.0
    y: 0.0
    z: 1.5
  rotation:
    pitch: -10.0
    yaw: 0.0
    roll: 0.0

# Lane Analysis & Control
lane_analyzer:
  kp: 0.5              # Proportional gain
  kd: 0.1              # Derivative gain
  drift_threshold: 0.15
  departure_threshold: 0.35

# Adaptive Throttle Policy
throttle_policy:
  base: 0.15           # Base throttle
  min: 0.05            # Minimum during turns
  steer_threshold: 0.15
  steer_max: 0.70
```

See [config.yaml](config.yaml) for full configuration options.

## 🧪 Testing

### Verify Installation

```bash
# Check if entry points are installed
which simulation
which lane-detection

# Test import
python -c "import detection; import simulation; import decision; print('✓ All modules imported')"
```

### Test Detection Server

```bash
# Terminal 1: Start server
lane-detection --method cv --port 5556

# Terminal 2: Test connection
python -c "from simulation.integration.communication import DetectionClient; print('✓ Detection server works')"
```

### Run Tests (if dev dependencies installed)

```bash
# Install with dev tools
pip install -e ".[dev]"

# Run tests
pytest
```

## 🔍 Keyboard Controls

When running with visualization:

- **Q** - Quit
- **S** - Toggle autopilot
- **O** - Toggle spectator overlay
- **F** - Toggle spectator follow mode
- **R** - Respawn vehicle
- **T** - Teleport to next spawn point

## 📊 Performance Metrics

```
Frame 00150 | FPS: 28.5 | Lanes: LR | Steering: +0.123 | Timeouts: 0
```

## 📋 System Requirements

### For M1 Mac Development
- Docker Desktop with Rosetta 2 enabled
- VSCode with Dev Containers extension
- Remote Linux machine running CARLA server

### For Native Linux Development
- Ubuntu 18.04+
- CARLA 0.9.15+ simulator
- Python 3.10+
- GPU (optional, for deep learning)

## 🚀 Development Setup

### Native Development

```bash
# Clone and install
git clone <repository-url>
cd seame-ads
pip install -e ".[dev]"

# Start developing
lane-detection --help
simulation --help
```

### Dev Container (M1 Mac / Remote Development)

1. **Open in Dev Container:**
   ```bash
   cd seame-ads
   code .
   # VSCode: Cmd+Shift+P → "Reopen in Container"
   ```

2. **Package is auto-installed in container**
   ```bash
   # Use entry points directly
   lane-detection --method cv --port 5556
   simulation --detector-url tcp://localhost:5556 --viewer web
   ```

See [.docs/DEVCONTAINER_SETUP.md](.docs/DEVCONTAINER_SETUP.md) for details.

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [.docs/START_HERE.md](.docs/START_HERE.md) | 👈 Start here! |
| [simulation/README.md](simulation/README.md) | Simulation module guide |
| [.docs/ARCHITECTURE_DECISION.md](.docs/ARCHITECTURE_DECISION.md) | Architecture rationale |
| [.docs/DEVCONTAINER_SETUP.md](.docs/DEVCONTAINER_SETUP.md) | Dev container setup |
| [.docs/VISUALIZATION_GUIDE.md](.docs/VISUALIZATION_GUIDE.md) | Visualization options |
| [.docs/DISTRIBUTED_ARCHITECTURE.md](.docs/DISTRIBUTED_ARCHITECTURE.md) | Distributed system design |

## 🎓 For Students

This project demonstrates:

- ✅ **Clean Architecture**: Separation of concerns
- ✅ **Design Patterns**: Factory, Strategy, Observer
- ✅ **Distributed Systems**: ZMQ communication
- ✅ **Multiple Algorithms**: CV and DL approaches
- ✅ **Production Ready**: Error handling, logging, metrics

## 🆘 Quick Reference

### Installed Commands

After `pip install -e .`, you get two entry points:

| Command | Purpose | Equivalent Python Module |
|---------|---------|--------------------------|
| `simulation` | Main CARLA simulation | `python -m simulation.simulation` |
| `lane-detection` | Detection server | `python -m detection.detection` |

### Command Templates

```bash
# Start detection server (Terminal 1)
lane-detection --method cv --port 5556

# Start CARLA simulation (Terminal 2)
simulation \
  --detector-url tcp://localhost:5556 \
  --viewer web \
  --web-port 8080

# OpenCV viewer instead of web
simulation --detector-url tcp://localhost:5556 --viewer opencv

# Pygame viewer
simulation --detector-url tcp://localhost:5556 --viewer pygame

# Remote CARLA + custom config
simulation \
  --host <REMOTE_IP> \
  --port 2000 \
  --detector-url tcp://localhost:5556 \
  --config /path/to/config.yaml
```

### Package Structure

After installation, import modules directly:

```python
# Import detection
from detection.core.config import ConfigManager
from detection.core.models import Lane, DetectionResult
from detection import LaneDetection

# Import simulation
from simulation import CARLAConnection, VehicleManager
from simulation.integration.communication import DetectionClient

# Import decision
from decision import DecisionController, LaneAnalyzer
```

## ✅ Why This Structure?

1. **`simulation/` contains orchestration** - Everything related to running CARLA simulations
2. **`detection/` is pure algorithms** - Can be used in any project, no CARLA dependency
3. **`decision/` is reusable logic** - Works with any detection system
4. **Clear responsibilities** - Each module has ONE job
5. **Easy to test** - Pure functions, no entangled dependencies

## 🎁 Modern Python Package Benefits

This project uses modern Python packaging (`pyproject.toml`) instead of legacy `setup.py` and `requirements.txt`:

### ✅ Benefits

1. **Single Source of Truth** - All configuration in `pyproject.toml`
   - Dependencies, metadata, build config, tool settings
   - No more scattered `setup.py`, `requirements.txt`, `setup.cfg`, etc.

2. **Clean Imports** - No more `sys.path` hacks!
   ```python
   # ❌ Old way (brittle)
   sys.path.insert(0, str(Path(__file__).parent.parent))
   from detection.core.models import Lane

   # ✅ New way (clean)
   from detection.core.models import Lane
   ```

3. **Entry Point Scripts** - Installed commands available system-wide
   ```bash
   simulation --help      # Works from any directory
   lane-detection --help  # No need to cd into specific folders
   ```

4. **Editable Install** - Changes reflect immediately
   ```bash
   pip install -e .       # Edit code and run without reinstalling
   ```

5. **Optional Dependencies** - Install only what you need
   ```bash
   pip install -e .           # Basic install
   pip install -e ".[dev]"    # + development tools
   pip install -e ".[train]"  # + ML training tools
   pip install -e ".[all]"    # Everything
   ```

6. **Auto-Config Discovery** - Config file found automatically
   - Looks for `pyproject.toml` to find project root
   - Loads `config.yaml` from project root automatically
   - No hardcoded paths or relative path issues

7. **Tool Configuration** - Unified config for dev tools
   - pytest, black, mypy, isort all configured in `pyproject.toml`
   - Consistent formatting across team

### 📦 Package Info

- **Name**: `seame-ads`
- **Version**: 0.1.0
- **Python**: 3.10+
- **License**: See LICENSE file

## 📝 License

See [LICENSE](LICENSE) file.

---

**Ready to start?** 👉 See [Quick Start](#-quick-start) above
