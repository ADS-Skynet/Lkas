# Autonomous Driving Lane Keeping System

A modular, production-ready lane keeping system for CARLA simulator with clean separation of concerns.

## 🌟 Features

- **Clean 3-Module Architecture**: Simulation, Detection, Decision
- **Dual Detection Methods**: Computer Vision (OpenCV) and Deep Learning (PyTorch CNN)
- **Distributed System**: Run detection on remote GPU servers
- **Multiple Visualization Options**: OpenCV, Pygame, and Web viewer (no X11 needed!)
- **Production Ready**: Process isolation, ZMQ communication, fault tolerance

## 🚀 Quick Start

```bash
# Terminal 1: Start CARLA server
./CarlaUE4.sh

# Terminal 2: Start detection server
cd detection
python detection_server.py --method cv --port 5555

# Terminal 3: Start CARLA simulation with web viewer
cd simulation
python main_distributed_v2.py --detector-url tcp://localhost:5555 --viewer web --web-port 8080

# Open browser: http://localhost:8080
```

## 📁 Project Structure

```
ads_ld/
├── simulation/              ⭐ CARLA simulation & orchestration
│   ├── main_distributed_v2.py  # Main entry point (distributed system)
│   ├── config.yaml          # Configuration
│   │
│   ├── connection.py        # CARLA connection
│   ├── vehicle.py           # Vehicle control
│   ├── sensors.py           # Camera sensors
│   │
│   ├── integration/         # System orchestration
│   │   ├── distributed_orchestrator.py  # Multi-process orchestrator
│   │   ├── communication.py           # ZMQ communication
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
│   ├── detection_server.py  # Standalone detection server
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
cd detection
python detection_server.py --method cv --port 5555

# Terminal 2: Start CARLA simulation with web viewer
cd simulation
python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --viewer web \
  --web-port 8080
```

### Remote CARLA Server

```bash
# Terminal 1: Detection server (on GPU machine)
cd detection
python detection_server.py --method cv --port 5555

# Terminal 2: CARLA simulation (on CARLA machine)
cd simulation
python main_distributed_v2.py \
  --detector-url tcp://gpu-server-ip:5555 \
  --carla-host localhost \
  --carla-port 2000 \
  --viewer web \
  --web-port 8080
```

### Deep Learning Detection

```bash
# Terminal 1: DL detection server
cd detection
python detection_server.py --method dl --model path/to/model.pth --port 5555

# Terminal 2: CARLA simulation
cd simulation
python main_distributed_v2.py --detector-url tcp://localhost:5555 --viewer web
```

## 🔧 Configuration

Edit `simulation/config.yaml`:

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
  fov: 90
  position: [2.5, 0.0, 1.0]
  rotation: [-15.0, 0.0, 0.0]

# Controller
controller:
  kp: 0.5
  kd: 0.1
  max_steering: 0.8
```

## 🧪 Testing

### Test Without CARLA

```bash
cd detection
python tests/test_setup.py
```

### Test CARLA Connection

```bash
cd detection
python tests/test_connection.py --host localhost --port 2000
```

### Test Detection Server

```bash
# Terminal 1
cd detection
python detection_server.py --port 5555

# Terminal 2
python -c "from simulation.integration.communication import DetectionClient; print('✓ Works')"
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

## 🚀 Development Setup (M1 Mac)

1. **Enable Rosetta 2 in Docker**
2. **Open in Dev Container:**
   ```bash
   cd ads_ld
   code .
   # VSCode: Cmd+Shift+P → "Reopen in Container"
   ```
3. **Start detection server and connect to Remote CARLA:**
   ```bash
   # Terminal 1: Detection server
   cd detection
   python detection_server.py --method cv --port 5555

   # Terminal 2: CARLA simulation
   cd simulation
   python main_distributed_v2.py \
     --detector-url tcp://localhost:5555 \
     --carla-host <LINUX_IP> \
     --carla-port 2000 \
     --viewer web
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

### Entry Points

| File | Purpose | Location |
|------|---------|----------|
| `main_distributed_v2.py` | Main system entry point | `simulation/` |
| `detection_server.py` | Standalone detection server | `detection/` |

### Command Templates

```bash
# Start detection server (Terminal 1)
cd detection && python detection_server.py --method cv --port 5555

# Start CARLA simulation (Terminal 2)
cd simulation && python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --viewer web \
  --web-port 8080

# OpenCV viewer instead of web
cd simulation && python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --viewer opencv

# Pygame viewer
cd simulation && python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --viewer pygame
```

## ✅ Why This Structure?

1. **`simulation/` contains orchestration** - Everything related to running CARLA simulations
2. **`detection/` is pure algorithms** - Can be used in any project, no CARLA dependency
3. **`decision/` is reusable logic** - Works with any detection system
4. **Clear responsibilities** - Each module has ONE job
5. **Easy to test** - Pure functions, no entangled dependencies

## 📝 License

See [LICENSE](LICENSE) file.

---

**Ready to start?** 👉 See [Quick Start](#-quick-start) above
