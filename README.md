# Lane Detection for Autonomous Driving - CARLA Implementation

A modular, production-ready lane detection system for CARLA simulator with support for both Computer Vision and Deep Learning methods.

## 🌟 Features

- **Modular Architecture**: Clean separation between CARLA, Detection, and Decision modules
- **Dual Detection Methods**: Computer Vision (OpenCV) and Deep Learning (PyTorch CNN)
- **Distributed System**: Run detection on remote GPU servers
- **Multiple Visualization Options**: OpenCV, Pygame, and Web viewer (no X11 needed!)
- **Production Ready**: Process isolation, ZMQ communication, fault tolerance
- **Dev Container**: Seamless development on M1 Mac with Docker

## 🚀 Quick Start

### Option 1: Single-Process Mode (Recommended for Testing)

```bash
# Start CARLA server (on Linux machine or same machine)
./CarlaUE4.sh

# Run lane detection (in dev container or local environment)
cd lane_detection
python main_modular.py --method cv --host localhost --port 2000
```

### Option 2: Distributed Mode (Recommended for Production)

```bash
# Terminal 1: Start detection server
cd lane_detection
python detection_server.py --method cv --port 5555

# Terminal 2: Start CARLA client with web viewer
python main_distributed_v2.py --detector-url tcp://localhost:5555 --viewer web --web-port 8080

# Open browser: http://localhost:8080
```

## 📋 System Requirements

### For M1 Mac Development
- **Docker Desktop** with Rosetta 2 enabled
- **VSCode** with Dev Containers extension
- **Remote Linux machine** running CARLA server (x86_64)

### For Native Linux Development
- **Ubuntu 18.04+** (x86_64)
- **CARLA 0.9.15** simulator
- **Python 3.10+**
- **GPU** (optional, for deep learning)

## 📁 Project Structure

```
ads_ld/
├── lane_detection/                 # Main package
│   ├── main_modular.py            # Single-process entry point ⭐
│   ├── main_distributed_v2.py     # Distributed system with web viewer ⭐
│   ├── detection_server.py        # Standalone detection server ⭐
│   ├── config.yaml                # Configuration file
│   │
│   ├── core/                      # Core abstractions
│   │   ├── interfaces.py          # Abstract base classes
│   │   ├── models.py              # Data models (Lane, Metrics)
│   │   ├── config.py              # Configuration management
│   │   └── factory.py             # Factory pattern for detectors
│   │
│   ├── modules/                   # Three main modules
│   │   ├── carla_module/          # CARLA simulator integration
│   │   │   ├── connection.py      # CARLA connection
│   │   │   ├── vehicle.py         # Vehicle management
│   │   │   └── sensors.py         # Camera sensor
│   │   ├── detection_module/      # Lane detection
│   │   │   └── detector.py        # Detection wrapper
│   │   └── decision_module/       # Control decisions
│   │       ├── analyzer.py        # Lane analysis
│   │       └── controller.py      # PD controller
│   │
│   ├── method/                    # Detection implementations
│   │   ├── computer_vision/       # OpenCV-based detection
│   │   │   └── cv_lane_detector.py
│   │   └── deep_learning/         # CNN-based detection
│   │       ├── lane_net.py        # Network architectures
│   │       └── lane_net_base.py   # Training/inference base
│   │
│   ├── integration/               # System integration
│   │   ├── orchestrator.py        # Single-process orchestrator
│   │   ├── distributed_orchestrator.py  # Multi-process orchestrator
│   │   ├── communication.py       # ZMQ client/server
│   │   ├── messages.py            # Message protocols
│   │   └── visualization.py       # Visualization manager
│   │
│   ├── ui/                        # User interface components
│   │   ├── web_viewer.py          # Web-based viewer (no X11!) ⭐
│   │   ├── pygame_viewer.py       # Pygame viewer
│   │   ├── keyboard_handler.py    # Keyboard controls
│   │   └── video_recorder.py      # Video recording
│   │
│   ├── processing/                # Frame processing
│   │   ├── frame_processor.py     # Frame processing pipeline
│   │   ├── pd_controller.py       # PD controller
│   │   └── metrics_logger.py      # Performance metrics
│   │
│   ├── utils/                     # Utilities
│   │   ├── lane_analyzer.py       # Lane analysis
│   │   ├── visualizer.py          # Visualization helpers
│   │   └── spectator_overlay.py   # CARLA spectator overlay
│   │
│   ├── tests/                     # Test suite
│   │   ├── test_connection.py     # CARLA connection tests
│   │   └── test_setup.py          # Setup verification
│   │
│   └── scripts/                   # Utility scripts
│       └── start_distributed_system.sh
│
├── docs/                          # Documentation
│   ├── START_HERE.md              # 👈 Start here!
│   ├── QUICK_START.md             # Quick start guide
│   ├── ARCHITECTURE_DECISION.md   # Architecture rationale
│   ├── MODULAR_ARCHITECTURE.md    # Architecture explanation
│   ├── DEVCONTAINER_SETUP.md      # Dev container setup
│   ├── MACOS_M1_SETUP.md          # M1 Mac specific setup
│   └── DL_QUICKSTART.md           # Deep learning setup
│
├── archive/                       # Deprecated files (for reference)
│   ├── deprecated_main_files/     # Old entry points
│   ├── old_temp_files/            # Old demo files
│   └── deprecated_docs/           # Historical documentation
│
├── .devcontainer/                 # Dev container configuration
│   ├── devcontainer.json          # VSCode configuration
│   └── Dockerfile                 # Container definition
│
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Docker compose configuration
├── CLEANUP_SUMMARY.md             # Recent cleanup details
└── README.md                      # This file
```

## 🎯 Architecture

### Modular Design

The system follows a clean **three-module architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestrator                             │
│                  (Coordinates modules)                        │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  CARLA Module   │  │Detection Module │  │ Decision Module  │
│                 │  │                 │  │                  │
│ • Connection    │  │ • CV Detector   │  │ • Lane Analyzer  │
│ • Vehicle       │  │ • DL Detector   │  │ • PD Controller  │
│ • Camera        │  │ • Factory       │  │ • Control Logic  │
└─────────────────┘  └─────────────────┘  └──────────────────┘
```

### Distributed Architecture (Production)

For production deployments, the detection can run on a separate process/machine:

```
┌─────────────────────────────────────────────────────────────┐
│  CARLA Client Process                                        │
│  • Vehicle control                                           │
│  • Decision making                                           │
│  • Visualization                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ ZMQ (TCP)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Detection Server Process (Can be remote GPU machine!)      │
│  • Lane detection                                            │
│  • Computer Vision or Deep Learning                          │
│  • Optimized for GPU inference                               │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Config File (`lane_detection/config.yaml`)

```yaml
# CARLA Connection
carla:
  host: "localhost"        # CARLA server host
  port: 2000               # CARLA server port
  vehicle_type: "vehicle.tesla.model3"

# Camera Settings
camera:
  width: 800
  height: 600
  fov: 90
  position: [2.5, 0.0, 1.0]     # [x, y, z] relative to vehicle
  rotation: [-15.0, 0.0, 0.0]   # [pitch, yaw, roll]

# Lane Detection
detection:
  method: "cv"             # "cv" or "dl"

  cv:
    roi_top_ratio: 0.4
    canny_low: 50
    canny_high: 150

  dl:
    model_path: null       # Path to trained model
    input_size: [256, 256]

# Controller
controller:
  kp: 0.5                  # Proportional gain
  kd: 0.1                  # Derivative gain
  max_steering: 0.8        # Maximum steering angle

# Lane Analysis
analyzer:
  drift_threshold: 50      # Pixels
  departure_threshold: 150 # Pixels

# Visualization
visualization:
  show_spectator_overlay: true
  follow_with_spectator: true
```

## 🎮 Usage Examples

### 1. Single-Process with Computer Vision

```bash
cd lane_detection
python main_modular.py --method cv --host localhost --port 2000
```

### 2. Single-Process with Deep Learning

```bash
cd lane_detection
python main_modular.py --method dl --model path/to/model.pth
```

### 3. Distributed with Web Viewer (Best for Docker/Remote)

```bash
# Terminal 1: Detection server
python detection_server.py --method cv --port 5555

# Terminal 2: CARLA client with web viewer
python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --viewer web \
  --web-port 8080

# Open browser: http://localhost:8080
```

### 4. Distributed with Remote Detection Server

```bash
# On GPU machine: Start detection server
python detection_server.py --method dl --port 5555

# On local machine: Run CARLA client
python main_distributed_v2.py \
  --detector-url tcp://192.168.1.100:5555 \
  --host localhost \
  --port 2000
```

### 5. No Display Mode (Headless)

```bash
python main_modular.py --method cv --no-display
```

## 🧪 Testing

### Test Without CARLA (Standalone)

```bash
cd lane_detection
python tests/test_setup.py
```

### Test CARLA Connection

```bash
cd lane_detection
python tests/test_connection.py --host localhost --port 2000
```

### Test Detection Server

```bash
# Terminal 1: Start server
python detection_server.py --port 5555

# Terminal 2: Test client
python -c "
from integration.communication import DetectionClient
from integration.messages import ImageMessage
import numpy as np
import time

client = DetectionClient('tcp://localhost:5555', timeout_ms=1000)
image = np.zeros((600, 800, 3), dtype=np.uint8)
msg = ImageMessage(image=image, timestamp=time.time(), frame_id=0)
result = client.detect(msg)
print(f'Detection result: {result}')
client.close()
"
```

## 🐛 Troubleshooting

### 1. "Cannot import carla"

**Inside Dev Container:**
```bash
# Check Python path
echo $PYTHONPATH

# Should include: /opt/carla/PythonAPI/carla

# Rebuild container if needed
# VSCode: Cmd+Shift+P → "Dev Containers: Rebuild Container"
```

### 2. "Connection refused" to CARLA

**Check CARLA is running:**
```bash
# On CARLA machine
ps aux | grep Carla
netstat -tuln | grep 2000
```

**Check network connectivity:**
```bash
# From your machine
ping <CARLA_HOST>
nc -zv <CARLA_HOST> 2000
```

### 3. "Detection timeout" in Distributed Mode

**Check detection server:**
```bash
# Is server running?
ps aux | grep detection_server

# Check logs
python detection_server.py --method cv --port 5555
```

**Check ZMQ connection:**
```bash
# Test with netcat
nc -zv localhost 5555
```

### 4. Web Viewer Not Loading

**Check Flask server:**
```bash
# Is port available?
lsof -i :8080

# Try different port
python main_distributed_v2.py --web-port 8081
```

### 5. Slow Performance on M1 Mac

This is expected due to x86_64 emulation. Optimizations:

```bash
# Reduce camera resolution
# Edit config.yaml:
camera:
  width: 640
  height: 480

# Use low quality on CARLA server
./CarlaUE4.sh -quality-level=Low

# Use web viewer (lighter than OpenCV window)
python main_distributed_v2.py --viewer web
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

The system logs real-time performance metrics:

```
Frame 00150 | FPS: 28.5 | Lanes: LR | Steering: +0.123 | Timeouts: 0
```

- **FPS**: Frames per second
- **Lanes**: Detected lanes (L=left, R=right, -=not detected)
- **Steering**: Control output (-1.0 to +1.0)
- **Timeouts**: Detection timeouts (distributed mode only)

## 🚀 Development Setup (M1 Mac)

### Using Dev Container (Recommended)

1. **Install Prerequisites:**
   - Docker Desktop for Mac
   - VSCode with Dev Containers extension

2. **Enable Rosetta 2 in Docker:**
   - Docker Desktop → Settings → Features in Development
   - ✅ "Use Rosetta for x86/amd64 emulation on Apple Silicon"

3. **Open in Container:**
   ```bash
   cd /path/to/ads_ld
   code .
   # VSCode: Cmd+Shift+P → "Reopen in Container"
   ```

4. **Connect to Remote CARLA:**
   ```bash
   # In VSCode terminal (inside container)
   cd lane_detection
   python main_modular.py --host <LINUX_IP> --port 2000
   ```

See [docs/DEVCONTAINER_SETUP.md](docs/DEVCONTAINER_SETUP.md) for detailed setup.

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/START_HERE.md](docs/START_HERE.md) | 👈 **New to the project? Start here!** |
| [docs/QUICK_START.md](docs/QUICK_START.md) | Quick start guide |
| [docs/ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md) | Why this architecture? |
| [docs/MODULAR_ARCHITECTURE.md](docs/MODULAR_ARCHITECTURE.md) | Detailed architecture explanation |
| [docs/DEVCONTAINER_SETUP.md](docs/DEVCONTAINER_SETUP.md) | Dev container setup for M1 Mac |
| [docs/MACOS_M1_SETUP.md](docs/MACOS_M1_SETUP.md) | M1 Mac specific instructions |
| [docs/DL_QUICKSTART.md](docs/DL_QUICKSTART.md) | Deep learning model setup |
| [lane_detection/DISTRIBUTED_ARCHITECTURE.md](lane_detection/DISTRIBUTED_ARCHITECTURE.md) | Distributed system guide |
| [lane_detection/SYSTEM_OVERVIEW.md](lane_detection/SYSTEM_OVERVIEW.md) | System components overview |
| [lane_detection/VISUALIZATION_GUIDE.md](lane_detection/VISUALIZATION_GUIDE.md) | Visualization options |
| [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) | Recent codebase cleanup details |

## 🎓 For Students

This project demonstrates:

- ✅ **Clean Architecture**: Modular design with single responsibility
- ✅ **Design Patterns**: Factory, Strategy, Observer
- ✅ **Process Communication**: ZMQ for inter-process messaging
- ✅ **Configuration Management**: YAML-based configuration
- ✅ **Multiple Algorithms**: CV and DL approaches
- ✅ **Production Ready**: Error handling, logging, metrics
- ✅ **Docker & DevOps**: Containerized development environment

## 🤝 Contributing

When adding new features:

1. Follow the modular architecture
2. Maintain separation of concerns (CARLA / Detection / Decision)
3. Use the factory pattern for new detectors
4. Add tests in `lane_detection/tests/`
5. Update relevant documentation

## 🔗 Related Projects

This project is designed to work with:

- **CARLA Simulator** (0.9.15): https://carla.org
- **PiRacer** (future integration): Real vehicle deployment

## 📝 License

See [LICENSE](LICENSE) file.

## 🆘 Getting Help

1. **Start with docs**: Check [docs/START_HERE.md](docs/START_HERE.md)
2. **Check troubleshooting**: See sections above
3. **Architecture questions**: See [docs/ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md)
4. **M1 Mac issues**: See [docs/DEVCONTAINER_SETUP.md](docs/DEVCONTAINER_SETUP.md)

## ✅ Quick Reference

### Entry Points

| File | Use Case |
|------|----------|
| `main_modular.py` | Single-process, easy testing |
| `main_distributed_v2.py` | Multi-process, production, web viewer |
| `detection_server.py` | Standalone detection service |

### Command Templates

```bash
# Local development
python main_modular.py --method cv

# Remote CARLA
python main_modular.py --method cv --host 192.168.1.100

# Distributed with web viewer (Docker/Remote friendly)
python detection_server.py --port 5555 &
python main_distributed_v2.py --viewer web --web-port 8080

# Headless mode
python main_modular.py --method cv --no-display

# Custom config
python main_modular.py --config my_config.yaml
```

---

**Ready to start?** 👉 Open [docs/START_HERE.md](docs/START_HERE.md)
