# Lane Detection Module

A modular lane detection system supporting both Computer Vision (CV) and Deep Learning (DL) approaches for lane keeping assist in autonomous driving applications.

## Overview

This module provides a flexible, extensible framework for detecting lane markings in road images. It's designed to work with CARLA simulator and PiRacer hardware platforms as part of a larger lane keeping assist system.

### Key Features

- **Multiple Detection Methods**: Switch between Computer Vision and Deep Learning approaches
- **Plug-and-Play Architecture**: Easy to add new detection algorithms
- **Type-Safe Configuration**: YAML-based config with dataclass validation
- **Distributed Processing**: Run detection on separate machines (e.g., GPU server)
- **Real-time Performance**: Optimized for live video processing
- **Comprehensive Metrics**: Lane position, heading angle, departure warnings

## Architecture

```
detection/
├── core/                  # Core abstractions and utilities
│   ├── config.py         # Configuration management (YAML + dataclasses)
│   ├── factory.py        # Factory pattern for detector creation
│   ├── interfaces.py     # Abstract base classes (LaneDetector, etc.)
│   └── models.py         # Data models (Lane, DetectionResult, etc.)
│
├── method/               # Detection algorithm implementations
│   ├── computer_vision/  # Traditional CV approach (Canny + Hough)
│   └── deep_learning/    # Neural network approach (PyTorch)
│
├── detection_module/     # High-level detection wrapper
│   └── detector.py       # LaneDetectionModule class
│
└── detection.py          # Standalone detection server (ZMQ-based)
```

## Core Components

### 1. Data Models ([core/models.py](core/models.py))

**Lane**
```python
@dataclass
class Lane:
    x1, y1: int  # Start point (bottom)
    x2, y2: int  # End point (top)
    confidence: float = 1.0

    # Computed properties
    slope: float
    length: float
```

**DetectionResult**
```python
@dataclass
class DetectionResult:
    left_lane: Lane | None
    right_lane: Lane | None
    debug_image: np.ndarray | None
    processing_time_ms: float
```

**LaneMetrics**
- Lateral offset (pixels, meters, normalized)
- Heading angle
- Departure status (CENTERED, DRIFT, DEPARTURE)
- Lane width measurements

### 2. Detector Interface ([core/interfaces.py](core/interfaces.py))

All detectors implement the `LaneDetector` abstract base class:

```python
class LaneDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect lanes in RGB image."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return detector name."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current parameters."""
        pass
```

### 3. Configuration System ([core/config.py](core/config.py))

Type-safe configuration using Python dataclasses:

```python
@dataclass
class Config:
    carla: CARLAConfig
    camera: CameraConfig
    cv_detector: CVDetectorConfig
    dl_detector: DLDetectorConfig
    analyzer: AnalyzerConfig
    controller: ControllerConfig
    visualization: VisualizationConfig
    detection_method: str = "cv"  # 'cv' or 'dl'
```

Load from YAML:
```python
config = ConfigManager.load('config.yaml')
```

### 4. Factory Pattern ([core/factory.py](core/factory.py))

Centralized detector creation:

```python
factory = DetectorFactory(config)
detector = factory.create('cv')  # or 'dl'

# Override config parameters
detector = factory.create('cv', canny_low=40, canny_high=120)
```

## Detection Methods

### Computer Vision ([method/computer_vision/](method/computer_vision/))

Traditional image processing pipeline:
1. **Grayscale conversion**
2. **Gaussian blur** - Noise reduction
3. **Canny edge detection** - Edge extraction
4. **Region of Interest (ROI)** - Focus on road area
5. **Hough line transform** - Line detection
6. **Line filtering & grouping** - Separate left/right lanes
7. **Temporal smoothing** - Stabilize across frames

**Parameters:**
- `canny_low`, `canny_high`: Edge detection thresholds
- `hough_threshold`: Line voting threshold
- `hough_min_line_len`: Minimum line length
- `smoothing_factor`: Temporal smoothing weight

### Deep Learning ([method/deep_learning/](method/deep_learning/))

Neural network-based approach using PyTorch:
- Segmentation network architecture
- Pre-trained or custom models
- GPU acceleration support
- Configurable input sizes and thresholds

**Parameters:**
- `model_type`: 'pretrained', 'simple', or 'full'
- `input_size`: Input image dimensions (e.g., 256x256)
- `threshold`: Detection confidence threshold
- `device`: 'cpu', 'cuda', or 'auto'

## Usage

### Basic Usage

```python
from detection.core.config import ConfigManager
from detection.core.factory import DetectorFactory

# Load configuration
config = ConfigManager.load('config.yaml')

# Create detector
factory = DetectorFactory(config)
detector = factory.create('cv')  # or 'dl'

# Process image
import cv2
image = cv2.imread('road.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = detector.detect(image_rgb)

# Access results
if result.has_both_lanes:
    print(f"Left lane: ({result.left_lane.x1}, {result.left_lane.y1}) -> "
          f"({result.left_lane.x2}, {result.left_lane.y2})")
    print(f"Processing time: {result.processing_time_ms:.2f} ms")
```

### Standalone Detection Server

Run detection as a separate service (useful for distributed systems):

```bash
# Start CV detector server
python detection/detection.py --method cv --port 5556

# Start DL detector server with GPU
python detection/detection.py --method dl --port 5556 --gpu 0
```

Server features:
- ZMQ-based communication
- Can run on different machine
- Multiple clients supported
- Independent lifecycle from main application

### Integration with Lane Detection Module

```python
from detection.detection_module.detector import LaneDetectionModule
from simulation.integration.messages import ImageMessage

# Initialize module
module = LaneDetectionModule(config, method='cv')

# Process image message
image_msg = ImageMessage(
    image=image_array,
    frame_id=123,
    timestamp=time.time()
)

detection_msg = module.process_image(image_msg)
```

## Configuration Example

```yaml
# Detection method selection
detection_method: cv  # 'cv' or 'dl'

# Computer Vision parameters
lane_detection:
  cv_params:
    canny_low: 50
    canny_high: 150
    hough_threshold: 50
    hough_min_line_len: 40
    hough_max_line_gap: 100
    smoothing_factor: 0.7
    min_slope: 0.5

    # ROI (fraction of image)
    roi_bottom_left_x: 0.1
    roi_top_left_x: 0.45
    roi_top_right_x: 0.55
    roi_bottom_right_x: 0.9
    roi_top_y: 0.6

# Deep Learning parameters
dl_detector:
  model_type: pretrained
  input_size: [256, 256]
  threshold: 0.5
  device: auto  # 'cpu', 'cuda', or 'auto'
```

## Extending the Module

### Adding a New Detection Method

1. **Create detector class** implementing `LaneDetector`:

```python
from detection.core.interfaces import LaneDetector
from detection.core.models import DetectionResult

class MyCustomDetector(LaneDetector):
    def detect(self, image: np.ndarray) -> DetectionResult:
        # Your detection logic
        return DetectionResult(...)

    def get_name(self) -> str:
        return "My Custom Detector"

    def get_parameters(self) -> dict:
        return {"param1": value1}
```

2. **Register in factory** ([core/factory.py](core/factory.py)):

```python
def create(self, detector_type: str | None = None) -> LaneDetector:
    if detector_type == "custom":
        return self._create_custom_detector()
    # ... existing code
```

3. **Add configuration** if needed ([core/config.py](core/config.py))

## Performance Characteristics

| Method | Processing Time | Accuracy | Hardware Req |
|--------|----------------|----------|--------------|
| CV     | ~5-10 ms       | Good     | CPU only     |
| DL     | ~15-30 ms (GPU)| Excellent| GPU recommended |

## Dependencies

- **Core**: numpy, opencv-python, PyYAML
- **Deep Learning**: torch, torchvision (optional)
- **Communication**: pyzmq (for distributed mode)

## Related Modules

- **simulation/**: CARLA integration and vehicle control
- **decision/**: Lane analysis and control algorithms

## Design Principles

1. **Separation of Concerns**: Detection logic isolated from control/visualization
2. **Interface Segregation**: Clean abstractions via ABC
3. **Dependency Injection**: Configuration passed at construction
4. **Factory Pattern**: Centralized object creation
5. **Type Safety**: Dataclasses with type hints throughout
6. **Testability**: Modular design enables unit testing

## Version

Current version: 0.1.0 (see [__init__.py](__init__.py))
