# Detection Module

**Lane detection for autonomous driving using Computer Vision and Deep Learning.**

## Overview

The Detection module provides a flexible, extensible framework for detecting lane markings in road images. It supports both classical Computer Vision (Canny + Hough) and Deep Learning (PyTorch) approaches, with a clean plugin architecture for adding new methods.

### Key Features

- **Multiple Detection Methods**: Switch between CV and DL approaches
- **Plug-and-Play Architecture**: Easy to add new detection algorithms via factory pattern
- **Type-Safe Design**: Dataclass-based configuration and message protocols
- **Distributed Processing**: Run detection as separate process/service
- **Real-time Performance**: Optimized for 30+ FPS live video processing
- **Shared Memory IPC**: Ultra-low latency communication (<0.01ms)
- **ZMQ Parameter Updates**: Live tuning without restart

## Architecture

```
detection/
├── core/                  # Core abstractions and utilities
│   ├── config.py         # Configuration management (YAML + dataclasses)
│   ├── factory.py        # Factory pattern for detector creation
│   ├── interfaces.py     # Abstract base classes (LaneDetector, etc.)
│   └── models.py         # Data models (Lane, DetectionResult, etc.)
│
├── integration/          # Inter-process communication infrastructure
│   ├── messages.py       # Message definitions for IPC
│   ├── shared_memory_detection.py  # Image/detection shared memory channels
│   └── shared_memory_control.py    # Control command shared memory channel
│
├── method/               # Detection algorithm implementations
│   ├── computer_vision/  # Traditional CV approach (Canny + Hough)
│   └── deep_learning/    # Neural network approach (PyTorch)
│
├── client.py             # DetectionClient for reading detections
├── detector.py           # LaneDetection core wrapper class
├── server.py             # DetectionServer with shared memory I/O
└── run.py                # CLI entrypoint (lane-detection command)
```

## Core Components

### 1. Data Models

**Lane** (from `common.types.models`)
```python
@dataclass
class Lane:
    x1: int  # Start point X (bottom of image)
    y1: int  # Start point Y (bottom of image)
    x2: int  # End point X (top of region)
    y2: int  # End point Y (top of region)
    confidence: float = 1.0

    # Computed properties
    slope: float
    length: float
```

**DetectionResult** (from `common.types.models`)
```python
@dataclass
class DetectionResult:
    left_lane: Lane | None
    right_lane: Lane | None
    debug_image: np.ndarray | None
    processing_time_ms: float
    has_both_lanes: bool
```

**DetectionMessage** (from `lkas.integration.shared_memory.messages`)
```python
@dataclass
class DetectionMessage:
    left_lane: LaneMessage | None
    right_lane: LaneMessage | None
    frame_id: int
    timestamp: float
    processing_time_ms: float
```

### 2. Detector Interface ([core/interfaces.py](core/interfaces.py))

All detectors implement the `LaneDetector` abstract base class:

```python
from abc import ABC, abstractmethod
from common.types.models import DetectionResult

class LaneDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect lanes in RGB image.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            DetectionResult with left_lane, right_lane, and processing time
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return detector name (e.g., 'CV Lane Detector')."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current detector parameters as dict."""
        pass

    @abstractmethod
    def update_parameter(self, name: str, value: float) -> bool:
        """
        Update a detector parameter in real-time.

        Args:
            name: Parameter name (e.g., 'canny_low', 'hough_threshold')
            value: New parameter value

        Returns:
            True if parameter was updated, False otherwise
        """
        pass
```

### 3. Configuration Management

Configuration is loaded from the project-wide `config.yaml` file:

```python
from common.config import ConfigManager

# Load configuration
config = ConfigManager.load('config.yaml')

# Access detection config
detection_config = config.detection
cv_params = detection_config.cv_params
```

Configuration is validated at load time using dataclasses with type hints.

### 4. Factory Pattern ([core/factory.py](core/factory.py))

Centralized detector creation with method selection:

```python
from lkas.detection.core.factory import DetectorFactory

# Create factory
factory = DetectorFactory(config)

# Create detector by method
detector = factory.create('cv')  # Computer Vision
# or
detector = factory.create('dl')  # Deep Learning

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
from common.config import ConfigManager
from lkas.detection.core.factory import DetectorFactory

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
lane-detection --method cv

# Start DL detector server
lane-detection --method dl

# Or use the module directly
python -m lkas.detection.run --method cv
```

Server features:
- **Shared memory-based communication** for high performance
- Independent process lifecycle
- Bidirectional communication (control commands + detection results)
- Low latency compared to network-based IPC

### Detection Client Integration

```python
from lkas.detection import DetectionClient
from common.config import ConfigManager

# Initialize client
config = ConfigManager.load('config.yaml')
client = DetectionClient(config)

# Write image for detection
client.write_image(image_array, frame_id=123)

# Read detection result
detection_msg = client.read_detection()
if detection_msg:
    print(f"Left lane: {detection_msg.left_lane}")
    print(f"Right lane: {detection_msg.right_lane}")
```

### Direct Detection Usage

```python
from lkas.detection import LaneDetection
from common.config import ConfigManager

# Initialize detector
config = ConfigManager.load('config.yaml')
detector = LaneDetection(config, method='cv')

# Process image directly
result = detector.detect(image_array)
if result.has_both_lanes:
    print(f"Processing time: {result.processing_time_ms:.2f} ms")
```

## Configuration Example

```yaml
detection:
  method: cv  # 'cv' or 'dl'

  # Computer Vision parameters
  cv:
    # Edge detection
    canny_low: 50
    canny_high: 150

    # Line detection
    hough_threshold: 50
    hough_min_line_len: 40
    hough_max_line_gap: 100

    # Processing
    smoothing_factor: 0.7
    min_slope: 0.5

  # Region of Interest (fraction of image)
  roi:
    top_trim: 0.55     # Ignore top 55%
    bottom_trim: 0.1   # Ignore bottom 10%
    side_trim: 0.1     # Ignore sides 10%

  # Deep Learning parameters
  dl:
    model_path: "models/lane_detector.pth"
    model_type: pretrained
    input_size: [256, 256]
    threshold: 0.5
    device: auto  # 'cpu', 'cuda', or 'auto'
```

## Extending the Module

### Adding a New Detection Method

1. **Create detector class** implementing `LaneDetector`:

```python
from lkas.detection.core.interfaces import LaneDetector
from common.types.models import DetectionResult

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
- **Communication**: multiprocessing.shared_memory (Python 3.8+)

## Related Modules

- **decision/**: Steering control and lane analysis
- **integration/**: Shared memory and ZMQ communication
- **simulation/**: CARLA integration and vehicle platform

## Design Principles

1. **Separation of Concerns**: Detection logic isolated from control/visualization
2. **Interface Segregation**: Clean abstractions via ABC
3. **Dependency Injection**: Configuration passed at construction
4. **Factory Pattern**: Centralized object creation
5. **Type Safety**: Dataclasses with type hints throughout
6. **Testability**: Modular design enables unit testing

## Troubleshooting

### Poor lane detection quality
- **Adjust Canny thresholds:** Lower for more edges, higher for fewer
  - `canny_low`: 30-70 (default: 50)
  - `canny_high`: 100-200 (default: 150)
- **Tune Hough parameters:**
  - `hough_threshold`: Lower to detect more lines (default: 50)
  - `hough_min_line_len`: Minimum line length in pixels (default: 40)
- **Modify ROI:** Focus on lane area, exclude irrelevant regions
- **Increase smoothing:** Higher `smoothing_factor` for stability (0.0-1.0)

### Detection too slow
- **Use CV method:** 3-5x faster than DL (5-15ms vs 15-30ms)
- **Reduce image size:** Smaller input = faster processing
- **Optimize ROI:** Smaller region = less processing
- **For DL:** Use GPU, TensorRT optimization, or lighter model

### Flickering/unstable lanes
- **Increase smoothing_factor:** Smooth across frames (0.7-0.9)
- **Check lighting:** Poor lighting affects CV methods
- **Verify lane visibility:** Detection needs clear lane markings

### No lanes detected
- **Check camera feed:** Ensure image is valid RGB
- **Verify ROI settings:** Make sure lane area is included
- **Lower thresholds:** More lenient detection parameters
- **Check lane markers:** Detection requires visible lane lines

## Version

Current version: 0.1.0 (see [\_\_init\_\_.py](__init__.py))
