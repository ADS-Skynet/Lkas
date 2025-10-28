# Lane Detection Module

This module implements Lane Keeping Assist System (LKAS) for CARLA simulation and PiRacer hardware.

## Features

- **Dual Detection Methods**:
  - Traditional Computer Vision (OpenCV-based Hough Transform)
  - Deep Learning (CNN-based segmentation)

- **Real-time Analysis**:
  - Lane position calculation
  - Lateral offset measurement
  - Heading angle estimation
  - Lane departure detection

- **Visual Feedback**:
  - HUD with metrics display
  - Lane visualization overlay
  - Departure warnings with visual alerts
  - Steering correction indicator

## Project Structure

```
lane_detection/
├── carla_interface.py          # CARLA connection and camera management
├── main.py                      # Main integration script
├── traditional/
│   ├── __init__.py
│   └── cv_lane_detector.py     # OpenCV-based lane detection
├── deep_learning/
│   ├── __init__.py
│   └── lane_net.py             # CNN models for lane detection
├── utils/
│   ├── __init__.py
│   ├── lane_analyzer.py        # Lane position analysis
│   └── visualizer.py           # Visualization tools
├── tests/                       # Unit tests
└── data/                        # Model weights and datasets
```

## Installation

### 1. Install Python Dependencies

```bash
cd /Users/donghyun/All/seame/ads_ld
pip install -r requirements.txt
```

### 2. Install CARLA

Download and install CARLA simulator from [https://carla.org/](https://carla.org/)

For CARLA 0.9.13+, install the Python package:
```bash
pip install carla==0.9.13
```

## Usage

### Running with CARLA

1. **Start CARLA Server**:
```bash
cd /path/to/CARLA
./CarlaUE4.sh  # Linux/Mac
# or CarlaUE4.exe on Windows
```

2. **Run Lane Detection** (Traditional CV method):
```bash
cd lane_detection
python main.py --method cv
```

3. **Run with Deep Learning** (requires trained model):
```bash
python main.py --method dl --model path/to/model.pth
```

### Command Line Options

```bash
python main.py --help

Options:
  --method {cv,dl}        Lane detection method (default: cv)
  --width WIDTH           Camera image width (default: 800)
  --height HEIGHT         Camera image height (default: 600)
  --host HOST            CARLA server host (default: localhost)
  --port PORT            CARLA server port (default: 2000)
  --vehicle VEHICLE      Vehicle type to spawn (default: vehicle.tesla.model3)
  --model MODEL          Path to pretrained DL model
  --no-display           Disable visualization display
  --save-video PATH      Path to save output video
```

### Examples

**Save output video**:
```bash
python main.py --method cv --save-video output.mp4
```

**Custom vehicle and resolution**:
```bash
python main.py --vehicle vehicle.audi.a2 --width 1280 --height 720
```

**Headless mode** (no display):
```bash
python main.py --no-display --save-video output.mp4
```

## Module Usage

### Using Individual Components

```python
from carla_interface import CARLAInterface
from traditional.cv_lane_detector import CVLaneDetector
from utils.lane_analyzer import LaneAnalyzer
from utils.visualizer import LKASVisualizer

# Setup CARLA
carla = CARLAInterface()
carla.connect()
carla.spawn_vehicle()
carla.setup_camera()

# Setup detector
detector = CVLaneDetector()
analyzer = LaneAnalyzer(image_width=800, image_height=600)
visualizer = LKASVisualizer()

# Process frames
while True:
    image = carla.get_latest_image()
    if image is not None:
        # Detect lanes
        left_lane, right_lane, debug_img = detector.detect(image)

        # Analyze
        metrics = analyzer.get_metrics(left_lane, right_lane)
        steering = analyzer.get_steering_correction(left_lane, right_lane)

        # Visualize
        vis_img = visualizer.draw_lanes(image, left_lane, right_lane)
        vis_img = visualizer.draw_hud(vis_img, metrics, steering_value=steering)
```

## Algorithm Details

### Traditional CV Method

1. **Preprocessing**: RGB → Grayscale → Gaussian Blur
2. **Edge Detection**: Canny edge detection
3. **ROI Selection**: Focus on road area ahead
4. **Line Detection**: Hough transform for line segments
5. **Lane Separation**: Separate left/right based on slope and position
6. **Lane Fitting**: Linear regression to average line segments
7. **Temporal Smoothing**: Exponential moving average for stability

### Deep Learning Method

1. **Architecture**: UNet-style CNN for lane segmentation
2. **Input**: RGB image (256x256)
3. **Output**: Binary lane mask
4. **Post-processing**: Threshold and resize to original size

## Lane Departure Detection

The system uses thresholds based on lateral offset:

- **Centered**: Offset < 15% of lane width
- **Drift Warning**: 15% ≤ Offset < 35%
- **Departure Warning**: Offset ≥ 35%

## Steering Correction

PD controller for steering suggestions:
```
correction = -(Kp × offset + Kd × heading_angle)
```

Default gains:
- Kp = 0.5 (proportional to lateral offset)
- Kd = 0.1 (proportional to heading angle)

## Configuration

### Tuning CV Parameters

Edit parameters in `traditional/cv_lane_detector.py`:

```python
detector = CVLaneDetector(
    canny_low=50,           # Lower Canny threshold
    canny_high=150,         # Upper Canny threshold
    hough_threshold=50,     # Minimum votes for line
    hough_min_line_len=40,  # Minimum line length
    hough_max_line_gap=100  # Maximum gap between segments
)
```

### Tuning Analysis Parameters

Edit parameters in `utils/lane_analyzer.py`:

```python
analyzer = LaneAnalyzer(
    drift_threshold=0.15,      # Drift warning threshold
    departure_threshold=0.35,  # Departure warning threshold
    lane_width_meters=3.7      # Standard lane width
)
```

## Training Deep Learning Model

(To be implemented)

1. Collect training data from CARLA
2. Annotate lane markings
3. Train model using provided architecture
4. Save model weights to `data/models/`

## Testing

Run unit tests:
```bash
cd lane_detection
python -m pytest tests/
```

## Troubleshooting

### CARLA Connection Issues
- Ensure CARLA server is running
- Check host/port settings
- Verify firewall settings

### Poor Lane Detection
- Adjust Canny/Hough parameters for CV method
- Ensure proper camera positioning
- Check ROI settings for your vehicle height

### Performance Issues
- Reduce image resolution
- Use SimpleLaneNet instead of LaneNet
- Disable visualization display

## Next Steps for Real Hardware (PiRacer)

1. Replace `CARLAInterface` with PiRacer camera interface
2. Integrate with PiRacer motor control
3. Calibrate steering correction gains
4. Add safety constraints for real-world operation

## References

- [OpenCV Documentation](https://opencv.org/)
- [CARLA Documentation](https://carla.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

See LICENSE file in project root.
