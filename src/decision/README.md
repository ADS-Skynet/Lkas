# Decision Module

**Steering control and lane analysis for lane keeping assist systems.**

## Overview

The Decision module processes lane detection results and generates steering, throttle, and brake commands to keep the vehicle centered in its lane. It supports two detection pipelines:

- **CV pipeline**: Canny + Hough lane lines → LaneAnalyzer → PD/PID controller
- **DL pipeline**: BiSeNet V2 segmentation mask → SegmentationLaneParser → Pure Pursuit controller

### Key Features

- **Multi-Controller Support**: PD, PID, Pure Pursuit, and MPC control strategies
- **DL Segmentation Support**: Polynomial lane fitting from binary masks
- **Pure Pursuit Path Tracking**: Lookahead-based steering that anticipates curves
- **Temporal Smoothing**: EMA on polynomial coefficients for stable steering
- **Adaptive Throttle**: Automatic speed reduction during sharp turns
- **Lane Departure Detection**: Multi-level warnings (CENTERED, DRIFT, DEPARTURE)
- **Debug Overlay**: Polynomial lane boundaries and center path visualized in viewer
- **Real-time Parameter Tuning**: Live adjustment of controller gains via viewer

## Architecture

```
decision/
├── core/                      # Core abstractions
│   ├── interfaces.py         # SteeringController ABC
│   ├── factory.py            # ControllerFactory for instantiation
│   └── __init__.py
│
├── method/                    # Control algorithm implementations
│   ├── pd_controller.py      # Proportional-Derivative controller
│   ├── pid_controller.py     # Proportional-Integral-Derivative controller
│   ├── pure_pursuit_controller.py  # Pure Pursuit path tracking
│   ├── mpc_controller.py     # Model Predictive Control (future)
│   └── __init__.py
│
├── segmentation_lane_parser.py  # DL mask → polynomial lane fitting
├── controller.py              # DecisionController (main orchestrator)
├── lane_analyzer.py           # LaneAnalyzer for CV position metrics
├── client.py                  # DecisionClient for reading control commands
├── server.py                  # DecisionServer with shared memory I/O
└── run.py                     # CLI entrypoint (decision-server command)
```

## Core Components

### 1. DecisionController ([controller.py](controller.py))

Main orchestrator that combines lane analysis with steering control.

```python
from lkas.decision import DecisionController
from lkas.integration.shared_memory.messages import DetectionMessage

# Initialize controller
controller = DecisionController(
    image_width=800,
    image_height=600,
    kp=0.5,              # Proportional gain
    kd=0.1,              # Derivative gain
    controller_method="pd",  # or "pid"
    throttle_policy={
        "base": 0.15,
        "min": 0.05,
        "steer_threshold": 0.15,
        "steer_max": 0.70
    }
)

# Process detection and get control command
detection = DetectionMessage(...)  # From detection module
control = controller.process_detection(detection)

print(f"Steering: {control.steering:.3f}")
print(f"Throttle: {control.throttle:.3f}")
print(f"Lateral offset: {control.lateral_offset:.3f}")
print(f"Heading angle: {control.heading_angle:.2f}°")
```

### 2. LaneAnalyzer ([lane_analyzer.py](lane_analyzer.py))

Analyzes lane geometry and computes vehicle position metrics.

```python
from lkas.decision import LaneAnalyzer

# Initialize analyzer
analyzer = LaneAnalyzer(
    image_width=800,
    image_height=600,
    drift_threshold=0.15,      # Drift warning at 15% of lane width
    departure_threshold=0.35,  # Departure warning at 35%
    lane_width_meters=3.7      # Standard US highway lane
)

# Get metrics from lane lines
left_lane = (200, 600, 350, 360)   # (x1, y1, x2, y2)
right_lane = (600, 600, 450, 360)

metrics = analyzer.get_metrics(left_lane, right_lane)

print(f"Lateral offset: {metrics.lateral_offset_meters:.2f} m")
print(f"Heading angle: {metrics.heading_angle_deg:.2f}°")
print(f"Lane width: {metrics.lane_width_pixels} px")
print(f"Departure status: {metrics.departure_status}")
```

### 3. SteeringController Interface ([core/interfaces.py](core/interfaces.py))

Abstract base class for all control implementations:

```python
from abc import ABC, abstractmethod
from common.types.models import LaneMetrics

class SteeringController(ABC):
    @abstractmethod
    def compute_steering(self, metrics: LaneMetrics) -> float | None:
        """Compute steering command from lane metrics."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return controller name."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current parameters."""
        pass

    @abstractmethod
    def update_parameter(self, name: str, value: float) -> bool:
        """Update a parameter."""
        pass
```

## Control Algorithms

### PD Controller ([method/pd_controller.py](method/pd_controller.py))

**Proportional-Derivative control** - Fast, responsive, no integral windup.

```python
steering = -(Kp × lateral_offset + Kd × heading_angle)
```

**Parameters:**
- `kp`: Proportional gain (0.0 - 2.0) - Responsiveness to lateral offset
- `kd`: Derivative gain (0.0 - 1.0) - Damping based on heading angle

**Use when:** You need fast response without state accumulation.

```python
from lkas.decision.method import PDController

controller = PDController(kp=0.5, kd=0.1)
steering = controller.compute_steering(metrics)
```

### PID Controller ([method/pid_controller.py](method/pid_controller.py))

**Proportional-Integral-Derivative control** - Eliminates steady-state error.

```python
steering = -(Kp × error + Ki × ∫error·dt + Kd × d(error)/dt)
```

**Parameters:**
- `kp`: Proportional gain (0.0 - 2.0)
- `ki`: Integral gain (0.0 - 0.5) - Accumulates error over time
- `kd`: Derivative gain (0.0 - 1.0)

**Use when:** You need zero steady-state error (curved roads, crosswinds).

```python
from lkas.decision.method import PIDController

controller = PIDController(kp=0.5, ki=0.01, kd=0.1)
steering = controller.compute_steering(metrics)

# Reset integral accumulation
controller.reset_state()
```

### MPC Controller ([method/mpc_controller.py](method/mpc_controller.py))

**Model Predictive Control** - Optimal control with constraints (future implementation).

**Use when:** You need optimal trajectories with hard constraints.

---

## DL Pipeline: Segmentation + Pure Pursuit

The DL pipeline processes binary segmentation masks from BiSeNet V2 and uses
geometric path tracking for smoother curve handling than PD/PID.

```
Camera Frame
     │
     ▼
┌─────────────────┐
│  BiSeNet V2     │  Detection Module
│  Segmentation   │  → binary mask (H,W) uint8, 0=background, 1=lane
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Segmentation   │  1. Row-scan mask → find lane pixel clusters
│  Lane Parser    │  2. Adjacent-pair selection → left/right boundaries
│                 │  3. Polynomial fit x = f(y) → left, right, center curves
│                 │  4. Temporal smoothing (EMA on polynomial coefficients)
│                 │  5. Compute metrics: offset, heading, lane width
└────────┬────────┘
         │  LaneMetrics + center polynomial
         ▼
┌─────────────────┐
│  Pure Pursuit   │  1. Evaluate center poly at lookahead point
│  Controller     │  2. Compute lateral error at lookahead
│                 │  3. Compute heading from poly tangent at lookahead
│                 │  4. Control law → steering command
└────────┬────────┘
         │  steering [-1, 1]
         ▼
    Vehicle Control
```

### SegmentationLaneParser ([segmentation_lane_parser.py](segmentation_lane_parser.py))

Converts a binary segmentation mask into polynomial lane representations.

**Step 1 — Row scanning:** For each row in the ROI (bottom 60% of image), find
all non-zero (lane) pixels and group them into contiguous clusters separated by
gaps > 3% of image width.

**Step 2 — Adjacent-pair selection:** When 2+ clusters are found in a row,
evaluate all consecutive pairs and pick the pair whose midpoint is closest to
a reference center (previous frame's center path, or image center on first frame).
This correctly identifies the vehicle's current lane even when offset — both
lane markings can be on the same side of the image center and it still works.

**Step 3 — Outlier filtering:** IQR-based filtering removes boundary points
with x-coordinates outside 1.5x the interquartile range.

**Step 4 — Polynomial fitting:** `np.polyfit` fits a 2nd-degree polynomial
`x = ay² + by + c` to each boundary's points. The center path is the average
of left and right polynomials (or estimated from one side + assumed half lane
width if only one boundary is visible).

**Step 5 — Temporal smoothing:** Exponential moving average on polynomial
coefficients between frames to reduce jitter (see [Smoothing Factor](#smoothing-factor)).

**Step 6 — Metrics:** Lateral offset, heading angle, lane width, and departure
status are computed from the polynomials.

### Pure Pursuit Controller ([method/pure_pursuit_controller.py](method/pure_pursuit_controller.py))

Geometric path tracking algorithm. Instead of reacting to the current lateral
offset (like PD/PID), it evaluates the offset at a **lookahead point** ahead
on the path. This naturally anticipates curves.

**Control law:**

```
y_lookahead = image_height × (1 - lookahead_ratio)
x_path      = center_poly evaluated at y_lookahead
error       = (vehicle_center - x_path) / (image_width / 2)      normalized [-1, 1]
heading     = arctan(center_poly derivative at y_lookahead) / 30°  clipped [-1, 1]

steering = -(gain × error / lookahead_ratio + heading_gain × heading)
```

**Sign conventions:**
- `error > 0` → vehicle is right of path → steer left (negative)
- `error < 0` → vehicle is left of path → steer right (positive)
- Heading is computed at the lookahead point (not at the vehicle position)
  to avoid unreliable polynomial slopes at the bottom edge of the image

**Parameters:**

| Parameter | Config key | Default | Description |
|-----------|-----------|---------|-------------|
| Gain | `kp` | `0.5` | Scales lateral error. Higher = more aggressive correction. |
| Heading gain | `kd` | `0.1` | Damping term from road direction. |
| Lookahead ratio | `lookahead_ratio` | `0.5` | Fraction of image height to look ahead. Higher = smoother but less reactive. |

**Effective gain = kp / lookahead_ratio.** With defaults: 0.5 / 0.5 = 1.0.

**Tuning guide:**
- Vehicle oscillates/wiggles → decrease `kp` or increase `lookahead_ratio`
- Vehicle cuts corners on curves → decrease `lookahead_ratio`
- Vehicle reacts too slowly → increase `kp`
- Vehicle overshoots corrections → decrease `kd`

All three parameters are adjustable in real-time via the viewer sliders.

### Smoothing Factor

The `smoothing_factor` controls temporal smoothing of polynomial coefficients
between consecutive frames using an exponential moving average (EMA).

**Formula:**

```
smoothed_poly = α × previous_poly + (1 - α) × current_poly

where α = smoothing_factor
```

**How different values behave:**

| Value | Blend | Effect |
|-------|-------|--------|
| `0.0` | 100% current | No smoothing. Raw polynomial each frame. Maximum responsiveness, jittery steering. |
| `0.3` | 70% current, 30% previous | Light smoothing. Mostly follows current frame with slight memory. |
| **`0.5`** | **50/50** | **Default. Balanced jitter reduction without noticeable lag.** |
| `0.7` | 30% current, 70% previous | Heavy smoothing. Very stable but may feel sluggish on sharp curves. |
| `0.9` | 10% current, 90% previous | Nearly ignores current frame. Too laggy for real-time control. |

**Why it matters:** The segmentation mask changes slightly every frame due to
camera noise, lighting, and small differences in the neural network output.
Without smoothing, the polynomial coefficients jump frame-to-frame, causing the
steering to wiggle even on a straight road. The EMA averages this noise out.

**Trade-off:** Higher α = smoother steering but slower reaction to actual road
changes (entering a curve). The default of 0.5 balances responsiveness and stability.

**Warmup:** During the first 10 frames, smoothing is disabled — raw polynomials
are used directly so the system locks onto the lane quickly at startup.

### Debug Overlay

When the DL pipeline is active, the viewer draws polynomial debug lines:

| Color | Line | Meaning |
|-------|------|---------|
| Blue | Left boundary | Inner edge of left lane marking |
| Red | Right boundary | Inner edge of right lane marking |
| Yellow (thick) | Center path | Path the controller follows (average of left/right) |
| Yellow circle | Lookahead point | Where Pure Pursuit evaluates lateral error |
| White (thin) | Vehicle center | Image center reference line |

The polynomial coefficients (9 doubles: left/right/center × a,b,c) are
transmitted through the shared memory control channel and broadcast via ZMQ.

---

## Lane Metrics

The `LaneMetrics` dataclass contains all computed lane analysis:

```python
@dataclass
class LaneMetrics:
    # Position metrics
    vehicle_center_x: float          # Vehicle center in image (pixels)
    lane_center_x: float | None      # Lane center in image (pixels)
    lane_width_pixels: float | None  # Lane width (pixels)

    # Lateral offset
    lateral_offset_pixels: float | None      # Offset in pixels
    lateral_offset_meters: float | None      # Offset in meters
    lateral_offset_normalized: float | None  # Offset / lane_width

    # Heading
    heading_angle_deg: float | None  # Angle relative to lane (degrees)

    # Status
    departure_status: LaneDepartureStatus  # CENTERED, DRIFT, DEPARTURE
    has_left_lane: bool
    has_right_lane: bool
    has_both_lanes: bool
```

### Departure Status Levels

```python
class LaneDepartureStatus(Enum):
    CENTERED = "centered"              # Within drift threshold
    LEFT_DRIFT = "left_drift"          # Between drift and departure threshold
    RIGHT_DRIFT = "right_drift"
    LEFT_DEPARTURE = "left_departure"  # Beyond departure threshold
    RIGHT_DEPARTURE = "right_departure"
    NO_LANES = "no_lanes"             # No lane detection
```

## Adaptive Throttle

The adaptive throttle system automatically reduces speed during turns to maintain stable control:

```python
# Throttle decreases as steering magnitude increases
if |steering| <= steer_threshold:
    throttle = throttle_base
else:
    # Linear interpolation between base and min
    t = (|steering| - steer_threshold) / (steer_max - steer_threshold)
    throttle = throttle_base - t × (throttle_base - throttle_min)
```

**Configuration:**
```python
throttle_policy = {
    "base": 0.15,          # Throttle on straight roads
    "min": 0.05,           # Minimum throttle in sharp turns
    "steer_threshold": 0.15,  # Start reducing at |steering| > 0.15
    "steer_max": 0.70      # Full reduction at |steering| >= 0.70
}
```

**Disable adaptive throttle:**
```python
controller = DecisionController(
    ...,
    throttle_policy=None  # Use default constant throttle
)
controller.set_throttle_brake(throttle=0.3, brake=0.0)
```

## Usage

### Standalone Decision Server

Run decision processing as a separate service:

```bash
# Start decision server with PD controller
decision-server --method pd --kp 0.5 --kd 0.1

# Start with PID controller
decision-server --method pid --kp 0.5 --ki 0.01 --kd 0.1

# Or use module directly
python -m lkas.decision.run --method pd
```

The server:
- Reads lane detections from shared memory
- Computes control commands
- Writes control commands to shared memory
- Accepts real-time parameter updates via ZMQ

### Decision Client Integration

```python
from lkas.decision import DecisionClient

# Initialize client
client = DecisionClient()

# Read control command
control_msg = client.read_control()
if control_msg:
    print(f"Steering: {control_msg.steering:.3f}")
    print(f"Throttle: {control_msg.throttle:.3f}")
    print(f"Lateral offset: {control_msg.lateral_offset_meters:.2f} m")
```

### Direct Usage (In-Process)

```python
from lkas.decision import DecisionController
from lkas.detection import DetectionClient

# Initialize
detection_client = DetectionClient()
controller = DecisionController(
    image_width=800,
    image_height=600,
    kp=0.5,
    kd=0.1,
    controller_method="pd"
)

# Processing loop
while running:
    # Get detection from detection module
    detection = detection_client.read_detection()

    if detection:
        # Compute control command
        control = controller.process_detection(detection)

        # Apply to vehicle
        vehicle.apply_control(
            steering=control.steering,
            throttle=control.throttle,
            brake=control.brake
        )
```

## Real-time Parameter Tuning

Update controller parameters on-the-fly via ZMQ:

```python
# From viewer or control interface
parameter_update = {
    "category": "decision",
    "name": "kp",
    "value": 0.6
}

# Controller receives and applies update
controller.update_parameter("kp", 0.6)
```

**Tunable parameters:**
- `kp`, `ki`, `kd`: Controller gains
- `throttle_base`, `throttle_min`: Adaptive throttle bounds
- `steer_threshold`, `steer_max`: Throttle reduction thresholds
- `reset`: Special parameter to reset PID state

## Configuration Example

```yaml
decision:
  method: pd  # or 'pid'

  # PD/PID gains
  kp: 0.5
  ki: 0.01  # PID only
  kd: 0.1

  # Adaptive throttle
  throttle_base: 0.15
  throttle_min: 0.05
  steer_threshold: 0.15
  steer_max: 0.70

  # Lane analyzer
  drift_threshold: 0.15       # 15% of lane width
  departure_threshold: 0.35   # 35% of lane width
  lane_width_meters: 3.7      # US highway standard
```

## Extending the Module

### Adding a New Controller

1. **Create controller class** implementing `SteeringController`:

```python
from lkas.decision.core.interfaces import SteeringController
from common.types.models import LaneMetrics

class MyController(SteeringController):
    def __init__(self, custom_param: float = 1.0):
        self.custom_param = custom_param

    def compute_steering(self, metrics: LaneMetrics) -> float | None:
        if not metrics.has_both_lanes:
            return None
        # Your control logic
        return steering

    def get_name(self) -> str:
        return "My Custom Controller"

    def get_parameters(self) -> dict:
        return {"custom_param": self.custom_param}

    def update_parameter(self, name: str, value: float) -> bool:
        if name == "custom_param":
            self.custom_param = value
            return True
        return False
```

2. **Register in factory** ([core/factory.py](core/factory.py)):

```python
def create(self, controller_type: str, **kwargs) -> SteeringController:
    if controller_type == "my_controller":
        return MyController(**kwargs)
    # ... existing code
```

3. **Use it:**

```bash
decision-server --method my_controller --custom-param 1.5
```

## Performance

### Typical Latency

- **Lane analysis:** <1ms
- **PD control:** <1ms
- **PID control:** <1ms
- **Total decision latency:** <2ms

### Optimization Tips

1. **Use PD for lowest latency** - No integral state to maintain
2. **Reset PID state** when vehicle deviates significantly
3. **Tune gains conservatively** - Start low, increase gradually
4. **Monitor departure status** - Adjust drift thresholds as needed

## Dependencies

- **Core**: numpy, dataclasses
- **Communication**: multiprocessing.shared_memory, zmq
- **Types**: common.types.models

## Related Modules

- **detection/**: Lane detection input
- **integration/**: Shared memory and ZMQ communication
- **simulation/**: Vehicle control output

## Design Principles

1. **Single Responsibility**: Lane analysis separate from control logic
2. **Strategy Pattern**: Pluggable controller implementations
3. **Factory Pattern**: Centralized controller creation
4. **Interface Segregation**: Clean abstractions via ABC
5. **Type Safety**: Dataclasses with type hints throughout

## Troubleshooting

### Steering oscillation
- **Reduce Kp** - Too much proportional gain
- **Increase Kd** - Add more damping
- **Check detection quality** - Unstable lane detection causes oscillation

### Vehicle drifts off center
- **Increase Kp** - More aggressive centering
- **Add integral term** - Use PID instead of PD
- **Check calibration** - Ensure camera is centered

### Sluggish response
- **Increase Kp** - More responsive to lateral offset
- **Reduce Kd** - Less damping
- **Check throttle policy** - May be reducing speed too much

### PID windup
- **Reset state** periodically or on large deviations
- **Reduce Ki** - Less integral accumulation
- **Consider PD** - No integral windup

## Version

Current version: 0.1.0 (see [\_\_init\_\_.py](__init__.py))
