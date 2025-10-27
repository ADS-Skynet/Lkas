"""
Inter-Module Message Models

Defines the data structures passed between the three modules:
- CARLA Module → Detection Module: Image data
- Detection Module → Decision Module: Lane detection results
- Decision Module → CARLA Module: Control commands
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from enum import Enum


# =============================================================================
# CARLA → DETECTION: Image Data
# =============================================================================

@dataclass
class ImageMessage:
    """
    Image data from CARLA camera to detection module.

    Attributes:
        image: RGB image array (H, W, 3)
        timestamp: Simulation timestamp
        frame_id: Frame sequence number
        camera_transform: Camera position/rotation (optional)
    """
    image: np.ndarray
    timestamp: float
    frame_id: int
    camera_transform: Optional[dict] = None

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


# =============================================================================
# DETECTION → DECISION: Lane Detection Results
# =============================================================================

@dataclass
class LaneMessage:
    """
    Single lane line representation.

    Attributes:
        x1, y1: Starting point (bottom of image)
        x2, y2: Ending point (top of region of interest)
        confidence: Detection confidence [0, 1]
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    @property
    def slope(self) -> float:
        """Calculate lane slope."""
        if self.y2 == self.y1:
            return 0.0
        return (self.x2 - self.x1) / (self.y2 - self.y1)


@dataclass
class DetectionMessage:
    """
    Lane detection results from detection module to decision module.

    Attributes:
        left_lane: Left lane line (if detected)
        right_lane: Right lane line (if detected)
        processing_time_ms: Detection processing time
        debug_image: Visualization image (optional)
        frame_id: Corresponding frame ID
        timestamp: Detection timestamp
    """
    left_lane: Optional[LaneMessage]
    right_lane: Optional[LaneMessage]
    processing_time_ms: float
    frame_id: int
    timestamp: float
    debug_image: Optional[np.ndarray] = None

    @property
    def has_both_lanes(self) -> bool:
        """Check if both lanes were detected."""
        return self.left_lane is not None and self.right_lane is not None

    @property
    def has_any_lane(self) -> bool:
        """Check if at least one lane was detected."""
        return self.left_lane is not None or self.right_lane is not None


# =============================================================================
# DECISION → CARLA: Control Commands
# =============================================================================

class ControlMode(Enum):
    """Control mode for the vehicle."""
    MANUAL = "manual"
    AUTOPILOT = "autopilot"
    LANE_KEEPING = "lane_keeping"


@dataclass
class ControlMessage:
    """
    Control commands from decision module to CARLA module.

    Attributes:
        steering: Steering angle [-1, 1] (left to right)
        throttle: Throttle [0, 1]
        brake: Brake [0, 1]
        mode: Control mode
        lateral_offset: Lateral offset from lane center (for logging)
        heading_angle: Heading angle relative to lane (for logging)
    """
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    mode: ControlMode = ControlMode.LANE_KEEPING

    # Diagnostic info (not used for control)
    lateral_offset: Optional[float] = None
    heading_angle: Optional[float] = None

    def clamp_values(self):
        """Ensure all control values are within valid ranges."""
        self.steering = max(-1.0, min(1.0, self.steering))
        self.throttle = max(0.0, min(1.0, self.throttle))
        self.brake = max(0.0, min(1.0, self.brake))


# =============================================================================
# SYSTEM STATUS MESSAGES
# =============================================================================

@dataclass
class SystemStatus:
    """
    Overall system status message.

    Attributes:
        carla_connected: CARLA connection status
        vehicle_spawned: Vehicle spawn status
        camera_ready: Camera sensor status
        detector_ready: Lane detector status
        controller_ready: Controller status
    """
    carla_connected: bool = False
    vehicle_spawned: bool = False
    camera_ready: bool = False
    detector_ready: bool = False
    controller_ready: bool = False

    @property
    def is_ready(self) -> bool:
        """Check if all systems are ready."""
        return (self.carla_connected and
                self.vehicle_spawned and
                self.camera_ready and
                self.detector_ready and
                self.controller_ready)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for monitoring.

    Attributes:
        fps: Frames per second
        detection_time_ms: Average detection time
        control_time_ms: Average control computation time
        total_frames: Total frames processed
    """
    fps: float = 0.0
    detection_time_ms: float = 0.0
    control_time_ms: float = 0.0
    total_frames: int = 0
