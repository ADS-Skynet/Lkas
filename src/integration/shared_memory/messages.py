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
    Image data from camera to detection module.

    Attributes:
        image: RGB image array (H, W, 3)
        timestamp: Simulation timestamp
        frame_id: Frame sequence number
        camera_transform: Camera position/rotation (optional)
        depth_image: Depth image array (H, W) uint16 (optional)
        depth_scale: Depth scale factor (meters per raw unit)
    """
    image: np.ndarray
    timestamp: float
    frame_id: int
    camera_transform: dict | None = None
    depth_image: np.ndarray | None = None
    depth_scale: float = 0.0

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
    Single lane line representation (for CV detection - two endpoints).

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
class LaneContour:
    """
    Lane contour representation (for DL detection - multiple points).

    Attributes:
        points: List of (x, y) points forming the lane contour
        class_id: Lane class ID from segmentation (1-4 for multi-class)
        confidence: Detection confidence [0, 1]
    """
    points: list  # List of [x, y] points
    class_id: int = 1
    confidence: float = 1.0

    @property
    def num_points(self) -> int:
        """Number of points in the contour."""
        return len(self.points)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'points': self.points,
            'class_id': self.class_id,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LaneContour':
        """Create from dictionary."""
        return cls(
            points=data['points'],
            class_id=data.get('class_id', 1),
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class DetectionMessage:
    """
    Lane detection results from detection module to decision module.

    Attributes:
        left_lane: Left lane line for CV detection (if detected)
        right_lane: Right lane line for CV detection (if detected)
        lanes: List of lane contours for DL detection (multiple lanes)
        processing_time_ms: Detection processing time
        debug_image: Visualization image (optional)
        frame_id: Corresponding frame ID
        timestamp: Detection timestamp
        segmentation_mask: DL segmentation mask for visualization (optional, H x W, uint8)
        detection_method: Detection method used ('cv' or 'dl')
    """
    left_lane: LaneMessage | None
    right_lane: LaneMessage | None
    processing_time_ms: float
    frame_id: int
    timestamp: float
    debug_image: np.ndarray | None = None
    segmentation_mask: np.ndarray | None = None  # For DL visualization
    detection_method: str = "cv"  # 'cv' or 'dl'
    lanes: list | None = None  # List of LaneContour for DL detection

    @property
    def has_both_lanes(self) -> bool:
        """Check if both lanes were detected (CV mode)."""
        return self.left_lane is not None and self.right_lane is not None

    @property
    def has_any_lane(self) -> bool:
        """Check if at least one lane was detected."""
        if self.lanes:
            return len(self.lanes) > 0
        return self.left_lane is not None or self.right_lane is not None

    @property
    def has_segmentation(self) -> bool:
        """Check if segmentation mask is available (DL detection)."""
        return self.segmentation_mask is not None

    @property
    def num_lanes(self) -> int:
        """Get number of detected lanes."""
        if self.lanes:
            return len(self.lanes)
        count = 0
        if self.left_lane:
            count += 1
        if self.right_lane:
            count += 1
        return count


# =============================================================================
# DECISION → CARLA: Control Commands
# =============================================================================

class ObstacleAction(Enum):
    """Obstacle avoidance action type."""
    NORMAL = "normal"
    AVOID_LEFT = "avoid_left"
    AVOID_RIGHT = "avoid_right"
    STOP = "stop"
    SLOW = "slow"


@dataclass
class ObstacleMessage:
    """
    Obstacle avoidance status from YOLO obstacle detection module.

    Written by the YOLO obstacle avoidance script, read by the decision server
    to integrate obstacle awareness into lane keeping control.

    Attributes:
        active: Whether obstacle avoidance is currently intervening
        action: Avoidance action being taken
        distance: Distance to nearest obstacle in meters (-1 if none)
        steering: Recommended steering override [-1, 1]
        throttle: Recommended throttle [0, 1]
        brake: Recommended brake [0, 1]
        timestamp: When obstacle data was last written
        frame_id: Frame ID when obstacle was detected
    """
    active: bool = False
    action: ObstacleAction = ObstacleAction.NORMAL
    distance: float = -1.0
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    timestamp: float = 0.0
    frame_id: int = 0


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
        lateral_offset: Lateral offset from lane center (normalized, for logging)
        heading_angle: Heading angle relative to lane (for logging)
        lane_width_pixels: Lane width in pixels (for logging/metrics)
        departure_status: Lane departure status string (for logging/metrics)
    """
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    mode: ControlMode = ControlMode.LANE_KEEPING

    # Diagnostic info (not used for control, broadcasted to viewer for metrics)
    lateral_offset: float | None = None  # Normalized offset [-1, 1]
    lateral_offset_meters: float | None = None  # Offset in meters
    heading_angle: float | None = None
    lane_width_pixels: float | None = None
    departure_status: str | None = None

    # Debug polynomial coefficients for viewer overlay (x = ay^2 + by + c)
    left_poly: tuple | None = None    # (a, b, c) or None
    right_poly: tuple | None = None   # (a, b, c) or None
    center_poly: tuple | None = None  # (a, b, c) or None

    # Lane boundary confidence scores [0, 1] (fit_quality * coverage)
    left_confidence: float = 0.0
    right_confidence: float = 0.0

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
