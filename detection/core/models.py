"""
Data models for lane detection system.
Uses dataclasses for type safety and clean data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import numpy as np


class LaneDepartureStatus(Enum):
    """Lane departure status enumeration."""
    UNKNOWN = "Unknown"
    CENTERED = "Centered"
    LEFT_DRIFT = "Drifting Left"
    RIGHT_DRIFT = "Drifting Right"
    LEFT_DEPARTURE = "LEFT DEPARTURE!"
    RIGHT_DEPARTURE = "RIGHT DEPARTURE!"


@dataclass
class Lane:
    """
    Represents a detected lane line.

    Attributes:
        x1, y1: Starting point (bottom of image)
        x2, y2: Ending point (top of ROI)
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
        if self.x2 - self.x1 == 0:
            return float('inf')
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    @property
    def length(self) -> float:
        """Calculate lane line length."""
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple format (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_tuple(cls, coords: Tuple[int, int, int, int], confidence: float = 1.0) -> 'Lane':
        """Create Lane from tuple (x1, y1, x2, y2)."""
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3], confidence=confidence)

    def __bool__(self) -> bool:
        """Lane is truthy if it exists."""
        return True


@dataclass
class LaneMetrics:
    """
    Lane analysis metrics.

    All measurements and derived values for lane keeping assist.
    """
    # Raw measurements
    vehicle_center_x: Optional[float] = None
    lane_center_x: Optional[float] = None
    lane_width_pixels: Optional[float] = None

    # Calculated metrics
    lateral_offset_pixels: Optional[float] = None
    lateral_offset_meters: Optional[float] = None
    lateral_offset_normalized: Optional[float] = None
    heading_angle_deg: Optional[float] = None

    # Status
    departure_status: LaneDepartureStatus = LaneDepartureStatus.UNKNOWN

    # Metadata
    has_left_lane: bool = False
    has_right_lane: bool = False
    has_both_lanes: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return {
            'vehicle_center_x': self.vehicle_center_x,
            'lane_center_x': self.lane_center_x,
            'lane_width_pixels': self.lane_width_pixels,
            'lateral_offset_pixels': self.lateral_offset_pixels,
            'lateral_offset_meters': self.lateral_offset_meters,
            'lateral_offset_normalized': self.lateral_offset_normalized,
            'heading_angle_deg': self.heading_angle_deg,
            'departure_status': self.departure_status,
            'has_left_lane': self.has_left_lane,
            'has_right_lane': self.has_right_lane,
            'has_both_lanes': self.has_both_lanes,
        }


@dataclass
class VehicleTelemetry:
    """Vehicle telemetry data."""
    speed_kmh: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0
    gear: int = 0

    # Position (optional)
    location_x: Optional[float] = None
    location_y: Optional[float] = None
    location_z: Optional[float] = None

    # Rotation (optional)
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None


@dataclass
class DetectionResult:
    """
    Complete result from lane detection process.

    Combines detected lanes with debug visualization.
    """
    left_lane: Optional[Lane] = None
    right_lane: Optional[Lane] = None
    debug_image: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0

    @property
    def has_left_lane(self) -> bool:
        return self.left_lane is not None

    @property
    def has_right_lane(self) -> bool:
        return self.right_lane is not None

    @property
    def has_both_lanes(self) -> bool:
        return self.left_lane is not None and self.right_lane is not None
