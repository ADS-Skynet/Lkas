"""
Lane Position Analyzer
Calculates vehicle position relative to lane and provides metrics for LKAS.
"""

import numpy as np
from typing import Tuple, Dict
from enum import Enum

from detection.core.models import Lane, LaneMetrics
from detection.core.models import LaneDepartureStatus as CoreLaneDepartureStatus


class LaneDepartureStatus(Enum):
    """Enum for lane departure status."""
    CENTERED = "centered"
    LEFT_DRIFT = "left_drift"
    RIGHT_DRIFT = "right_drift"
    LEFT_DEPARTURE = "left_departure"
    RIGHT_DEPARTURE = "right_departure"
    NO_LANES = "no_lanes"


class LaneAnalyzer:
    """Analyzes lane position and calculates metrics for LKAS."""

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 drift_threshold: float = 0.15,
                 departure_threshold: float = 0.35,
                 lane_width_meters: float = 3.7):
        """
        Initialize lane analyzer.

        Args:
            image_width: Width of camera image in pixels
            image_height: Height of camera image in pixels
            drift_threshold: Threshold for drift warning (fraction of lane width)
            departure_threshold: Threshold for departure warning (fraction of lane width)
            lane_width_meters: Standard lane width in meters (default 3.7m for US highways)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.drift_threshold = drift_threshold
        self.departure_threshold = departure_threshold
        self.lane_width_meters = lane_width_meters

        # Camera assumed to be centered on vehicle
        self.vehicle_center_x = image_width // 2

    def calculate_lane_center(self,
                              left_lane: Lane | Tuple[int, int, int, int] | None,
                              right_lane: Lane | Tuple[int, int, int, int] | None,
                              y_position: int | None = None) -> float | None:
        """
        Calculate the center of the lane at a given y position.

        Args:
            left_lane: Left lane (Lane object or tuple (x1, y1, x2, y2))
            right_lane: Right lane (Lane object or tuple (x1, y1, x2, y2))
            y_position: Y position to calculate center at (default: bottom of image)

        Returns:
            X coordinate of lane center or None if cannot be calculated
        """
        if left_lane is None or right_lane is None:
            return None

        if y_position is None:
            y_position = self.image_height - 1

        # Interpolate x positions at the given y
        left_x = self._interpolate_x(left_lane, y_position)
        right_x = self._interpolate_x(right_lane, y_position)

        if left_x is None or right_x is None:
            return None

        return (left_x + right_x) / 2.0

    def _interpolate_x(self, lane: Lane | Tuple[int, int, int, int], y: int) -> float | None:
        """
        Interpolate x coordinate at given y position on a lane line.

        Args:
            lane: Lane object or tuple (x1, y1, x2, y2)
            y: Y position

        Returns:
            Interpolated x coordinate or None
        """
        # Handle both Lane objects and tuples for backward compatibility
        if isinstance(lane, Lane):
            x1, y1, x2, y2 = lane.x1, lane.y1, lane.x2, lane.y2
        else:
            x1, y1, x2, y2 = lane

        if y2 == y1:
            return float(x1)

        # Linear interpolation
        t = (y - y1) / (y2 - y1)

        # Check if y is within the lane segment
        if t < 0 or t > 1:
            # Extrapolate if needed
            pass

        x = x1 + t * (x2 - x1)
        return float(x)

    def calculate_lateral_offset(self,
                                 left_lane: Lane | Tuple[int, int, int, int] | None,
                                 right_lane: Lane | Tuple[int, int, int, int] | None) -> float | None:
        """
        Calculate lateral offset from lane center in pixels.
        Positive value means vehicle is to the right of center.

        Args:
            left_lane: Left lane (Lane object or tuple)
            right_lane: Right lane (Lane object or tuple)

        Returns:
            Lateral offset in pixels or None
        """
        lane_center = self.calculate_lane_center(left_lane, right_lane)

        if lane_center is None:
            return None

        offset = self.vehicle_center_x - lane_center
        return offset

    def calculate_lateral_offset_meters(self,
                                       left_lane: Lane | Tuple[int, int, int, int] | None,
                                       right_lane: Lane | Tuple[int, int, int, int] | None) -> float | None:
        """
        Calculate lateral offset from lane center in meters.

        Args:
            left_lane: Left lane line
            right_lane: Right lane line

        Returns:
            Lateral offset in meters or None
        """
        offset_pixels = self.calculate_lateral_offset(left_lane, right_lane)

        if offset_pixels is None:
            return None

        # Calculate lane width in pixels
        lane_width_pixels = self.calculate_lane_width(left_lane, right_lane)

        if lane_width_pixels is None or lane_width_pixels == 0:
            return None

        # Convert to meters
        pixels_per_meter = lane_width_pixels / self.lane_width_meters
        offset_meters = offset_pixels / pixels_per_meter

        return offset_meters

    def calculate_lane_width(self,
                            left_lane: Lane | Tuple[int, int, int, int] | None,
                            right_lane: Lane | Tuple[int, int, int, int] | None) -> float | None:
        """
        Calculate lane width in pixels.

        Args:
            left_lane: Left lane line
            right_lane: Right lane line

        Returns:
            Lane width in pixels or None
        """
        if left_lane is None or right_lane is None:
            return None

        y_position = self.image_height - 1

        left_x = self._interpolate_x(left_lane, y_position)
        right_x = self._interpolate_x(right_lane, y_position)

        if left_x is None or right_x is None:
            return None

        return abs(right_x - left_x)

    def calculate_heading_angle(self,
                                left_lane: Lane | Tuple[int, int, int, int] | None,
                                right_lane: Lane | Tuple[int, int, int, int] | None) -> float | None:
        """
        Calculate vehicle heading angle relative to lane direction in degrees.
        Positive angle means vehicle is pointing right relative to lane.

        Args:
            left_lane: Left lane (Lane object or tuple)
            right_lane: Right lane (Lane object or tuple)

        Returns:
            Heading angle in degrees or None
        """
        if left_lane is None and right_lane is None:
            return None

        # Use available lane to estimate heading
        lane = left_lane if left_lane is not None else right_lane

        # Handle both Lane objects and tuples
        if isinstance(lane, Lane):
            x1, y1, x2, y2 = lane.x1, lane.y1, lane.x2, lane.y2
        else:
            x1, y1, x2, y2 = lane

        # Calculate angle
        dx = x2 - x1
        dy = y2 - y1

        if dy == 0:
            return 0.0

        angle_rad = np.arctan2(dx, dy)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def get_departure_status(self,
                            left_lane: Lane | Tuple[int, int, int, int] | None,
                            right_lane: Lane | Tuple[int, int, int, int] | None) -> LaneDepartureStatus:
        """
        Determine lane departure status.

        Args:
            left_lane: Left lane line
            right_lane: Right lane line

        Returns:
            LaneDepartureStatus enum value
        """
        if left_lane is None and right_lane is None:
            return LaneDepartureStatus.NO_LANES

        offset_pixels = self.calculate_lateral_offset(left_lane, right_lane)

        if offset_pixels is None:
            return LaneDepartureStatus.NO_LANES

        lane_width = self.calculate_lane_width(left_lane, right_lane)

        if lane_width is None or lane_width == 0:
            return LaneDepartureStatus.NO_LANES

        # Calculate offset as fraction of lane width
        offset_fraction = abs(offset_pixels) / lane_width

        # Determine status
        if offset_fraction >= self.departure_threshold:
            if offset_pixels > 0:
                return LaneDepartureStatus.RIGHT_DEPARTURE
            else:
                return LaneDepartureStatus.LEFT_DEPARTURE
        elif offset_fraction >= self.drift_threshold:
            if offset_pixels > 0:
                return LaneDepartureStatus.RIGHT_DRIFT
            else:
                return LaneDepartureStatus.LEFT_DRIFT
        else:
            return LaneDepartureStatus.CENTERED

    def get_metrics(self,
                   left_lane: Lane | Tuple[int, int, int, int] | None,
                   right_lane: Lane | Tuple[int, int, int, int] | None) -> LaneMetrics:
        """
        Get all lane analysis metrics.

        Args:
            left_lane: Left lane (Lane object or tuple)
            right_lane: Right lane (Lane object or tuple)

        Returns:
            LaneMetrics object with all calculated metrics
        """
        # Calculate all metrics
        lateral_offset_pixels = self.calculate_lateral_offset(left_lane, right_lane)
        lateral_offset_meters = self.calculate_lateral_offset_meters(left_lane, right_lane)
        lane_width_pixels = self.calculate_lane_width(left_lane, right_lane)
        heading_angle_deg = self.calculate_heading_angle(left_lane, right_lane)
        departure_status = self.get_departure_status(left_lane, right_lane)
        lane_center_x = self.calculate_lane_center(left_lane, right_lane)

        # Calculate normalized offset
        lateral_offset_normalized = None
        if lateral_offset_pixels is not None and lane_width_pixels is not None and lane_width_pixels > 0:
            lateral_offset_normalized = lateral_offset_pixels / lane_width_pixels

        # Convert local enum to core enum
        core_status = self._convert_departure_status(departure_status)

        # Create LaneMetrics object
        return LaneMetrics(
            vehicle_center_x=float(self.vehicle_center_x),
            lane_center_x=lane_center_x,
            lane_width_pixels=lane_width_pixels,
            lateral_offset_pixels=lateral_offset_pixels,
            lateral_offset_meters=lateral_offset_meters,
            lateral_offset_normalized=lateral_offset_normalized,
            heading_angle_deg=heading_angle_deg,
            departure_status=core_status,
            has_left_lane=(left_lane is not None),
            has_right_lane=(right_lane is not None),
            has_both_lanes=(left_lane is not None and right_lane is not None)
        )

    def _convert_departure_status(self, status: LaneDepartureStatus) -> CoreLaneDepartureStatus:
        """Convert local LaneDepartureStatus to core LaneDepartureStatus."""
        mapping = {
            LaneDepartureStatus.CENTERED: CoreLaneDepartureStatus.CENTERED,
            LaneDepartureStatus.LEFT_DRIFT: CoreLaneDepartureStatus.LEFT_DRIFT,
            LaneDepartureStatus.RIGHT_DRIFT: CoreLaneDepartureStatus.RIGHT_DRIFT,
            LaneDepartureStatus.LEFT_DEPARTURE: CoreLaneDepartureStatus.LEFT_DEPARTURE,
            LaneDepartureStatus.RIGHT_DEPARTURE: CoreLaneDepartureStatus.RIGHT_DEPARTURE,
            LaneDepartureStatus.NO_LANES: CoreLaneDepartureStatus.UNKNOWN
        }
        return mapping.get(status, CoreLaneDepartureStatus.UNKNOWN)

    def get_steering_correction(self,
                                left_lane: Lane | Tuple[int, int, int, int] | None,
                                right_lane: Lane | Tuple[int, int, int, int] | None,
                                kp: float = 0.5,
                                kd: float = 0.1) -> float | None:
        """
        Calculate suggested steering correction using PD controller.

        Args:
            left_lane: Left lane line
            right_lane: Right lane line
            kp: Proportional gain
            kd: Derivative gain

        Returns:
            Steering correction value [-1, 1] or None
            Negative = steer left, Positive = steer right
        """
        offset = self.calculate_lateral_offset(left_lane, right_lane)
        heading = self.calculate_heading_angle(left_lane, right_lane)

        if offset is None or heading is None:
            return None

        # Normalize offset to [-1, 1]
        lane_width = self.calculate_lane_width(left_lane, right_lane)
        if lane_width is None or lane_width == 0:
            return None

        normalized_offset = offset / (lane_width / 2.0)
        normalized_offset = np.clip(normalized_offset, -1.0, 1.0)

        # Normalize heading to [-1, 1] (assuming max ±30 degrees)
        normalized_heading = heading / 30.0
        normalized_heading = np.clip(normalized_heading, -1.0, 1.0)

        # PD control
        correction = -(kp * normalized_offset + kd * normalized_heading)
        correction = np.clip(correction, -1.0, 1.0)

        return correction


if __name__ == "__main__":
    # Example usage
    analyzer = LaneAnalyzer(image_width=800, image_height=600)

    # Example lane lines
    left_lane = (200, 600, 350, 360)
    right_lane = (600, 600, 450, 360)

    # Get metrics
    metrics = analyzer.get_metrics(left_lane, right_lane)

    print("Lane Analysis Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Get steering correction
    correction = analyzer.get_steering_correction(left_lane, right_lane)
    print(f"\nSteering correction: {correction:.3f}")
