"""
Visualization Tools
Provides visualization utilities for lane detection and LKAS feedback.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from .lane_analyzer import LaneDepartureStatus


class LKASVisualizer:
    """Visualizer for Lane Keeping Assist System."""

    def __init__(self, image_width: int = 800, image_height: int = 600):
        """
        Initialize visualizer.

        Args:
            image_width: Width of output images
            image_height: Height of output images
        """
        self.image_width = image_width
        self.image_height = image_height

        # Colors (BGR format)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)

    def draw_lanes(
        self,
        image: np.ndarray,
        left_lane: Optional[Tuple[int, int, int, int]],
        right_lane: Optional[Tuple[int, int, int, int]],
        fill_lane: bool = True,
    ) -> np.ndarray:
        """
        Draw detected lane lines on image.

        Args:
            image: Input image
            left_lane: Left lane line (x1, y1, x2, y2)
            right_lane: Right lane line (x1, y1, x2, y2)
            fill_lane: Whether to fill the lane area

        Returns:
            Image with drawn lanes
        """
        output = image.copy()

        if fill_lane and left_lane and right_lane:
            # Fill lane area
            lane_poly = np.array(
                [
                    [
                        [left_lane[0], left_lane[1]],
                        [left_lane[2], left_lane[3]],
                        [right_lane[2], right_lane[3]],
                        [right_lane[0], right_lane[1]],
                    ]
                ],
                dtype=np.int32,
            )

            overlay = output.copy()
            cv2.fillPoly(overlay, lane_poly, self.COLOR_GREEN)
            output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

        # Draw lane lines
        if left_lane:
            cv2.line(
                output,
                (left_lane[0], left_lane[1]),
                (left_lane[2], left_lane[3]),
                self.COLOR_BLUE,
                3,
            )

        if right_lane:
            cv2.line(
                output,
                (right_lane[0], right_lane[1]),
                (right_lane[2], right_lane[3]),
                self.COLOR_BLUE,
                3,
            )

        return output

    def draw_vehicle_position(
        self,
        image: np.ndarray,
        vehicle_center_x: int,
        lane_center_x: Optional[float],
        departure_status: LaneDepartureStatus,
    ) -> np.ndarray:
        """
        Draw vehicle position indicator.

        Args:
            image: Input image
            vehicle_center_x: X coordinate of vehicle center
            lane_center_x: X coordinate of lane center
            departure_status: Current departure status

        Returns:
            Image with position indicator
        """
        output = image.copy()
        height = output.shape[0]

        # Draw vehicle center line
        cv2.line(
            output,
            (vehicle_center_x, height - 50),
            (vehicle_center_x, height),
            self.COLOR_WHITE,
            2,
        )

        # Draw lane center line if available
        if lane_center_x is not None:
            color = self._get_status_color(departure_status)
            cv2.line(
                output,
                (int(lane_center_x), height - 50),
                (int(lane_center_x), height),
                color,
                2,
            )

            # Draw offset arrow
            if abs(vehicle_center_x - lane_center_x) > 5:
                cv2.arrowedLine(
                    output,
                    (vehicle_center_x, height - 25),
                    (int(lane_center_x), height - 25),
                    color,
                    2,
                    tipLength=0.3,
                )

        return output

    def draw_hud(
        self,
        image: np.ndarray,
        metrics: Dict,
        show_steering: bool = True,
        steering_value: Optional[float] = None,
        vehicle_telemetry: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Draw heads-up display with metrics.

        Args:
            image: Input image
            metrics: Dictionary of lane metrics
            show_steering: Whether to show steering indicator
            steering_value: Steering correction value [-1, 1]
            vehicle_telemetry: Optional dict with vehicle data (speed, position, etc.)

        Returns:
            Image with HUD overlay
        """
        output = image.copy()

        # Create semi-transparent overlay for HUD
        overlay = output.copy()
        hud_height = 200 if vehicle_telemetry else 150
        cv2.rectangle(
            overlay, (0, 0), (output.shape[1], hud_height), self.COLOR_BLACK, -1
        )
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

        # Display metrics
        y_offset = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Departure status
        status = metrics.get("departure_status", LaneDepartureStatus.NO_LANES)
        status_color = self._get_status_color(status)
        cv2.putText(
            output,
            f"Status: {status.value.upper()}",
            (10, y_offset),
            font,
            font_scale,
            status_color,
            thickness,
        )
        y_offset += 30

        # Lateral offset
        offset_m = metrics.get("lateral_offset_meters")
        if offset_m is not None:
            direction = "RIGHT" if offset_m > 0 else "LEFT"
            cv2.putText(
                output,
                f"Offset: {abs(offset_m):.2f}m {direction}",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        else:
            cv2.putText(
                output,
                "Offset: N/A",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        y_offset += 30

        # Heading angle
        heading = metrics.get("heading_angle_deg")
        if heading is not None:
            cv2.putText(
                output,
                f"Heading: {heading:.1f} deg",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        else:
            cv2.putText(
                output,
                "Heading: N/A",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        y_offset += 30

        # Lane width
        lane_width = metrics.get("lane_width_pixels")
        if lane_width is not None:
            cv2.putText(
                output,
                f"Lane Width: {lane_width:.0f}px",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )

        # Vehicle telemetry (if provided)
        if vehicle_telemetry:
            y_offset += 40

            # Speed
            speed = vehicle_telemetry.get("speed_kmh", 0)
            cv2.putText(
                output,
                f"Speed: {speed:.1f} km/h",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_GREEN,
                thickness,
            )

            # Position
            pos = vehicle_telemetry.get("position")
            if pos:
                y_offset += 30
                cv2.putText(
                    output,
                    f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                    (10, y_offset),
                    font,
                    0.5,
                    self.COLOR_WHITE,
                    1,
                )

        # Steering indicator
        if show_steering and steering_value is not None:
            self._draw_steering_indicator(output, steering_value)

        return output

    def _draw_steering_indicator(self, image: np.ndarray, steering_value: float):
        """
        Draw steering wheel indicator.

        Args:
            image: Input image
            steering_value: Steering value [-1, 1]
        """
        # Position in top-right corner
        center_x = image.shape[1] - 100
        center_y = 75
        radius = 50

        # Draw steering wheel circle
        cv2.circle(image, (center_x, center_y), radius, self.COLOR_WHITE, 2)

        # Draw center mark
        cv2.circle(image, (center_x, center_y), 3, self.COLOR_WHITE, -1)

        # Draw steering indicator
        angle = steering_value * 90  # Max ±90 degrees
        angle_rad = np.radians(angle - 90)  # Adjust for coordinate system

        end_x = int(center_x + radius * 0.8 * np.cos(angle_rad))
        end_y = int(center_y + radius * 0.8 * np.sin(angle_rad))

        color = (
            self.COLOR_GREEN
            if abs(steering_value) < 0.3
            else self.COLOR_YELLOW if abs(steering_value) < 0.6 else self.COLOR_RED
        )

        cv2.line(image, (center_x, center_y), (end_x, end_y), color, 3)

        # Label
        cv2.putText(
            image,
            f"{steering_value:.2f}",
            (center_x - 30, center_y + radius + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.COLOR_WHITE,
            2,
        )

    def _get_status_color(self, status: LaneDepartureStatus) -> Tuple[int, int, int]:
        """
        Get color based on departure status.

        Args:
            status: Departure status

        Returns:
            BGR color tuple
        """
        if status == LaneDepartureStatus.CENTERED:
            return self.COLOR_GREEN
        elif status in [
            LaneDepartureStatus.LEFT_DRIFT,
            LaneDepartureStatus.RIGHT_DRIFT,
        ]:
            return self.COLOR_YELLOW
        elif status in [
            LaneDepartureStatus.LEFT_DEPARTURE,
            LaneDepartureStatus.RIGHT_DEPARTURE,
        ]:
            return self.COLOR_RED
        else:
            return self.COLOR_WHITE

    def create_alert_overlay(
        self,
        image: np.ndarray,
        departure_status: LaneDepartureStatus,
        blink: bool = False,
    ) -> np.ndarray:
        """
        Create visual alert overlay for lane departure warnings.

        Args:
            image: Input image
            departure_status: Current departure status
            blink: Whether to show blinking effect

        Returns:
            Image with alert overlay
        """
        output = image.copy()

        if departure_status in [
            LaneDepartureStatus.LEFT_DEPARTURE,
            LaneDepartureStatus.RIGHT_DEPARTURE,
        ]:
            if not blink or (blink and np.random.rand() > 0.5):
                # Create red border
                border_thickness = 10
                color = self.COLOR_RED

                height, width = output.shape[:2]
                cv2.rectangle(output, (0, 0), (width, height), color, border_thickness)

                # Warning text
                warning_text = "LANE DEPARTURE WARNING!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3

                text_size = cv2.getTextSize(warning_text, font, font_scale, thickness)[
                    0
                ]
                text_x = (width - text_size[0]) // 2
                text_y = 50

                # Draw text with background
                cv2.rectangle(
                    output,
                    (text_x - 10, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 10, text_y + 10),
                    self.COLOR_BLACK,
                    -1,
                )
                cv2.putText(
                    output,
                    warning_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    thickness,
                )

        elif departure_status in [
            LaneDepartureStatus.LEFT_DRIFT,
            LaneDepartureStatus.RIGHT_DRIFT,
        ]:
            # Yellow border for drift warning
            border_thickness = 5
            color = self.COLOR_YELLOW

            height, width = output.shape[:2]
            cv2.rectangle(output, (0, 0), (width, height), color, border_thickness)

        return output

    def create_combined_view(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        metrics: Dict,
        steering_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create combined view with original and processed images side by side.

        Args:
            original_image: Original camera image
            processed_image: Processed image with lane detection
            metrics: Lane metrics dictionary
            steering_value: Steering correction value

        Returns:
            Combined visualization
        """
        # Resize images to same size
        height = 480
        aspect_ratio = original_image.shape[1] / original_image.shape[0]
        width = int(height * aspect_ratio)

        original_resized = cv2.resize(original_image, (width, height))
        processed_resized = cv2.resize(processed_image, (width, height))

        # Create combined image
        combined = np.hstack([original_resized, processed_resized])

        # Add HUD to combined view
        combined = self.draw_hud(
            combined, metrics, show_steering=True, steering_value=steering_value
        )

        # Add labels
        cv2.putText(
            combined,
            "ORIGINAL",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLOR_WHITE,
            2,
        )
        cv2.putText(
            combined,
            "LANE DETECTION",
            (width + 10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLOR_WHITE,
            2,
        )

        return combined


if __name__ == "__main__":
    # Example usage
    from lane_analyzer import LaneAnalyzer

    # Create test image
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)

    # Create visualizer and analyzer
    visualizer = LKASVisualizer(800, 600)
    analyzer = LaneAnalyzer(800, 600)

    # Example lane lines
    left_lane = (200, 600, 350, 360)
    right_lane = (600, 600, 450, 360)

    # Get metrics
    metrics = analyzer.get_metrics(left_lane, right_lane)

    # Draw visualization
    output = visualizer.draw_lanes(test_image, left_lane, right_lane)
    output = visualizer.draw_vehicle_position(
        output, 400, metrics["lane_center_x"], metrics["departure_status"]
    )
    output = visualizer.draw_hud(
        output, metrics, show_steering=True, steering_value=-0.3
    )

    print("Visualization created successfully")
