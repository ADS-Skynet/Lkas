"""
Decision Controller

Main controller that receives detection results and generates control commands.
Combines lane analysis with PD control logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from detection.integration.messages import (
    DetectionMessage,
    ControlMessage,
    LaneMessage,
    ControlMode,
)
from .analyzer import LaneAnalyzer
from detection.processing.pd_controller import PDController


class DecisionController:
    """
    Decision controller for lane keeping.

    Responsibility:
    - Receive lane detection results
    - Analyze lane geometry and vehicle position
    - Compute control commands (steering, throttle, brake)
    - Generate control messages for CARLA module
    """

    def __init__(
        self, image_width: int, image_height: int, kp: float = 0.5, kd: float = 0.1
    ):
        """
        Initialize decision controller.

        Args:
            image_width: Camera image width
            image_height: Camera image height
            kp: Proportional gain for steering control
            kd: Derivative gain for steering control
        """
        # Lane analysis
        self.analyzer = LaneAnalyzer(image_width=image_width, image_height=image_height)

        # Steering control
        self.pd_controller = PDController(kp=kp, kd=kd)

        # Default throttle/brake
        self.default_throttle = 0.3
        self.default_brake = 0.0

        # Control mode
        self.mode = ControlMode.LANE_KEEPING

    def process_detection(self, detection: DetectionMessage) -> ControlMessage:
        """
        Process detection results and generate control commands.

        Args:
            detection: Lane detection message

        Returns:
            Control message with steering, throttle, brake commands
        """
        # Convert detection message lanes to internal format
        left_lane = None
        right_lane = None

        if detection.left_lane:
            left_lane = (
                detection.left_lane.x1,
                detection.left_lane.y1,
                detection.left_lane.x2,
                detection.left_lane.y2,
            )

        if detection.right_lane:
            right_lane = (
                detection.right_lane.x1,
                detection.right_lane.y1,
                detection.right_lane.x2,
                detection.right_lane.y2,
            )

        # Analyze lanes to get metrics
        metrics = self.analyzer.get_metrics(left_lane, right_lane)

        # Compute steering from metrics
        steering = self.pd_controller.compute_steering(metrics)

        # If no steering computed (e.g., no lanes detected), use safe default
        if steering is None:
            steering = 0.0
            # Apply brake when no lanes detected
            throttle = 0.0
            brake = 0.3
        else:
            throttle = self.default_throttle
            brake = self.default_brake

        # Create control message
        control = ControlMessage(
            steering=steering,
            throttle=throttle,
            brake=brake,
            mode=self.mode,
            lateral_offset=metrics.lateral_offset_normalized,
            heading_angle=metrics.heading_angle_deg,
        )

        # Ensure values are clamped
        control.clamp_values()

        return control

    def set_control_mode(self, mode: ControlMode):
        """Set control mode."""
        self.mode = mode

    def set_throttle_brake(self, throttle: float, brake: float):
        """Set default throttle and brake values."""
        self.default_throttle = max(0.0, min(1.0, throttle))
        self.default_brake = max(0.0, min(1.0, brake))

    def set_controller_gains(self, kp: float, kd: float):
        """Update PD controller gains."""
        self.pd_controller.set_gains(kp, kd)

    def get_controller_gains(self) -> tuple:
        """Get current controller gains."""
        return self.pd_controller.get_gains()

    def get_analyzer(self) -> LaneAnalyzer:
        """Get lane analyzer instance."""
        return self.analyzer
