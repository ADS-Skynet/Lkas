"""
Decision Controller

Main controller that receives detection results and generates control commands.
Combines lane analysis with PD control logic.
"""

from simulation.integration.messages import (
    DetectionMessage,
    ControlMessage,
    LaneMessage,
    ControlMode,
)
from simulation.utils.lane_analyzer import LaneAnalyzer
from simulation.processing.pd_controller import PDController


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
        self,
        image_width: int,
        image_height: int,
        kp: float = 0.5,
        kd: float = 0.1,
        throttle_policy: dict | None = None,
    ):
        """
        Initialize decision controller.

        Args:
            image_width: Camera image width
            image_height: Camera image height
            kp: Proportional gain for steering control
            kd: Derivative gain for steering control
            throttle_policy: Adaptive throttle configuration dict with keys:
                - base: Base throttle value (default: 0.45)
                - min: Minimum throttle value (default: 0.18)
                - steer_threshold: Steering magnitude to start reducing throttle (default: 0.15)
                - steer_max: Maximum steering for throttle calculation (default: 0.70)
        """
        # Lane analysis
        self.analyzer = LaneAnalyzer(image_width=image_width, image_height=image_height)

        # Steering control
        self.pd_controller = PDController(kp=kp, kd=kd)

        # Adaptive throttle policy
        self.throttle_policy = throttle_policy or {
            "base": 0.15,
            "min": 0.05,
            "steer_threshold": 0.15,
            "steer_max": 0.70,
        }

        # Default throttle/brake (used when adaptive throttle is disabled)
        self.default_throttle = 0.3
        self.default_brake = 0.0
        self.use_adaptive_throttle = throttle_policy is not None

        # Control mode
        self.mode = ControlMode.LANE_KEEPING

    def compute_adaptive_throttle(self, steering: float) -> float:
        """
        Compute adaptive throttle based on steering magnitude.

        The throttle decreases as steering increases to prevent overshooting in turns.
        This helps maintain stable control during sharp maneuvers.

        Args:
            steering: Steering value in range [-1, 1]

        Returns:
            Throttle value in range [throttle_min, throttle_base]
        """
        abs_steering = abs(steering)
        policy = self.throttle_policy

        # If steering is below threshold, use base throttle
        if abs_steering <= policy["steer_threshold"]:
            return policy["base"]

        # Calculate linear interpolation factor between threshold and max
        steer_range = policy["steer_max"] - policy["steer_threshold"]
        steer_delta = abs_steering - policy["steer_threshold"]
        t = max(0.0, min(1.0, steer_delta / max(1e-6, steer_range)))

        # Interpolate between base and min throttle
        throttle_range = policy["base"] - policy["min"]
        throttle = policy["base"] - (throttle_range * t)

        return max(policy["min"], min(policy["base"], throttle))

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
            # Use adaptive throttle if enabled, otherwise use default
            if self.use_adaptive_throttle:
                throttle = self.compute_adaptive_throttle(steering)
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
