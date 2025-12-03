"""
Decision Controller

Main controller that receives detection results and generates control commands.
Combines lane analysis with PD control logic.
"""

from lkas.integration.messages import (
    DetectionMessage,
    ControlMessage,
    LaneMessage,
    ControlMode,
)
from lkas.decision.lane_analyzer import LaneAnalyzer
from lkas.decision.pd_controller import PDController
from lkas.decision.pid_controller import PIDController


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
        ki: float = 0.01,
        kd: float = 0.1,
        controller_method: str = "pd",
        throttle_policy: dict | None = None,
    ):
        """
        Initialize decision controller.

        Args:
            image_width: Camera image width
            image_height: Camera image height
            kp: Proportional gain for steering control
            ki: Integral gain for steering control (PID only)
            kd: Derivative gain for steering control
            controller_method: Controller type ('pd' or 'pid')
            throttle_policy: Adaptive throttle configuration dict with keys:
                - base: Base throttle value (default: 0.45)
                - min: Minimum throttle value (default: 0.18)
                - steer_threshold: Steering magnitude to start reducing throttle (default: 0.15)
                - steer_max: Maximum steering for throttle calculation (default: 0.70)
        """
        # Lane analysis
        self.analyzer = LaneAnalyzer(image_width=image_width, image_height=image_height)

        # Steering control - instantiate based on controller_method
        self.controller_method = controller_method.lower()
        if self.controller_method == "pid":
            self.controller = PIDController(kp=kp, ki=ki, kd=kd)
        else:  # Default to PD
            self.controller = PDController(kp=kp, kd=kd)

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
        steering = self.controller.compute_steering(metrics)

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

        # Create control message with complete metrics
        control = ControlMessage(
            steering=steering,
            throttle=throttle,
            brake=brake,
            mode=self.mode,
            lateral_offset=metrics.lateral_offset_normalized,
            lateral_offset_meters=metrics.lateral_offset_meters,
            heading_angle=metrics.heading_angle_deg,
            lane_width_pixels=metrics.lane_width_pixels,
            departure_status=metrics.departure_status.value if metrics.departure_status else None,
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

    def set_controller_gains(self, kp: float, ki: float | None = None, kd: float | None = None):
        """
        Update controller gains.

        Args:
            kp: Proportional gain
            ki: Integral gain (for PID only, optional)
            kd: Derivative gain (optional)
        """
        if self.controller_method == "pid":
            # For PID, set all three gains
            current_gains = self.controller.get_gains()
            ki_val = ki if ki is not None else current_gains[1]
            kd_val = kd if kd is not None else current_gains[2]
            self.controller.set_gains(kp, ki_val, kd_val)
        else:
            # For PD, only set kp and kd
            kd_val = kd if kd is not None else self.controller.get_gains()[1]
            self.controller.set_gains(kp, kd_val)

    def get_controller_gains(self) -> tuple:
        """
        Get current controller gains.

        Returns:
            Tuple of (kp, kd) for PD or (kp, ki, kd) for PID
        """
        return self.controller.get_gains()

    def get_analyzer(self) -> LaneAnalyzer:
        """Get lane analyzer instance."""
        return self.analyzer

    def reset_state(self):
        """
        Reset controller state (PID error accumulation and derivative).

        Used when:
            - User presses reset button
            - Starting a new session
            - After significant disturbance

        Only affects PID controller (PD controller has no state to reset).
        """
        if self.controller_method == "pid":
            self.controller.reset_state()
            # print("✓ PID controller state reset (integral and derivative cleared)")
        # else:
            # print("ℹ PD controller has no state to reset")

    def update_parameter(self, param_name: str, value: float) -> bool:
        """
        Update a decision parameter in real-time.

        Args:
            param_name: Name of parameter to update
            value: New value

        Returns:
            True if parameter was updated successfully, False otherwise
        """
        # Handle special 'reset' parameter
        if param_name == 'reset':
            self.reset_state()
            return True

        # Map of valid parameters and their value constraints
        valid_params = {
            'kp': (0.0, 2.0),              # Proportional gain
            'ki': (0.0, 0.5),              # Integral gain (PID only)
            'kd': (0.0, 1.0),              # Derivative gain
            'throttle_base': (0.0, 1.0),   # Base throttle
            'throttle_min': (0.0, 1.0),    # Minimum throttle
            'steer_threshold': (0.0, 1.0), # Steering threshold
            'steer_max': (0.0, 1.0),       # Maximum steering
        }

        if param_name not in valid_params:
            print(f"⚠ Unknown parameter: {param_name}")
            return False

        # Validate value range
        min_val, max_val = valid_params[param_name]
        if not (min_val <= value <= max_val):
            print(f"⚠ Value {value} out of range [{min_val}, {max_val}] for {param_name}")
            return False

        # Update the parameter
        if param_name == 'kp':
            self.controller.kp = float(value)
        elif param_name == 'ki':
            if self.controller_method == 'pid':
                self.controller.ki = float(value)
            else:
                print(f"⚠ Parameter 'ki' is only valid for PID controller")
                return False

        elif param_name == 'kd':
            self.controller.kd = float(value)
        elif param_name == 'throttle_base':
            self.throttle_policy['base'] = float(value)
        elif param_name == 'throttle_min':
            self.throttle_policy['min'] = float(value)
        elif param_name == 'steer_threshold':
            self.throttle_policy['steer_threshold'] = float(value)
        elif param_name == 'steer_max':
            self.throttle_policy['steer_max'] = float(value)

        return True
