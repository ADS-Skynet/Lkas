"""
Decision Controller

Main controller that receives detection results and generates control commands.
Combines lane analysis with steering control logic.
"""

from lkas.integration.shared_memory.messages import (
    DetectionMessage,
    ControlMessage,
    LaneMessage,
    ControlMode,
)
from lkas.decision.lane_analyzer import LaneAnalyzer
from lkas.decision.segmentation_lane_parser import SegmentationLaneParser
from lkas.decision.core.factory import ControllerFactory
from lkas.decision.core.interfaces import SteeringController


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
        controller_method: str = "pid",
        throttle_policy: dict | None = None,
        config=None,
        camera_offset_x: int = 0,
    ):
        """
        Initialize decision controller.

        Args:
            image_width: Camera image width
            image_height: Camera image height
            kp: Proportional gain for steering control
            ki: Integral gain for steering control (PID only)
            kd: Derivative gain for steering control
            controller_method: Controller type ('pd', 'pid', 'mpc', etc.)
            throttle_policy: Adaptive throttle configuration dict with keys:
                - base: Base throttle value (default: 0.45)
                - min: Minimum throttle value (default: 0.18)
                - steer_threshold: Steering magnitude to start reducing throttle (default: 0.15)
                - steer_max: Maximum steering for throttle calculation (default: 0.70)
            config: Optional system configuration object
            camera_offset_x: Pixel offset of camera center from vehicle center
        """
        self.camera_offset_x = camera_offset_x

        # Lane analysis (CV detection path)
        self.analyzer = LaneAnalyzer(image_width=image_width, image_height=image_height)

        # Segmentation lane parser (DL detection path)
        self.seg_parser = SegmentationLaneParser(
            image_width=image_width,
            image_height=image_height,
            camera_offset_x=camera_offset_x,
        )

        # Steering control - use factory pattern for instantiation
        self.controller_method = controller_method.lower()
        factory = ControllerFactory(config=config)

        # Create controller with appropriate parameters
        controller_params = {"kp": kp, "kd": kd}
        if self.controller_method == "pid":
            controller_params["ki"] = ki
        elif self.controller_method == "pure_pursuit":
            controller_params["image_width"] = image_width
            controller_params["image_height"] = image_height
            controller_params["camera_offset_x"] = camera_offset_x

        self.controller: SteeringController = factory.create(
            controller_type=self.controller_method,
            **controller_params
        )

        # Fixed throttle levels (simple 3-level system)
        self.normal_throttle = (throttle_policy or {}).get("base", 0.4)
        self.default_brake = 0.0

        # Control mode
        self.mode = ControlMode.LANE_KEEPING

        # Last known good steering + throttle.
        # Held when lanes are not detected so the vehicle keeps moving
        # rather than braking immediately on transient lane loss.
        self._last_steering = 0.0
        self._last_throttle = self.normal_throttle
        self._hold_frames = 0         # consecutive frames with no detection
        self._hold_max = 8            # max frames to hold before returning to 0

    def process_detection(self, detection: DetectionMessage) -> ControlMessage:
        """
        Process detection results and generate control commands.

        Handles two detection paths:
        - DL detection: Parse segmentation mask via SegmentationLaneParser,
          optionally provide center path polynomial to Pure Pursuit controller
        - CV detection: Use traditional LaneAnalyzer with left/right lane endpoints

        Args:
            detection: Lane detection message

        Returns:
            Control message with steering, throttle, brake commands
        """
        if detection.detection_method == "dl" and detection.segmentation_mask is not None:
            # DL path: parse segmentation mask into polynomial lane boundaries
            metrics = self.seg_parser.parse(detection.segmentation_mask)

            # Provide center path polynomial to pure pursuit controller
            if hasattr(self.controller, 'set_path'):
                self.controller.set_path(self.seg_parser.get_center_poly())
        else:
            # CV path: use traditional lane analyzer with two-endpoint lanes
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

            metrics = self.analyzer.get_metrics(left_lane, right_lane)

        # Compute steering from metrics (works with any controller)
        steering = self.controller.compute_steering(metrics)

        # If both lanes are not detected, hold last known values.
        # steering is None only when zero lanes are detected (no lane pixels at all).
        # When one lane is visible (curves, object blocking one side), the parser
        # estimates a center poly so steering is still computed — let it through.
        if steering is None:
            self._hold_frames += 1
            if self._hold_frames > self._hold_max:
                steering = 0.0
                throttle = self.normal_throttle
            else:
                steering = self._last_steering
                throttle = self._last_throttle
            brake = self.default_brake
        else:
            self._hold_frames = 0
            throttle = self.normal_throttle
            brake = self.default_brake
            self._last_steering = steering
            self._last_throttle = throttle

        # Collect polynomial coefficients and confidence for debug overlay
        left_poly = None
        right_poly = None
        center_poly = None
        left_conf = 0.0
        right_conf = 0.0
        if detection.detection_method == "dl" and detection.segmentation_mask is not None:
            lp = self.seg_parser._left_poly
            rp = self.seg_parser._right_poly
            cp = self.seg_parser._center_poly
            if lp is not None:
                left_poly = tuple(lp)
            if rp is not None:
                right_poly = tuple(rp)
            if cp is not None:
                center_poly = tuple(cp)
            left_conf, right_conf = self.seg_parser.get_confidences()

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
            left_poly=left_poly,
            right_poly=right_poly,
            center_poly=center_poly,
            left_confidence=left_conf,
            right_confidence=right_conf,
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
        Reset controller and parser state.

        Used when:
            - User presses reset button
            - Starting a new session
            - After significant disturbance (e.g. obstacle avoidance ends)
        """
        self.controller.reset_state()
        self.seg_parser.reset()
        self._last_steering = 0.0
        self._last_throttle = self.normal_throttle
        self._hold_frames = 0

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
            'lookahead_ratio': (0.1, 0.8), # Pure Pursuit lookahead (fraction of image height)
            'camera_offset_x': (-200, 200),# Camera center offset (pixels)
            'min_confidence': (0.0, 1.0),  # Lane boundary confidence threshold
            'throttle_base': (0.0, 1.0),   # Normal throttle
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
        elif param_name == 'lookahead_ratio':
            if hasattr(self.controller, 'lookahead_ratio'):
                self.controller.lookahead_ratio = float(value)
            else:
                print(f"⚠ Parameter 'lookahead_ratio' is only valid for Pure Pursuit controller")
                return False
        elif param_name == 'camera_offset_x':
            offset = int(value)
            self.camera_offset_x = offset
            self.seg_parser.camera_offset_x = offset
            self.seg_parser.vehicle_center_x = self.seg_parser.image_width // 2 + offset
            if hasattr(self.controller, 'camera_offset_x'):
                self.controller.camera_offset_x = offset
        elif param_name == 'min_confidence':
            self.seg_parser.min_confidence = float(value)
        elif param_name == 'throttle_base':
            self.normal_throttle = float(value)
            self._last_throttle = self.normal_throttle

        return True
