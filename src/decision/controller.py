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
from lkas.decision.method.planner_controller import PlannerDecisionMethod, DEFAULT_MODEL_PATH


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
        # Planner-specific params (only used when controller_method == "planner")
        model_path: str | None = None,
        scenario: int = 0,
        planner_device: str = "cpu",
    ):
        """
        Initialize decision controller.

        Args:
            image_width: Camera image width
            image_height: Camera image height
            kp: Proportional gain for steering control
            ki: Integral gain for steering control (PID only)
            kd: Derivative gain for steering control
            controller_method: Controller type ('pd', 'pid', 'mpc', 'planner', etc.)
            throttle_policy: Adaptive throttle configuration dict with keys:
                - base: Base throttle value (default: 0.45)
                - min: Minimum throttle value (default: 0.18)
                - steer_threshold: Steering magnitude to start reducing throttle (default: 0.15)
                - steer_max: Maximum steering for throttle calculation (default: 0.70)
            config: Optional system configuration object
            camera_offset_x: Pixel offset of camera center from vehicle center
            model_path: Path to planner_model.pth (planner method only)
            scenario: Scenario token for planner (0 = LANE_FOLLOW)
            planner_device: Torch device for planner inference ("cpu" recommended)
        """
        self.camera_offset_x = camera_offset_x
        self.controller_method = controller_method.lower()

        # Lane analysis (CV detection path)
        self.analyzer = LaneAnalyzer(image_width=image_width, image_height=image_height)

        # Segmentation lane parser (DL detection path, hand-tuned controllers only)
        self.seg_parser = SegmentationLaneParser(
            image_width=image_width,
            image_height=image_height,
            camera_offset_x=camera_offset_x,
        )

        # ── Planner path (end-to-end neural network) ──────────────────────────
        self._planner: PlannerDecisionMethod | None = None
        self.controller: SteeringController | None = None

        if self.controller_method == "planner":
            self._planner = PlannerDecisionMethod(
                model_path=model_path or DEFAULT_MODEL_PATH,
                scenario=scenario,
                device=planner_device,
            )
        else:
            # ── Hand-tuned controller path ─────────────────────────────────────
            factory = ControllerFactory(config=config)
            controller_params = {"kp": kp, "kd": kd}
            if self.controller_method == "pid":
                controller_params["ki"] = ki
            elif self.controller_method == "pure_pursuit":
                controller_params["image_width"] = image_width
                controller_params["image_height"] = image_height
                controller_params["camera_offset_x"] = camera_offset_x

            self.controller = factory.create(
                controller_type=self.controller_method,
                **controller_params
            )

        # Adaptive throttle policy (used by hand-tuned controllers; planner outputs throttle directly)
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

        Handles three detection paths:
        - Planner (e2e): lane_grid (pre-computed) → PlannerModel → (steering, throttle)
        - DL + hand-tuned: mask → SegmentationLaneParser → LaneMetrics → PID/PD/PurePursuit
        - CV: left/right lane endpoints → LaneAnalyzer → LaneMetrics → PID/PD

        Args:
            detection: Lane detection message

        Returns:
            Control message with steering, throttle, brake commands
        """
        # ── Planner path ──────────────────────────────────────────────────────
        if self._planner is not None:
            if detection.lane_grid is not None:
                # Prefer pre-computed grid produced by the detection server
                steering, throttle = self._planner.infer(lane_grid=detection.lane_grid)
                brake = 0.0
            elif detection.segmentation_mask is not None:
                # Fallback: compute grid here (planner-e2e unavailable on detection side)
                steering, throttle = self._planner.infer(mask=detection.segmentation_mask)
                brake = 0.0
            else:
                # No lane features — hold neutral and apply light brake
                steering = 0.0
                throttle = 0.0
                brake = 0.3

            control = ControlMessage(
                steering=steering,
                throttle=throttle,
                brake=brake,
                mode=self.mode,
            )
            control.clamp_values()
            return control

        # ── Hand-tuned controller paths ───────────────────────────────────────
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

        # Compute steering from metrics
        steering = self.controller.compute_steering(metrics)

        if steering is None:
            steering = 0.0
            throttle = 0.0
            brake = 0.3
        else:
            if self.use_adaptive_throttle:
                throttle = self.compute_adaptive_throttle(steering)
            else:
                throttle = self.default_throttle
            brake = self.default_brake

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
        Update controller gains (no-op in planner mode).

        Args:
            kp: Proportional gain
            ki: Integral gain (for PID only, optional)
            kd: Derivative gain (optional)
        """
        if self.controller is None:
            return
        if self.controller_method == "pid":
            current_gains = self.controller.get_gains()
            ki_val = ki if ki is not None else current_gains[1]
            kd_val = kd if kd is not None else current_gains[2]
            self.controller.set_gains(kp, ki_val, kd_val)
        else:
            kd_val = kd if kd is not None else self.controller.get_gains()[1]
            self.controller.set_gains(kp, kd_val)

    def get_controller_gains(self) -> tuple:
        """
        Get current controller gains.

        Returns:
            Tuple of (kp, kd) for PD or (kp, ki, kd) for PID.
            Returns empty tuple in planner mode.
        """
        if self.controller is None:
            return ()
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
            - After significant disturbance
        """
        if self._planner is not None:
            self._planner.reset_state()
        if self.controller is not None:
            self.controller.reset_state()
        self.seg_parser.reset()

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

        # Planner method: no tunable gains; only scenario can be updated
        if self._planner is not None:
            if param_name == 'scenario':
                self._planner.set_scenario(int(value))
                return True
            print(f"⚠ Parameter '{param_name}' is not applicable in planner mode")
            return False

        # Map of valid parameters and their value constraints
        valid_params = {
            'kp': (0.0, 2.0),              # Proportional gain
            'ki': (0.0, 0.5),              # Integral gain (PID only)
            'kd': (0.0, 1.0),              # Derivative gain
            'lookahead_ratio': (0.1, 0.8), # Pure Pursuit lookahead (fraction of image height)
            'camera_offset_x': (-200, 200),# Camera center offset (pixels)
            'min_confidence': (0.0, 1.0),  # Lane boundary confidence threshold
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
            self.throttle_policy['base'] = float(value)
        elif param_name == 'throttle_min':
            self.throttle_policy['min'] = float(value)
        elif param_name == 'steer_threshold':
            self.throttle_policy['steer_threshold'] = float(value)
        elif param_name == 'steer_max':
            self.throttle_policy['steer_max'] = float(value)

        return True
