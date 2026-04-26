"""
Decision Controller

Main controller that receives detection results and generates control commands.
"""

from lkas.integration.shared_memory.messages import (
    DetectionMessage,
    ControlMessage,
    ControlMode,
)
from lkas.decision.segmentation_lane_parser import SegmentationLaneParser
from lkas.decision.core.factory import ControllerFactory
from lkas.decision.core.interfaces import SteeringController


class DecisionController:
    """
    Decision controller for lane keeping.

    Responsibility:
    - Receive DL lane detection results
    - Parse segmentation mask into lane polynomial boundaries
    - Compute control commands (steering, throttle, brake)
    - Generate control messages for CARLA module
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        kp: float = 0.8,
        kd: float = 0.15,
        controller_method: str = "pure_pursuit",
        throttle_policy: dict | None = None,
        config=None,
        camera_offset_x: int = 0,
    ):
        """
        Initialize decision controller.

        Args:
            image_width: Camera image width
            image_height: Camera image height
            kp: Main steering gain
            kd: Heading angle correction gain
            controller_method: Controller type (currently only 'pure_pursuit')
            throttle_policy: Adaptive throttle configuration dict with keys:
                - base: Base throttle value (default: 0.15)
                - min: Minimum throttle value (default: 0.05)
                - steer_threshold: Steering magnitude to start reducing throttle (default: 0.15)
                - steer_max: Maximum steering for throttle calculation (default: 0.70)
            config: Optional system configuration object
            camera_offset_x: Pixel offset of camera center from vehicle center
        """
        self.camera_offset_x = camera_offset_x

        # Segmentation lane parser (DL detection path)
        self.seg_parser = SegmentationLaneParser(
            image_width=image_width,
            image_height=image_height,
            camera_offset_x=camera_offset_x,
        )

        # Steering control
        self.controller_method = controller_method.lower()
        factory = ControllerFactory(config=config)

        controller_params = {
            "kp": kp,
            "kd": kd,
            "image_width": image_width,
            "image_height": image_height,
            "camera_offset_x": camera_offset_x,
        }

        self.controller: SteeringController = factory.create(
            controller_type=self.controller_method,
            **controller_params
        )

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

        Args:
            steering: Steering value in range [-1, 1]

        Returns:
            Throttle value in range [throttle_min, throttle_base]
        """
        abs_steering = abs(steering)
        policy = self.throttle_policy

        if abs_steering <= policy["steer_threshold"]:
            return policy["base"]

        steer_range = policy["steer_max"] - policy["steer_threshold"]
        steer_delta = abs_steering - policy["steer_threshold"]
        t = max(0.0, min(1.0, steer_delta / max(1e-6, steer_range)))

        throttle_range = policy["base"] - policy["min"]
        throttle = policy["base"] - (throttle_range * t)

        return max(policy["min"], min(policy["base"], throttle))

    def process_detection(self, detection: DetectionMessage) -> ControlMessage:
        """
        Process detection results and generate control commands.

        Args:
            detection: Lane detection message with segmentation mask

        Returns:
            Control message with steering, throttle, brake commands
        """
        metrics = self.seg_parser.parse(detection.segmentation_mask)

        if hasattr(self.controller, 'set_path'):
            self.controller.set_path(self.seg_parser.get_center_poly())

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

        left_poly = None
        right_poly = None
        center_poly = None
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

    def reset_state(self):
        """Reset controller and parser state."""
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
        if param_name == 'reset':
            self.reset_state()
            return True

        valid_params = {
            'kp': (0.0, 2.0),
            'kd': (0.0, 1.0),
            'lookahead_ratio': (0.1, 0.8),
            'camera_offset_x': (-200, 200),
            'min_confidence': (0.0, 1.0),
            'throttle_base': (0.0, 1.0),
            'throttle_min': (0.0, 1.0),
            'steer_threshold': (0.0, 1.0),
            'steer_max': (0.0, 1.0),
        }

        if param_name not in valid_params:
            print(f"⚠ Unknown parameter: {param_name}")
            return False

        min_val, max_val = valid_params[param_name]
        if not (min_val <= value <= max_val):
            print(f"⚠ Value {value} out of range [{min_val}, {max_val}] for {param_name}")
            return False

        if param_name == 'kp':
            self.controller.kp = float(value)
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
