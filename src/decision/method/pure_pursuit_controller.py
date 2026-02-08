"""
Pure Pursuit Controller

Well-known geometric path tracking algorithm widely used in autonomous driving.
Used in ROS Navigation Stack, Autoware, Apollo, and many research platforms.

Core idea:
    Instead of reacting to the current lateral offset (like PD/PID),
    evaluate the offset at a LOOKAHEAD point ahead on the path.
    This naturally anticipates curves and produces smoother steering.

    For a straight road: lookahead offset ~ current offset -> same as PD
    For a curve: lookahead offset shifts toward the curve -> starts turning early

Algorithm (image-space adaptation):
    1. Evaluate center path polynomial at a lookahead y-position
    2. Compute lateral error at lookahead: vehicle_center - path_center
    3. Compute steering proportional to this error
    4. Add heading angle correction for damping

References:
    - Coulter, R.C. (1992) "Implementation of the Pure Pursuit Path Tracking Algorithm"
    - Snider, J.M. (2009) "Automatic Steering Methods for Autonomous Automobile Path Tracking"
    - Comma.ai OpenPilot laterald: preview-based lateral control
"""

import numpy as np
from common.types.models import LaneMetrics
from lkas.decision.core.interfaces import SteeringController


class PurePursuitController(SteeringController):
    """
    Pure Pursuit path tracking controller for lane keeping.

    Uses a lookahead point on the center path to compute steering.
    The further the lookahead, the smoother but less responsive the steering.

    When center path polynomial is available (from SegmentationLaneParser):
        Uses pure pursuit with polynomial evaluation at lookahead point.

    When only LaneMetrics available (fallback for CV detection):
        Uses proportional + heading correction on current-position metrics.
    """

    def __init__(
        self,
        gain: float = 0.8,
        lookahead_ratio: float = 0.4,
        heading_gain: float = 0.15,
        image_width: int = 1280,
        image_height: int = 720,
        camera_offset_x: int = 0,
    ):
        """
        Args:
            gain: Main steering gain (scales lateral error at lookahead)
            lookahead_ratio: Fraction of image height for lookahead distance [0.2-0.7]
                - Lower (0.2-0.3): looks closer, reacts to nearby road, tighter tracking
                - Higher (0.5-0.7): looks further, anticipates curves early, smoother
                - Default 0.5: balanced for typical road driving
            heading_gain: Gain for heading angle correction (damping term)
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
            camera_offset_x: Pixel offset of camera center from vehicle center
                             (negative = shift reference left)
        """
        self.gain = gain
        self.lookahead_ratio = lookahead_ratio
        self.heading_gain = heading_gain
        self.image_width = image_width
        self.image_height = image_height
        self.camera_offset_x = camera_offset_x

        # Center path polynomial (set by DecisionController before each compute)
        self._center_poly = None

    def set_path(self, center_poly: np.ndarray | None):
        """
        Set the center path polynomial for pure pursuit computation.

        Called by DecisionController before compute_steering() when
        DL segmentation data is available.

        Args:
            center_poly: Polynomial coefficients for x = f(y)
                         e.g. [a, b, c] for x = a*y^2 + b*y + c
        """
        self._center_poly = center_poly

    def compute_steering(self, metrics: LaneMetrics) -> float | None:
        """
        Compute steering using pure pursuit algorithm.

        When center path polynomial is available (DL detection):
            Evaluates lateral error at lookahead point on the path.

        When only LaneMetrics available (CV fallback):
            Uses proportional + heading correction at current position.

        Args:
            metrics: Lane analysis metrics

        Returns:
            Steering in [-1, 1] or None if insufficient data
        """
        if self._center_poly is not None:
            return self._compute_pure_pursuit(metrics)

        # Fallback: proportional + heading control on current-position metrics
        return self._compute_fallback(metrics)

    def _compute_pure_pursuit(self, metrics: LaneMetrics) -> float | None:
        """
        Pure pursuit steering from center path polynomial.

        Evaluate the lane center at a lookahead point and compute
        the lateral error there. This is the key difference from
        standard PD control: we steer toward where the road WILL BE,
        not where it currently IS.

        Heading is computed from the polynomial tangent at the lookahead
        point (not at the vehicle position) to avoid unreliable edge slopes.
        """
        vehicle_cx = self.image_width / 2.0 + self.camera_offset_x

        # Lookahead y-position (higher up in image = further ahead on road)
        y_la = self.image_height * (1.0 - self.lookahead_ratio)

        # Path center at lookahead point
        x_la = float(np.polyval(self._center_poly, y_la))

        # Lateral error at lookahead (positive = vehicle right of path)
        # Same sign convention as LaneAnalyzer: vehicle_center - lane_center
        lateral_error = vehicle_cx - x_la

        # Normalize by half image width to get [-1, 1] range
        error_normalized = lateral_error / (self.image_width / 2.0)
        error_normalized = float(np.clip(error_normalized, -1.0, 1.0))

        # Heading from polynomial tangent AT the lookahead point
        # Using the derivative at the lookahead avoids wild tangent slopes
        # at the bottom edge of the image where the polynomial extrapolates.
        # dx/dy < 0 at lookahead means road goes right ahead → need to steer right
        # This matches the CV convention: negative heading → steer right
        deriv_poly = np.polyder(self._center_poly)
        slope_at_la = float(np.polyval(deriv_poly, y_la))
        heading_deg = float(np.degrees(np.arctan(slope_at_la)))
        heading_term = heading_deg / 30.0
        heading_term = float(np.clip(heading_term, -1.0, 1.0))

        # Pure pursuit control law
        # Negative sign: offset right -> steer left, offset left -> steer right
        # gain and lookahead_ratio are independent:
        #   - gain controls how aggressively to steer toward the path
        #   - lookahead_ratio controls how far ahead to evaluate the error
        # No division by lookahead_ratio — in image space the lateral error
        # already scales naturally with lookahead distance on curves.
        steering = -(
            self.gain * error_normalized / self.lookahead_ratio
            + self.heading_gain * heading_term
        )
        steering = float(np.clip(steering, -1.0, 1.0))

        # Clear path after use (prevents stale data on next frame)
        self._center_poly = None

        return steering

    def _compute_fallback(self, metrics: LaneMetrics) -> float | None:
        """
        Fallback proportional control when no path polynomial is available.

        Same structure as PD controller using current-position metrics.
        Used when receiving CV detection results instead of DL segmentation.
        """
        if metrics.lateral_offset_normalized is None:
            return None

        if not (metrics.has_both_lanes or metrics.has_left_lane or metrics.has_right_lane):
            return None

        heading_term = 0.0
        if metrics.heading_angle_deg is not None:
            heading_term = metrics.heading_angle_deg / 30.0
            heading_term = float(np.clip(heading_term, -1.0, 1.0))

        steering = -(
            self.gain * metrics.lateral_offset_normalized
            + self.heading_gain * heading_term
        )
        steering = float(np.clip(steering, -1.0, 1.0))

        return steering

    # ------------------------------------------------------------------
    # SteeringController interface
    # ------------------------------------------------------------------

    @property
    def kp(self):
        """Alias for gain (compatibility with DecisionController parameter updates)."""
        return self.gain

    @kp.setter
    def kp(self, value):
        print("setting kp:", value)
        self.gain = value

    @property
    def kd(self):
        """Alias for heading_gain (compatibility with DecisionController)."""
        return self.heading_gain

    @kd.setter
    def kd(self, value):
        self.heading_gain = value

    def set_gains(self, *args):
        """Update controller gains (kp/gain, kd/heading_gain)."""
        print("setting gains:", args)
        if len(args) >= 1:
            self.gain = float(args[0])
        if len(args) >= 2:
            self.heading_gain = float(args[1])

    def get_gains(self) -> tuple:
        """Get current controller gains."""
        return (self.gain, self.heading_gain)

    def get_name(self) -> str:
        return "Pure Pursuit Controller"

    def get_parameters(self) -> dict:
        return {
            "gain": self.gain,
            "lookahead_ratio": self.lookahead_ratio,
            "heading_gain": self.heading_gain,
        }

    def update_parameter(self, name: str, value: float) -> bool:
        if name in ("gain", "kp"):
            self.gain = float(value)
            return True
        elif name == "lookahead_ratio":
            self.lookahead_ratio = float(value)
            return True
        elif name in ("heading_gain", "kd"):
            self.heading_gain = float(value)
            return True
        elif name == "camera_offset_x":
            self.camera_offset_x = int(value)
            return True
        return False

    def reset_state(self):
        """Reset controller state."""
        self._center_poly = None

    def get_state(self) -> dict:
        return {
            "has_path": self._center_poly is not None,
        }
