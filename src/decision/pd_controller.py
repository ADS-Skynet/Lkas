"""
PD Controller - Computes steering corrections based on lane metrics.

SINGLE RESPONSIBILITY: Calculate steering control using PD (Proportional-Derivative) control.

For C++ developers:
    This is a control algorithm implementation.
    PD control is common in robotics and autonomous systems.

    Formula: output = Kp * error + Kd * derivative(error)
"""

from lkas.detection.core.models import LaneMetrics, Lane


class PDController:
    """
    PD (Proportional-Derivative) controller for steering correction.

    CONTROL THEORY:
        - P (Proportional): Reacts to current error
        - D (Derivative): Reacts to rate of change of error

        steering = -(Kp * lateral_offset + Kd * heading_angle)

    KEYWORD EXPLANATION:
        class PDController: - Class definition
        def __init__(self, kp, kd): - Constructor with parameters
        self.kp - Instance variable (gains stored per instance)
    """

    def __init__(self, kp: float = 0.5, kd: float = 0.1):
        """
        Initialize PD controller.

        Args:
            kp: Proportional gain (how much to react to lateral offset)
            kd: Derivative gain (how much to react to heading angle)

        TUNING TIPS:
            - Higher Kp: Stronger reaction to being off-center
            - Higher Kd: Stronger reaction to angle misalignment
            - Start with Kp=0.5, Kd=0.1 and adjust
        """
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain

        # State tracking for derivative term
        self.prev_error: float | None = None

    def compute_steering(self, metrics: LaneMetrics) -> float | None:
        """
        Compute steering correction from lane metrics.

        CONTROL LAW:
            steering = -(Kp * lateral_offset + Kd * heading_angle)

        Negative sign because:
            - If vehicle is left of center (positive offset) → steer right (negative)
            - If vehicle is right of center (negative offset) → steer left (positive)

        Args:
            metrics: Lane analysis metrics containing:
                - lateral_offset_normalized: How far off-center (-1 to 1)
                - heading_angle_deg: Angle relative to lane direction

        Returns:
            Steering correction in range [-1, 1] or None if insufficient data
                -1.0 = full left
                 0.0 = straight
                +1.0 = full right

        PYTHON KEYWORDS:
            -> Optional[float]: Returns float or None
            Like: float* compute_steering() in C++ (nullable)
        """
        # Check if we have enough data
        if metrics.lateral_offset_normalized is None:
            return None

        if not metrics.has_both_lanes:
            # Can't compute good steering without both lanes
            return None

        # Proportional term: lateral offset
        # How far are we from lane center?
        p_term = metrics.lateral_offset_normalized

        # Derivative term: heading angle
        # What direction are we pointing?
        d_term = 0.0
        if metrics.heading_angle_deg is not None:
            # Normalize heading angle to [-1, 1] range
            # Assume max heading is ±30 degrees
            max_heading = 30.0
            d_term = metrics.heading_angle_deg / max_heading
            d_term = max(-1.0, min(1.0, d_term))  # Clamp to [-1, 1]

        # PD control law
        # Negative sign: offset right → steer left, offset left → steer right
        steering = -(self.kp * p_term + self.kd * d_term)

        # Clamp to valid steering range [-1, 1]
        steering = max(-1.0, min(1.0, steering))

        return steering

    def compute_steering_simple(self, left_lane: Lane | None,
                                right_lane: Lane | None,
                                image_width: int) -> float | None:
        """
        Simplified steering computation directly from lane lines.

        ALTERNATIVE METHOD: Use lane endpoints instead of full metrics
        Good for quick prototyping or when LaneAnalyzer is not available.

        Args:
            left_lane: Left lane line
            right_lane: Right lane line
            image_width: Image width for calculating center

        Returns:
            Steering correction [-1, 1] or None
        """
        if not left_lane or not right_lane:
            return None

        # Calculate lane center at bottom of image
        lane_center_x = (left_lane.x1 + right_lane.x1) / 2.0

        # Vehicle center (assume camera is centered)
        vehicle_center_x = image_width / 2.0

        # Lateral offset
        offset_pixels = vehicle_center_x - lane_center_x

        # Normalize to [-1, 1]
        # Assume max offset is half the image width
        max_offset = image_width / 2.0
        offset_normalized = offset_pixels / max_offset

        # Simple proportional control
        steering = -self.kp * offset_normalized

        # Clamp
        steering = max(-1.0, min(1.0, steering))

        return steering

    def set_gains(self, kp: float, kd: float):
        """
        Update controller gains.

        Useful for:
            - Runtime tuning
            - Adaptive control
            - Different driving modes (highway vs city)

        Args:
            kp: New proportional gain
            kd: New derivative gain
        """
        self.kp = kp
        self.kd = kd

    def get_gains(self) -> tuple:
        """
        Get current controller gains.

        Returns:
            Tuple of (kp, kd)
        """
        return (self.kp, self.kd)


# =============================================================================
# PD CONTROL EXPLANATION (for those new to control theory)
# =============================================================================
"""
PD CONTROL IN SIMPLE TERMS:

Imagine you're steering a car to stay in your lane:

P (Proportional):
    - "How far am I from the center?"
    - If far from center → steer hard
    - If close to center → steer gently
    - Like: steering_amount = distance_from_center * Kp

D (Derivative):
    - "Am I pointing the right direction?"
    - If angled away from lane → steer to realign
    - If parallel to lane → don't add extra steering
    - Like: steering_amount += angle_from_lane * Kd

COMBINED:
    steering = -(Kp * offset + Kd * angle)

WHY NEGATIVE?
    - If you're left of center (positive offset) → need to steer RIGHT (negative)
    - If you're right of center (negative offset) → need to steer LEFT (positive)

C++ ANALOGY:
    float PDController::computeSteering(float offset, float angle) {
        float p_term = this->kp * offset;
        float d_term = this->kd * angle;
        float steering = -(p_term + d_term);
        return clamp(steering, -1.0f, 1.0f);
    }

TUNING:
    - Kp too high: Oscillation (zigzag driving)
    - Kp too low: Slow response (drift off-center)
    - Kd too high: Overcorrection
    - Kd too low: No damping (bouncy)

TYPICAL VALUES:
    - Highway: Kp=0.3, Kd=0.1 (gentle steering)
    - City: Kp=0.5, Kd=0.2 (responsive steering)
    - Sharp turns: Kp=0.8, Kd=0.3 (aggressive steering)
"""


if __name__ == "__main__":
    # Example usage
    print("Testing PDController...")

    from lkas.detection.core.models import LaneMetrics, LaneDepartureStatus

    # Create controller
    controller = PDController(kp=0.5, kd=0.1)
    print(f"Controller gains: Kp={controller.kp}, Kd={controller.kd}")

    # Test case 1: Vehicle left of center
    metrics1 = LaneMetrics(
        lateral_offset_normalized=0.3,  # 30% left of center
        heading_angle_deg=5.0,           # Pointing 5° left
        has_both_lanes=True
    )
    steering1 = controller.compute_steering(metrics1)
    print(f"\nTest 1 (left of center):")
    print(f"  Offset: {metrics1.lateral_offset_normalized}")
    print(f"  Heading: {metrics1.heading_angle_deg}°")
    print(f"  Steering: {steering1:.3f} (negative = steer right)")

    # Test case 2: Vehicle right of center
    metrics2 = LaneMetrics(
        lateral_offset_normalized=-0.3,  # 30% right of center
        heading_angle_deg=-5.0,           # Pointing 5° right
        has_both_lanes=True
    )
    steering2 = controller.compute_steering(metrics2)
    print(f"\nTest 2 (right of center):")
    print(f"  Offset: {metrics2.lateral_offset_normalized}")
    print(f"  Heading: {metrics2.heading_angle_deg}°")
    print(f"  Steering: {steering2:.3f} (positive = steer left)")

    # Test case 3: Centered
    metrics3 = LaneMetrics(
        lateral_offset_normalized=0.0,
        heading_angle_deg=0.0,
        has_both_lanes=True
    )
    steering3 = controller.compute_steering(metrics3)
    print(f"\nTest 3 (centered):")
    print(f"  Offset: {metrics3.lateral_offset_normalized}")
    print(f"  Heading: {metrics3.heading_angle_deg}°")
    print(f"  Steering: {steering3:.3f} (near zero = straight)")
