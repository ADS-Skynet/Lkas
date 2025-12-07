"""
PID Controller - Computes steering corrections based on lane metrics.

SINGLE RESPONSIBILITY: Calculate steering control using PID (Proportional-Integrative-Derivative) control.

For C++ developers:
    This is a control algorithm implementation.
    PID control is common in robotics and autonomous systems.

    Formula: output = Kp * error + Kd * derivative(error)
"""

from skynet_common.types.models import LaneMetrics, Lane
from lkas.decision.core.interfaces import SteeringController
import time


class PIDController(SteeringController):
    """
    PID (Proportional-Integral-Derivative) controller for steering correction.

    CONTROL THEORY:
        - P (Proportional): Reacts to current error
        - I (Integral): Reacts to accumulated error
        - D (Derivative): Reacts to rate of change of error

        steering = -(Kp * lateral_offset + Ki * integral_error + Kd * heading_angle)

    KEYWORD EXPLANATION:
        class PIDController: - Class definition
        def __init__(self, kp, ki, kd): - Constructor with parameters
        self.kp, self.ki, self.kd - Instance variables (gains stored per instance)
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.1, dt: float = 0.1):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain (how much to react to lateral offset)
            ki: Integral gain (how much to accumulate the error over time)
            kd: Derivative gain (how much to react to heading angle)
        """
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.dt = dt  # Time step for integral and derivative calculations
        # CORRECTED VERSION: Remove dt parameter entirely as it's unused
        # def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.1):
        # Remove: self.dt = dt

        # State tracking for derivative and integral terms
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

    def update(self,  left_lane: Lane | None,
               right_lane: Lane | None,
               image_width: int) -> float | None:
        """
        Update the PID controller with the current error and return the control signal.
        Deprecated method - use compute_steering instead.
        It's in here for archiving purposes.

        error: The current error (offset_normalized)
        """
        # Calculate lateral offset
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
        error = offset_normalized

        # Proportional term
        p_term = self.kp * error

        # Integral term
        # ISSUE: Uses fixed self.dt instead of actual time delta
        self.integral += error * self.dt
        # CORRECTED VERSION:
        # current_time = time.time()
        # dt = current_time - self.prev_time
        # self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        # ISSUE: Uses fixed self.dt instead of actual time delta
        d_term = self.kd * (error - self.prev_error) / self.dt
        # CORRECTED VERSION:
        # d_term = self.kd * (error - self.prev_error) / dt

        # Update previous error
        self.prev_error = error
        # CORRECTED VERSION: Also update time
        # self.prev_time = current_time

        # Total control signal (steering adjustment)
        control_signal = p_term + i_term + d_term

        return control_signal

    def compute_steering(self, metrics: LaneMetrics) -> float | None:
        """
        Compute steering correction from lane metrics using PID control.

        CONTROL LAW:
            steering = -(Kp * lateral_offset + Ki * integral_error + Kd * heading_angle)

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
        """
        if metrics.lateral_offset_normalized is None:
            return None

        if not metrics.has_both_lanes:
            # Can't compute good steering without both lanes
            return None

        # Get current time for time delta calculation
        current_time = time.time()
        dt = current_time - self.prev_time  # Time difference in seconds

        # Proportional term: lateral offset
        p_term = metrics.lateral_offset_normalized

        # Integral term: accumulate error over time
        self.integral += p_term * dt
        i_term = self.ki * self.integral

        # Derivative term: heading angle
        error = p_term
        # ISSUE 1: Duplicate assignment - d_term set to 0.0 then immediately overwritten
        d_term = 0.0
        # ISSUE 2: Uses fixed self.dt instead of calculated dt variable
        d_term = self.kd * (error - self.prev_error) / self.dt
        # CORRECTED VERSION:
        # d_term = self.kd * (error - self.prev_error) / dt

        # d_term = max(-1.0, min(1.0, d_term))  # Clamp to [-1, 1]
        # Update previous error
        self.prev_error = error

        # ISSUE 3: Commented-out alternative derivative implementation - should be removed
        # d_term = 0.0
        # if metrics.heading_angle_deg is not None:
        #    max_heading = 30.0  # Max heading to normalize angle
        #    d_term = metrics.heading_angle_deg / max_heading
        #    d_term = max(-1.0, min(1.0, d_term))  # Clamp to [-1, 1]
        # d_term *= self.kd

        # PID control law
        steering = -((self.kp * p_term) + i_term + d_term)

        # Clamp steering to valid range [-1, 1]
        steering = max(-1.0, min(1.0, steering))

        # Update previous error and time for the next iteration
        # ISSUE 4: Redundant assignment - self.prev_error already set above on line 158
        self.prev_error = p_term
        # CORRECTED VERSION: Remove the line above
        self.prev_time = current_time

        return steering

    def set_gains(self, kp: float, ki: float, kd: float):
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
        self.ki = ki
        self.kd = kd

    def get_gains(self) -> tuple:
        """
        Get current controller gains.

        Returns:
            Tuple of (kp, kd)
        """
        return (self.kp, self.ki, self.kd)

    def reset_state(self):
        """
        Reset PID error accumulation and derivative state.

        Used when:
            - Resetting the system
            - Starting a new driving session
            - After significant disturbance

        Resets:
            - integral: Error accumulation to 0
            - prev_error: Previous error to 0
            - prev_time: Time tracking to current time
        """
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def get_name(self) -> str:
        """
        Return the name of this controller.

        Returns:
            Controller name
        """
        return "PID Controller"

    def get_parameters(self) -> dict:
        """
        Return current controller parameters as dict.

        Returns:
            Dictionary of parameter names and values
        """
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
        }

    def update_parameter(self, name: str, value: float) -> bool:
        """
        Update a single controller parameter.

        Args:
            name: Parameter name ('kp', 'ki', or 'kd')
            value: New parameter value

        Returns:
            True if parameter was updated successfully, False otherwise
        """
        if name == "kp":
            self.kp = float(value)
            return True
        elif name == "ki":
            self.ki = float(value)
            return True
        elif name == "kd":
            self.kd = float(value)
            return True
        return False

    def get_state(self) -> dict:
        """
        Get current internal state of the controller.

        Returns:
            Dictionary of state variables
        """
        return {
            "integral": self.integral,
            "prev_error": self.prev_error,
            "prev_time": self.prev_time,
        }


# =============================================================================
# PID CONTROL EXPLANATION (for those new to control theory)
# =============================================================================
"""
PID CONTROL IN SIMPLE TERMS:

P (Proportional):
    - "How far am I from the center?"
    - If far from center → steer hard
    - If close to center → steer gently

I (Integral):
    - "Am I consistently off-center?"
    - If the car has been drifting for a while → gradually steer more
    - Helps correct long-term errors that P alone can't fix

D (Derivative):
    - "Am I turning too fast?"
    - If the car is moving quickly from left to right (or vice versa), slow it down
    - Helps to prevent oscillations or over-correction

COMBINED:
    steering = -(Kp * offset + Ki * accumulated_error + Kd * rate_of_change)

TUNING:
    - Kp too high: Oscillation (zigzag driving)
    - Kp too low: Slow response (drift off-center)
    - Ki too high: Integral wind-up (slow correction)
    - Ki too low: Lack of long-term correction
    - Kd too high: Overcorrection and jittery behavior
    - Kd too low: Lack of damping (bouncy)

"""


if __name__ == "__main__":
    # Example usage of the PID controller
    print("Testing PIDController...")

    # Create PID controller
    controller = PIDController(kp=0.5, ki=0.01, kd=0.1)
    print(
        f"Controller gains: Kp={controller.kp}, Ki={controller.ki}, Kd={controller.kd}")

    # Test case: Vehicle left of center
    metrics1 = LaneMetrics(
        lateral_offset_normalized=0.3,  # 30% left of center
        heading_angle_deg=5.0,           # Pointing 5° left
        has_both_lanes=True
    )
    steering1 = controller.compute_steering(metrics1)
    print(f"\nTest 1 (left of center): Steering = {steering1:.3f}")

    # Test case: Vehicle right of center
    # ISSUE: Incomplete line - will cause runtime error
    metrics2 = Lane
    # CORRECTED VERSION: Remove the line above
    metrics2 = LaneMetrics(
        lateral_offset_normalized=-0.2,  # 20% right of center
        heading_angle_deg=-3.0,          # Pointing 3° right
        has_both_lanes=True
    )
    steering2 = controller.compute_steering(metrics2)
    print(f"\nTest 2 (right of center): Steering = {steering2:.3f}")

    # Test case: Vehicle centered
    metrics3 = LaneMetrics(
        lateral_offset_normalized=0.0,   # Centered
        heading_angle_deg=0.0,           # Pointing straight
        has_both_lanes=True
    )
    steering3 = controller.compute_steering(metrics3)
    print(f"\nTest 3 (centered): Steering = {steering3:.3f}")
    print("\nPIDController tests completed.")