"""
Abstract base classes and interfaces for steering control system.

Defines contracts that all controller implementations must follow.
"""

from abc import ABC, abstractmethod
from skynet_common.types.models import LaneMetrics


class SteeringController(ABC):
    """
    Abstract base class for all steering controllers.

    All control implementations (PD, PID, MPC, PFC, etc.) must inherit from this class
    and implement the required methods with a consistent interface.
    """

    @abstractmethod
    def compute_steering(self, metrics: LaneMetrics) -> float | None:
        """
        Compute steering command from lane metrics.

        Args:
            metrics: Lane analysis metrics containing:
                - lateral_offset_normalized: How far off-center (-1 to 1)
                - heading_angle_deg: Angle relative to lane direction
                - has_both_lanes: Whether both lanes are detected

        Returns:
            Steering correction in range [-1, 1] or None if insufficient data
                -1.0 = full left
                 0.0 = straight
                +1.0 = full right
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of this controller.

        Returns:
            Controller name (e.g., "PD Controller", "MPC Controller")
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Return current controller parameters as dict.

        Returns:
            Dictionary of parameter names and their current values
            Example: {"kp": 0.5, "kd": 0.1}
        """
        pass

    @abstractmethod
    def update_parameter(self, name: str, value: float) -> bool:
        """
        Update a single controller parameter.

        Args:
            name: Parameter name (e.g., "kp", "kd", "horizon")
            value: New parameter value

        Returns:
            True if parameter was updated successfully, False otherwise
        """
        pass

    def reset_state(self) -> None:
        """
        Reset controller internal state.

        Optional method for stateful controllers (PID, MPC, PFC).
        Stateless controllers (PD) can use the default empty implementation.

        Called when:
            - User presses reset button
            - Starting a new session
            - After significant disturbance
        """
        pass

    def set_dt(self, dt: float) -> None:
        """
        Set time step for discrete-time controllers.

        Optional method for controllers that need explicit time step.
        Most controllers compute dt internally, so default implementation is empty.

        Args:
            dt: Time step in seconds
        """
        pass

    def get_state(self) -> dict:
        """
        Get current internal state of the controller.

        Optional method for debugging and monitoring.
        Returns empty dict by default.

        Returns:
            Dictionary of state variables
            Example (PID): {"integral": 0.0, "prev_error": 0.0}
            Example (MPC): {"predicted_trajectory": [...], "optimal_control": [...]}
        """
        return {}