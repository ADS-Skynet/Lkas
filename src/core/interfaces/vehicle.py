"""
Vehicle Controller Interface

Abstract interface for vehicle control.
Allows swapping between CARLA, other simulators, or real vehicles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VehicleTelemetry:
    """Vehicle telemetry data."""
    speed_kmh: float
    position: Optional[Tuple[float, float, float]] = None
    rotation: Optional[Tuple[float, float, float]] = None
    acceleration: Optional[Tuple[float, float, float]] = None


class IVehicleController(ABC):
    """
    Abstract interface for vehicle control.

    Implementations: CARLA, Mock (for testing), Real vehicle
    """

    @abstractmethod
    def apply_control(self, steering: float, throttle: float, brake: float) -> None:
        """
        Apply control to vehicle.

        Args:
            steering: Steering angle [-1.0, 1.0]
            throttle: Throttle [0.0, 1.0]
            brake: Brake [0.0, 1.0]
        """
        pass

    @abstractmethod
    def get_telemetry(self) -> VehicleTelemetry:
        """
        Get current vehicle telemetry.

        Returns:
            VehicleTelemetry object with current state
        """
        pass

    @abstractmethod
    def set_autopilot(self, enabled: bool) -> None:
        """Enable or disable autopilot."""
        pass

    @abstractmethod
    def is_autopilot_enabled(self) -> bool:
        """Check if autopilot is enabled."""
        pass

    @abstractmethod
    def respawn(self, spawn_point: Optional[int] = None) -> bool:
        """
        Respawn vehicle at spawn point.

        Args:
            spawn_point: Spawn point index (None for random)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Destroy vehicle and cleanup."""
        pass
