"""
Vehicle Manager

Handles vehicle spawning, control, and state management in CARLA.
"""

import carla
from typing import List
import random


class VehicleManager:
    """
    Manages vehicle in CARLA simulator.

    Responsibility: Spawn, control, and manage vehicle state.
    """

    def __init__(self, world: carla.World):
        """
        Initialize vehicle manager.

        Args:
            world: CARLA world instance
        """
        self.world = world
        self.vehicle: carla.Vehicle | None = None
        self.vehicle_type: str | None = None

        # Spawn points
        self.spawn_points: List[carla.Transform] = []
        self.current_spawn_index: int = 0  # Index of pre-defined spawn point

        # Autopilot state
        self.autopilot_enabled: bool = False

    def spawn_vehicle(
        self,
        vehicle_type: str = "vehicle.tesla.model3",
        spawn_point_index: int | None = None,
    ) -> bool:
        """
        Spawn a vehicle in the world.

        Args:
            vehicle_type: Blueprint ID for vehicle
            spawn_point_index: Specific spawn point (None for random)

        Returns:
            True if spawn successful
        """
        try:
            # Store vehicle type for later use
            self.vehicle_type = vehicle_type

            # Get blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(vehicle_type)[0]

            # Get spawn points
            self.spawn_points = self.world.get_map().get_spawn_points()
            if not self.spawn_points:
                print("✗ No spawn points available")
                return False

            # Select spawn point
            if spawn_point_index is not None:
                # Use specific index
                if spawn_point_index >= len(self.spawn_points):
                    print(f"✗ Invalid spawn point index: {spawn_point_index}")
                    return False
                spawn_point = self.spawn_points[spawn_point_index]
                self.current_spawn_index = spawn_point_index
            else:
                # Try random spawn points
                shuffled_indices = list(range(len(self.spawn_points)))
                random.shuffle(shuffled_indices)

                self.vehicle = None
                for idx in shuffled_indices:
                    spawn_point = self.spawn_points[idx]
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if self.vehicle:
                        self.current_spawn_index = idx
                        break

                if not self.vehicle:
                    print("✗ Failed to spawn vehicle at any spawn point")
                    return False

            # If specific spawn point, try to spawn
            if spawn_point_index is not None:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if not self.vehicle:
                    print(f"✗ Failed to spawn at index {spawn_point_index}")
                    return False

            print(f"✓ Vehicle spawned at spawn point {self.current_spawn_index}")
            return True

        except Exception as e:
            print(f"✗ Failed to spawn vehicle: {e}")
            return False

    def destroy_vehicle(self):
        """Destroy the current vehicle."""
        if self.vehicle:
            print("Destroying vehicle...")
            self.vehicle.destroy()
            self.vehicle = None
            self.autopilot_enabled = False

    def respawn_vehicle(self) -> bool:
        """
        Respawn vehicle at current location.

        Returns:
            True if respawn successful
        """
        if not self.vehicle or not self.vehicle_type:
            return False

        # Get current transform
        current_transform = self.vehicle.get_transform()

        # Destroy current vehicle
        self.destroy_vehicle()

        # Spawn new vehicle at same location
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(self.vehicle_type)[0]
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, current_transform)

            if self.vehicle:
                return True
            return False
        except Exception as e:
            print(f"✗ Respawn failed: {e}")
            return False

    def teleport_to_spawn_point(self, index: int | None = None) -> bool:
        """
        Teleport vehicle to a spawn point.

        Args:
            index: Spawn point index (None for next in sequence)

        Returns:
            True if teleport successful
        """
        if not self.vehicle or not self.spawn_points:
            return False

        if index is None:
            # Cycle to next spawn point
            self.current_spawn_index = (self.current_spawn_index + 1) % len(
                self.spawn_points
            )
        else:
            if index >= len(self.spawn_points):
                return False
            self.current_spawn_index = index

        spawn_point = self.spawn_points[self.current_spawn_index]
        self.vehicle.set_transform(spawn_point)
        # print(f"✓ Teleported to spawn point {self.current_spawn_index}")
        return True

    def apply_control(
        self, steering: float = 0.0, throttle: float = 0.0, brake: float = 0.0
    ):
        """
        Apply control to vehicle.

        Args:
            steering: Steering [-1, 1]
            throttle: Throttle [0, 1]
            brake: Brake [0, 1]
        """
        if not self.vehicle:
            return

        control = carla.VehicleControl()
        control.steer = max(-1.0, min(1.0, steering))
        control.throttle = max(0.0, min(1.0, throttle))
        control.brake = max(0.0, min(1.0, brake))
        self.vehicle.apply_control(control)

    def set_autopilot(self, enabled: bool):
        """
        Enable/disable autopilot.

        Args:
            enabled: Autopilot state
        """
        if self.vehicle:
            self.vehicle.set_autopilot(enabled)
            self.autopilot_enabled = enabled

    def is_autopilot_enabled(self) -> bool:
        """Check if autopilot is enabled."""
        return self.autopilot_enabled

    def get_vehicle(self) -> carla.Vehicle | None:
        """Get vehicle instance."""
        return self.vehicle

    def get_transform(self) -> carla.Transform | None:
        """Get vehicle transform."""
        if self.vehicle:
            return self.vehicle.get_transform()
        return None

    def get_velocity(self) -> carla.Vector3D | None:
        """Get vehicle velocity."""
        if self.vehicle:
            return self.vehicle.get_velocity()
        return None

    def get_speed_kmh(self) -> float:
        """Get vehicle speed in km/h."""
        if not self.vehicle:
            return 0.0

        velocity = self.vehicle.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        return speed_ms * 3.6  # Convert m/s to km/h
