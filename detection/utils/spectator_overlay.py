"""
Spectator Overlay Module
Provides overlay information on CARLA spectator view with vehicle tracking.
"""

import carla
import numpy as np
from typing import Optional, Tuple


class SpectatorOverlay:
    """Creates overlay information in CARLA spectator view."""

    def __init__(self, world: carla.World):
        """
        Initialize spectator overlay.

        Args:
            world: CARLA world instance
        """
        self.world = world
        self.debug = world.debug

    def draw_vehicle_position(
        self,
        vehicle: carla.Actor,
        color: carla.Color = carla.Color(0, 255, 0),
        lifetime: float = 0.1,
    ):
        """
        Draw vehicle position marker in CARLA world.

        Args:
            vehicle: Vehicle actor
            color: Color for the marker
            lifetime: How long the marker persists
        """
        if vehicle is None:
            return

        # Get vehicle transform
        transform = vehicle.get_transform()
        location = transform.location

        # Draw a vertical line above the vehicle
        begin = carla.Location(x=location.x, y=location.y, z=location.z + 3.0)
        end = carla.Location(x=location.x, y=location.y, z=location.z + 6.0)
        self.debug.draw_line(
            begin, end, thickness=0.3, color=color, life_time=lifetime
        )

        # Draw a circle at the top
        self.debug.draw_point(
            carla.Location(x=location.x, y=location.y, z=location.z + 6.5),
            size=0.3,
            color=color,
            life_time=lifetime,
        )

        # Draw direction arrow
        forward = transform.get_forward_vector()
        arrow_end = carla.Location(
            x=location.x + forward.x * 4.0,
            y=location.y + forward.y * 4.0,
            z=location.z + 0.5,
        )
        self.debug.draw_arrow(
            carla.Location(x=location.x, y=location.y, z=location.z + 0.5),
            arrow_end,
            thickness=0.2,
            arrow_size=0.3,
            color=color,
            life_time=lifetime,
        )

    def draw_vehicle_info_text(
        self,
        vehicle: carla.Actor,
        additional_info: str = "",
        color: carla.Color = carla.Color(255, 255, 255),
        lifetime: float = 0.1,
    ):
        """
        Draw vehicle information text in 3D space.

        Args:
            vehicle: Vehicle actor
            additional_info: Additional text to display
            color: Text color
            lifetime: How long the text persists
        """
        if vehicle is None:
            return

        # Get vehicle data
        transform = vehicle.get_transform()
        location = transform.location
        velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # Create info text
        info_text = f"Speed: {speed_kmh:.1f} km/h\n"
        info_text += f"Pos: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})\n"
        info_text += f"Yaw: {transform.rotation.yaw:.1f}°"

        if additional_info:
            info_text += f"\n{additional_info}"

        # Draw text above vehicle
        text_location = carla.Location(
            x=location.x, y=location.y, z=location.z + 7.0
        )
        self.debug.draw_string(
            text_location, info_text, draw_shadow=True, color=color, life_time=lifetime
        )

    def draw_coordinate_grid(
        self,
        center: carla.Location,
        grid_size: int = 50,
        spacing: float = 10.0,
        lifetime: float = 0.1,
    ):
        """
        Draw a coordinate grid on the ground.

        Args:
            center: Center location of the grid
            grid_size: Number of lines in each direction
            spacing: Distance between grid lines
            lifetime: How long the grid persists
        """
        color = carla.Color(100, 100, 100, 50)
        z = center.z - 0.1  # Slightly below ground

        # Draw X lines
        for i in range(-grid_size // 2, grid_size // 2 + 1):
            y = center.y + i * spacing
            begin = carla.Location(
                x=center.x - (grid_size // 2) * spacing, y=y, z=z
            )
            end = carla.Location(x=center.x + (grid_size // 2) * spacing, y=y, z=z)
            self.debug.draw_line(begin, end, thickness=0.05, color=color, life_time=lifetime)

        # Draw Y lines
        for i in range(-grid_size // 2, grid_size // 2 + 1):
            x = center.x + i * spacing
            begin = carla.Location(
                x=x, y=center.y - (grid_size // 2) * spacing, z=z
            )
            end = carla.Location(x=x, y=center.y + (grid_size // 2) * spacing, z=z)
            self.debug.draw_line(begin, end, thickness=0.05, color=color, life_time=lifetime)

    def update_spectator_camera(
        self, vehicle: carla.Actor, follow_distance: float = 20.0, height: float = 10.0
    ):
        """
        Update spectator camera to follow vehicle.

        Args:
            vehicle: Vehicle to follow
            follow_distance: Distance behind vehicle
            height: Height above vehicle
        """
        if vehicle is None:
            return

        # Get vehicle transform
        transform = vehicle.get_transform()
        location = transform.location

        # Calculate spectator position
        forward = transform.get_forward_vector()
        spectator_location = carla.Location(
            x=location.x - forward.x * follow_distance,
            y=location.y - forward.y * follow_distance,
            z=location.z + height,
        )

        # Look at vehicle
        pitch = -30.0  # Look down
        yaw = transform.rotation.yaw

        spectator_transform = carla.Transform(
            spectator_location, carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
        )

        # Set spectator transform
        spectator = self.world.get_spectator()
        spectator.set_transform(spectator_transform)
