"""
CARLA Connection Manager

Handles connection to CARLA simulator and world access.
"""

import carla
from typing import Optional


class CARLAConnection:
    """
    Manages connection to CARLA simulator.

    Responsibility: Connect to CARLA and provide world access.
    """

    def __init__(self, host: str = 'localhost', port: int = 2000, timeout: float = 10.0):
        """
        Initialize CARLA connection manager.

        Args:
            host: CARLA server hostname
            port: CARLA server port
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to CARLA server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to CARLA at {self.host}:{self.port}...")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self._connected = True
            print(f"✓ Connected to CARLA server")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from CARLA."""
        if self.client:
            print("Disconnecting from CARLA...")
            self.client = None
            self.world = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to CARLA."""
        return self._connected and self.world is not None

    def get_world(self) -> Optional[carla.World]:
        """
        Get CARLA world.

        Returns:
            CARLA world instance or None
        """
        return self.world

    def get_blueprint_library(self) -> Optional[carla.BlueprintLibrary]:
        """
        Get CARLA blueprint library.

        Returns:
            Blueprint library or None
        """
        if not self.world:
            return None
        return self.world.get_blueprint_library()

    def get_map(self) -> Optional[carla.Map]:
        """
        Get CARLA map.

        Returns:
            Map instance or None
        """
        if not self.world:
            return None
        return self.world.get_map()

    def set_weather(self, weather: carla.WeatherParameters):
        """
        Set weather conditions.

        Args:
            weather: Weather parameters
        """
        if self.world:
            self.world.set_weather(weather)

    def get_settings(self) -> Optional[carla.WorldSettings]:
        """Get world settings."""
        if self.world:
            return self.world.get_settings()
        return None

    def apply_settings(self, settings: carla.WorldSettings):
        """Apply world settings."""
        if self.world:
            self.world.apply_settings(settings)
