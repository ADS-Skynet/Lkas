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

    def __init__(
        self, host: str = "localhost", port: int = 2000, timeout: float = 10.0
    ):
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

    def set_map(self, map_name: str):
        """
        Load a different map in CARLA.

        Args:
            map_name: Name of the map to load (e.g., 'Town01')
        """
        if self.client:
            print(f"Loading map: {map_name}...")
            self.world = self.client.load_world(map_name)
            print(f"✓ Map loaded: {map_name}")

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

    def setup_synchronous_mode(
        self, enabled: bool = True, fixed_delta_seconds: float = 0.05
    ):
        """
        Configure synchronous mode for deterministic simulation.

        Args:
            enabled: Enable synchronous mode
            fixed_delta_seconds: Fixed time step for simulation (default: 0.05s = 20 FPS)
        """
        if not self.world:
            print("✗ Cannot setup synchronous mode: not connected to world")
            return

        settings = self.world.get_settings()
        settings.synchronous_mode = enabled
        if enabled:
            settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(settings)

        mode_str = f"sync (Δt={fixed_delta_seconds}s)" if enabled else "async"
        print(f"✓ World mode set to: {mode_str}")

    def cleanup_world(self):
        """
        Remove all pedestrians and vehicles from the world.
        Useful for clean simulation environment.
        """
        if not self.world:
            print("✗ Cannot cleanup world: not connected")
            return

        actors = self.world.get_actors()

        # Remove all pedestrians
        walkers = actors.filter("*walker.pedestrian*")
        destroyed_walkers = 0
        for walker in walkers:
            try:
                walker.destroy()
                destroyed_walkers += 1
            except:
                pass

        # Remove all vehicles
        vehicles = actors.filter("*vehicle*")
        destroyed_vehicles = 0
        for vehicle in vehicles:
            try:
                vehicle.destroy()
                destroyed_vehicles += 1
            except:
                pass

        print(
            f"✓ World cleaned: removed {destroyed_walkers} pedestrians and {destroyed_vehicles} vehicles"
        )

    def set_all_traffic_lights_green(self):
        """
        Set all traffic lights to green and freeze them.
        Prevents traffic lights from interfering with autonomous driving.
        """
        if not self.world:
            print("✗ Cannot set traffic lights: not connected")
            return

        traffic_lights = self.world.get_actors().filter("*traffic_light*")
        count = 0
        for light in traffic_lights:
            try:
                light.set_state(carla.TrafficLightState.Green)
                light.freeze(True)
                count += 1
            except:
                pass

        print(f"✓ Set {count} traffic lights to GREEN and frozen")
