"""
CARLA Interface Module
Handles connection to CARLA simulator and camera setup for lane detection.
"""

import carla
import numpy as np
import cv2
from typing import Optional, Callable, Tuple
import queue
import weakref


class CARLAInterface:
    """Interface for connecting to CARLA and managing sensors."""

    def __init__(self, host: str = 'localhost', port: int = 2000):
        """
        Initialize CARLA connection.

        Args:
            host: CARLA server host
            port: CARLA server port
        """
        self.host = host
        self.port = port
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.image_queue = queue.Queue()

        # Store spawn points and current index for cycling
        self.spawn_points = []
        self.current_spawn_index = 0
        self.vehicle_type = None

        # Track autopilot state
        self.autopilot_enabled = False

    def connect(self) -> bool:
        """
        Connect to CARLA server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"Connected to CARLA server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            return False

    def spawn_vehicle(self, vehicle_type: str = 'vehicle.tesla.model3',
                      spawn_point_index: Optional[int] = None) -> bool:
        """
        Spawn a vehicle in the CARLA world.

        Args:
            vehicle_type: Blueprint name of the vehicle
            spawn_point_index: Specific spawn point index (None for random try)

        Returns:
            True if spawn successful, False otherwise
        """
        try:
            # Store vehicle type for respawning
            self.vehicle_type = vehicle_type

            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(vehicle_type)[0]

            # Get and store spawn points
            self.spawn_points = self.world.get_map().get_spawn_points()
            if not self.spawn_points:
                print("No spawn points available on this map")
                return False

            # Try to spawn at specified index or try multiple random points
            if spawn_point_index is not None:
                # Use specific spawn point
                if spawn_point_index >= len(self.spawn_points):
                    print(f"Invalid spawn point index: {spawn_point_index} (max: {len(self.spawn_points)-1})")
                    return False
                spawn_point = self.spawn_points[spawn_point_index]
                self.current_spawn_index = spawn_point_index
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is None:
                    print(f"Failed to spawn at index {spawn_point_index} (collision)")
                    return False
            else:
                # Try random spawn points until one works
                import random
                shuffled_indices = list(range(len(self.spawn_points)))
                random.shuffle(shuffled_indices)
                self.vehicle = None
                for idx in shuffled_indices:
                    spawn_point = self.spawn_points[idx]
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if self.vehicle is not None:
                        self.current_spawn_index = idx
                        print(f"Vehicle spawned at spawn point {idx}: {vehicle_type}")
                        return True
                print("Failed to spawn vehicle: All spawn points occupied")
                return False

            print(f"Vehicle spawned at spawn point {self.current_spawn_index}: {vehicle_type}")
            return True
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            return False

    def setup_camera(self,
                     image_width: int = 800,
                     image_height: int = 600,
                     fov: float = 90.0,
                     position: Tuple[float, float, float] = (2.0, 0.0, 1.5),
                     rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> bool:
        """
        Attach a camera sensor to the vehicle.

        Args:
            image_width: Camera image width
            image_height: Camera image height
            fov: Field of view in degrees
            position: Camera position relative to vehicle (x, y, z)
            rotation: Camera rotation (pitch, yaw, roll)

        Returns:
            True if setup successful, False otherwise
        """
        if self.vehicle is None:
            print("No vehicle to attach camera to")
            return False

        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # Set camera attributes
            camera_bp.set_attribute('image_size_x', str(image_width))
            camera_bp.set_attribute('image_size_y', str(image_height))
            camera_bp.set_attribute('fov', str(fov))

            # Set camera transform
            camera_transform = carla.Transform(
                carla.Location(x=position[0], y=position[1], z=position[2]),
                carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2])
            )

            # Spawn and attach camera
            self.camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.vehicle
            )

            # Start camera listening
            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: CARLAInterface._process_image(weak_self, image))

            print(f"Camera attached: {image_width}x{image_height}, FOV: {fov}")
            return True
        except Exception as e:
            print(f"Failed to setup camera: {e}")
            return False

    @staticmethod
    def _process_image(weak_self, image):
        """Process incoming camera image."""
        self = weak_self()
        if not self:
            return

        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # Convert BGRA to RGB

        # Put image in queue
        self.image_queue.put(array)

    def get_latest_image(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the latest camera image.

        Args:
            timeout: Maximum time to wait for image

        Returns:
            Image as numpy array or None if timeout
        """
        try:
            return self.image_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_vehicle_autopilot(self, enabled: bool = True):
        """
        Enable or disable vehicle autopilot.

        Args:
            enabled: Whether to enable autopilot
        """
        if self.vehicle:
            self.vehicle.set_autopilot(enabled)
            self.autopilot_enabled = enabled
            status = "enabled" if enabled else "disabled"
            print(f"Autopilot {status}")

    def is_autopilot_enabled(self) -> bool:
        """
        Check if autopilot is currently enabled.

        Returns:
            True if autopilot is enabled, False otherwise
        """
        return self.autopilot_enabled

    def get_vehicle_transform(self) -> Optional[carla.Transform]:
        """Get current vehicle transform."""
        if self.vehicle:
            return self.vehicle.get_transform()
        return None

    def get_vehicle_velocity(self) -> Optional[carla.Vector3D]:
        """Get current vehicle velocity."""
        if self.vehicle:
            return self.vehicle.get_velocity()
        return None

    def get_vehicle_telemetry(self) -> dict:
        """
        Get comprehensive vehicle telemetry data.

        Returns:
            Dictionary with vehicle telemetry including speed, position, rotation
        """
        if not self.vehicle:
            return {}

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        location = transform.location
        rotation = transform.rotation

        # Calculate speed in km/h
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6

        return {
            'speed_kmh': speed_kmh,
            'speed_ms': speed_ms,
            'position': (location.x, location.y, location.z),
            'rotation': (rotation.pitch, rotation.yaw, rotation.roll),
            'velocity': (velocity.x, velocity.y, velocity.z),
        }

    def teleport_to_spawn_point(self, spawn_index: Optional[int] = None) -> bool:
        """
        Teleport vehicle to a specific spawn point.

        Args:
            spawn_index: Index of spawn point (None for next in sequence)

        Returns:
            True if teleport successful, False otherwise
        """
        if not self.vehicle:
            print("No vehicle to teleport")
            return False

        if not self.spawn_points:
            print("No spawn points available")
            return False

        try:
            # Determine target spawn index
            if spawn_index is not None:
                if spawn_index < 0 or spawn_index >= len(self.spawn_points):
                    print(f"Invalid spawn index: {spawn_index} (max: {len(self.spawn_points)-1})")
                    return False
                target_index = spawn_index
            else:
                # Go to next spawn point
                target_index = (self.current_spawn_index + 1) % len(self.spawn_points)

            # Get spawn point transform
            spawn_transform = self.spawn_points[target_index]

            # Teleport vehicle
            self.vehicle.set_transform(spawn_transform)

            # Reset vehicle physics
            self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

            self.current_spawn_index = target_index
            location = spawn_transform.location
            print(f"Vehicle teleported to spawn point {target_index}: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")
            return True

        except Exception as e:
            print(f"Failed to teleport vehicle: {e}")
            return False

    def respawn_vehicle(self) -> bool:
        """
        Respawn vehicle at next available spawn point.
        This is useful if the vehicle gets stuck.

        Returns:
            True if respawn successful, False otherwise
        """
        if not self.vehicle or not self.vehicle_type:
            print("Cannot respawn: vehicle not initialized")
            return False

        try:
            # Store camera settings if camera exists
            camera_config = None
            if self.camera:
                # Camera will be recreated with same settings
                camera_config = {
                    'width': int(self.camera.attributes['image_size_x']),
                    'height': int(self.camera.attributes['image_size_y']),
                    'fov': float(self.camera.attributes['fov'])
                }
                # Note: We lose camera transform, but main.py will reset it

                # Stop and destroy camera
                self.camera.stop()
                self.camera.destroy()
                self.camera = None

            # Store autopilot state
            autopilot_enabled = self.autopilot_enabled

            # Destroy old vehicle
            self.vehicle.destroy()
            self.vehicle = None

            # Clear image queue
            while not self.image_queue.empty():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    break

            # Spawn at next spawn point
            next_index = (self.current_spawn_index + 1) % len(self.spawn_points)

            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(self.vehicle_type)[0]

            # Try to spawn at next spawn point
            spawn_point = self.spawn_points[next_index]
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if self.vehicle is None:
                # If failed, try other spawn points
                for offset in range(1, len(self.spawn_points)):
                    idx = (self.current_spawn_index + offset) % len(self.spawn_points)
                    spawn_point = self.spawn_points[idx]
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if self.vehicle is not None:
                        next_index = idx
                        break

            if self.vehicle is None:
                print("Failed to respawn vehicle: All spawn points occupied")
                return False

            self.current_spawn_index = next_index

            # Restore autopilot
            if autopilot_enabled:
                self.vehicle.set_autopilot(True)

            location = spawn_point.location
            print(f"Vehicle respawned at spawn point {next_index}: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")

            # Note: Camera needs to be reattached by main.py
            return True

        except Exception as e:
            print(f"Failed to respawn vehicle: {e}")
            return False

    def cleanup(self):
        """Clean up all spawned actors."""
        print("Cleaning up CARLA actors...")
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        print("Cleanup complete")


if __name__ == "__main__":
    # Example usage
    carla_interface = CARLAInterface()

    try:
        # Connect to CARLA
        if not carla_interface.connect():
            exit(1)

        # Spawn vehicle
        if not carla_interface.spawn_vehicle():
            exit(1)

        # Setup camera
        if not carla_interface.setup_camera():
            exit(1)

        # Enable autopilot for testing
        carla_interface.set_vehicle_autopilot(True)

        # Capture and display images
        print("Press Ctrl+C to stop...")
        while True:
            image = carla_interface.get_latest_image()
            if image is not None:
                cv2.imshow('CARLA Camera', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        carla_interface.cleanup()
        cv2.destroyAllWindows()
