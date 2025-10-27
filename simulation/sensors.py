"""
Camera Sensor Manager

Handles camera sensor setup and image capture in CARLA.
"""

import carla
import numpy as np
import weakref
from typing import Optional, Callable
import queue


class CameraSensor:
    """
    Manages camera sensor for lane detection.

    Responsibility: Setup camera, capture images, and provide them to detection module.
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        """
        Initialize camera sensor.

        Args:
            world: CARLA world instance
            vehicle: Vehicle to attach camera to
        """
        self.world = world
        self.vehicle = vehicle
        self.camera: Optional[carla.Sensor] = None
        self.image_queue = queue.Queue()

        # Camera configuration
        self.width: int = 800
        self.height: int = 600
        self.fov: float = 90.0

        # Latest image
        self.latest_image: Optional[np.ndarray] = None
        self.frame_count: int = 0

    def setup_camera(self,
                     width: int = 800,
                     height: int = 600,
                     fov: float = 90.0,
                     position: tuple = (2.0, 0.0, 1.5),
                     rotation: tuple = (-10.0, 0.0, 0.0)) -> bool:
        """
        Setup camera sensor on vehicle.

        Args:
            width: Image width
            height: Image height
            fov: Field of view in degrees
            position: Camera position (x, y, z) relative to vehicle
            rotation: Camera rotation (pitch, yaw, roll) in degrees

        Returns:
            True if setup successful
        """
        try:
            # Store configuration
            self.width = width
            self.height = height
            self.fov = fov

            # Get camera blueprint
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # Configure camera
            camera_bp.set_attribute('image_size_x', str(width))
            camera_bp.set_attribute('image_size_y', str(height))
            camera_bp.set_attribute('fov', str(fov))

            # Set camera transform
            camera_transform = carla.Transform(
                carla.Location(x=position[0], y=position[1], z=position[2]),
                carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2])
            )

            # Spawn camera
            self.camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.vehicle
            )

            # Start listening
            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: CameraSensor._on_image_received(weak_self, image))

            print(f"✓ Camera setup: {width}x{height}, FOV={fov}°")
            return True

        except Exception as e:
            print(f"✗ Failed to setup camera: {e}")
            return False

    @staticmethod
    def _on_image_received(weak_self, carla_image):
        """
        Callback for camera image reception.

        Args:
            weak_self: Weak reference to self
            carla_image: CARLA image data
        """
        self = weak_self()
        if not self:
            return

        # Convert CARLA image to numpy array (RGB)
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))  # BGRA
        array = array[:, :, :3]  # Drop alpha, keep BGR
        array = array[:, :, ::-1]  # Convert BGR to RGB

        # Store latest image
        self.latest_image = array
        self.frame_count += 1

        # Put in queue (non-blocking, drop old frames if queue is full)
        try:
            self.image_queue.put_nowait(carla_image)
        except queue.Full:
            pass

    def get_latest_image(self) -> Optional[np.ndarray]:
        """
        Get the latest camera image.

        Returns:
            RGB image array or None
        """
        return self.latest_image

    def destroy_camera(self):
        """Destroy camera sensor."""
        if self.camera:
            print("Destroying camera...")
            self.camera.stop()
            self.camera.destroy()
            self.camera = None

    def get_frame_count(self) -> int:
        """Get total number of frames captured."""
        return self.frame_count

    def get_camera_info(self) -> dict:
        """
        Get camera configuration info.

        Returns:
            Dictionary with camera parameters
        """
        return {
            'width': self.width,
            'height': self.height,
            'fov': self.fov,
            'frames_captured': self.frame_count
        }
