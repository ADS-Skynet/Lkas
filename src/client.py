"""
Lane Keeping Assist System (LKAS)

High-level wrapper that provides complete LKAS functionality with a single, simple interface.
Encapsulates both detection and decision systems behind an easy-to-use API.

This is the highest level of abstraction - perfect for integration with vehicles or simulations.
"""

from typing import Optional
import numpy as np
from lkas.detection import DetectionClient
from lkas.decision import DecisionClient
from lkas.integration.shared_memory.messages import DetectionMessage, ControlMessage


class LKASClient:
    """
    Complete Lane Keeping Assist System.

    Ultra-simple interface that hides all complexity.
    Just send images and get control commands!

    Usage:
        # Make sure LKAS servers are running first (they create the shared memory)
        # Then initialize LKAS client
        lkas = LKASClient(
            image_shm_name="camera_feed",
            detection_shm_name="detection_results",
            control_shm_name="control_commands",
            image_shape=(600, 800, 3)
        )

        # In your vehicle loop
        while running:
            # Send image from camera
            lkas.send_image(camera_image, timestamp, frame_id)

            # Get control command
            control = lkas.get_control()

            # Apply to vehicle
            vehicle.apply(control.steering, control.throttle, control.brake)

        # Cleanup (just close connections, servers own the memory)
        lkas.close()
    """

    def __init__(
        self,
        image_shm_name: str = "camera_feed",
        detection_shm_name: str = "detection_results",
        control_shm_name: str = "control_commands",
        image_shape: tuple = (600, 800, 3),
        retry_count: int = 20,
        retry_delay: float = 0.5
    ):
        """
        Initialize Lane Keeping Assist System.

        Args:
            image_shm_name: Shared memory name for camera images
            detection_shm_name: Shared memory name for lane detections
            control_shm_name: Shared memory name for control commands
            image_shape: Camera image shape (height, width, channels)
            retry_count: Connection retry attempts (default: 20)
            retry_delay: Delay between retries in seconds (default: 0.5)
        """
        self.image_shm_name = image_shm_name
        self.detection_shm_name = detection_shm_name
        self.control_shm_name = control_shm_name
        self.image_shape = image_shape

        # Initialize detection client (bidirectional: write images, read detections)
        self._detection_client = DetectionClient(
            detection_shm_name=detection_shm_name,
            image_shm_name=image_shm_name,
            image_shape=image_shape,
            retry_count=retry_count,
            retry_delay=retry_delay
        )

        # Initialize decision client (read-only: read control commands)
        self._decision_client = DecisionClient(
            shm_name=control_shm_name,
            retry_count=retry_count,
            retry_delay=retry_delay
        )

    def send_image(self, image: np.ndarray, timestamp: float, frame_id: int) -> None:
        """
        Send camera image to LKAS system.

        Args:
            image: Camera image array (height, width, channels)
            timestamp: Image capture timestamp
            frame_id: Sequential frame identifier
        """
        self._detection_client.send_image(image, timestamp, frame_id)

    def get_detection(self, timeout: float = 1.0) -> DetectionMessage | None:
        """
        Get lane detection result (optional, for debugging/visualization).

        Args:
            timeout: Maximum wait time (not implemented yet)

        Returns:
            DetectionMessage with lane information or None
        """
        return self._detection_client.get_detection(timeout)

    def get_control(self, timeout: float = 1.0) -> ControlMessage | None:
        """
        Get control command from LKAS system.

        Args:
            timeout: Maximum wait time (not implemented yet)

        Returns:
            ControlMessage with steering, throttle, brake commands or None
        """
        return self._decision_client.get_control(timeout)

    def close(self):
        """Close all connections and cleanup resources."""
        self._detection_client.close()
        self._decision_client.close()


class LKASSimple:
    """
    Simplified LKAS interface with just two methods: send and receive.
    Even simpler than LKAS - perfect for minimal code.

    Usage:
        lkas = LKASSimple()

        while running:
            lkas.send(camera_image, timestamp, frame_id)
            control = lkas.receive()
            vehicle.apply(control.steering, control.throttle, control.brake)
    """

    def __init__(
        self,
        image_shm_name: str = "camera_feed",
        detection_shm_name: str = "detection_results",
        control_shm_name: str = "control_commands",
        image_shape: tuple = (600, 800, 3)
    ):
        """Initialize simplified LKAS with default settings."""
        self._lkas = LKASClient(
            image_shm_name=image_shm_name,
            detection_shm_name=detection_shm_name,
            control_shm_name=control_shm_name,
            image_shape=image_shape
        )

    def send(self, image: np.ndarray, timestamp: float, frame_id: int) -> None:
        """Send image to LKAS."""
        self._lkas.send_image(image, timestamp, frame_id)

    def receive(self, timeout: float = 1.0) -> ControlMessage | None:
        """Receive control from LKAS."""
        return self._lkas.get_control(timeout)

    def close(self):
        """Close LKAS."""
        self._lkas.close()
