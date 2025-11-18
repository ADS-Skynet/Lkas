"""
Detection Client

Bidirectional client for communicating with detection server.
Can write images to server and read detection results.

Architecture:
    Camera/Simulation Process
    ↓ writes images via client.send_image()
    DetectionServer (detection/server.py)
    ↓ writes detections
    ↓ reads via client.get_detection()
    Control/Simulation Process
"""

from typing import Optional
import numpy as np
from lkas.integration.shared_memory import (
    SharedMemoryDetectionChannel,
    SharedMemoryImageChannel
)
from lkas.integration.messages import DetectionMessage


class DetectionClient:
    """
    Bidirectional client for detection server communication.
    Can send images and receive detection results.

    Usage:
        # Full bidirectional usage (camera + control)
        client = DetectionClient(
            detection_shm_name="detection_results",
            image_shm_name="camera_feed",
            image_shape=(600, 800, 3)
        )
        client.send_image(image, timestamp=time.time(), frame_id=frame_count)
        detection = client.get_detection()
        client.close()

        # Read-only usage (control only)
        client = DetectionClient(detection_shm_name="detection_results")
        detection = client.get_detection()
        client.close()
    """

    def __init__(
        self,
        detection_shm_name: str,
        image_shm_name: Optional[str] = None,
        image_shape: Optional[tuple] = None,
        retry_count: int = 20,
        retry_delay: float = 0.5
    ):
        """
        Initialize detection client.

        Args:
            detection_shm_name: Name of detection shared memory (for reading results)
            image_shm_name: Name of image shared memory (optional, for writing images)
            image_shape: Image shape (height, width, channels) - required if image_shm_name provided
            retry_count: Number of retry attempts for connection (default: 20)
            retry_delay: Delay between retries in seconds (default: 0.5)
        """
        self.detection_shm_name = detection_shm_name
        self.image_shm_name = image_shm_name
        self.image_shape = image_shape

        # Connect to detection shared memory (reader) with retry
        self._detection_channel = SharedMemoryDetectionChannel(
            name=detection_shm_name,
            create=False,
            retry_count=retry_count,
            retry_delay=retry_delay
        )

        # Optionally connect to image channel (writer) if image_shm_name provided
        self._image_channel = None
        if image_shm_name is not None:
            if image_shape is None:
                raise ValueError("image_shape is required when image_shm_name is provided")

            self._image_channel = SharedMemoryImageChannel(
                name=image_shm_name,
                shape=image_shape,
                create=False,  # Connect to existing shared memory created by detection server
                retry_count=retry_count,
                retry_delay=retry_delay
            )

    def send_image(self, image: np.ndarray, timestamp: float, frame_id: int) -> None:
        """
        Send image to detection server.

        Args:
            image: Image array to send
            timestamp: Image timestamp
            frame_id: Frame identifier

        Raises:
            RuntimeError: If image channel was not initialized (image_shm_name not provided)
        """
        if self._image_channel is None:
            raise RuntimeError(
                "Cannot send image: client was not initialized with image_shm_name. "
                "Provide image_shm_name and image_shape during initialization."
            )
        self._image_channel.write(image, timestamp=timestamp, frame_id=frame_id)

    def get_detection(self, timeout: float = 1.0) -> Optional[DetectionMessage]:
        """
        Get latest detection results.

        Args:
            timeout: Maximum wait time (not implemented yet)

        Returns:
            DetectionMessage or None
        """
        return self._detection_channel.read()

    def close(self):
        """Close all connections."""
        self._detection_channel.close()
        if self._image_channel is not None:
            self._image_channel.close()
