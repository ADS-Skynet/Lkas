"""
Communication Interface

Abstract interface for detection communication channels.
Allows swapping between ZMQ, Shared Memory, or other transport mechanisms.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IDetectionChannel(ABC):
    """
    Abstract interface for detection communication.

    Implementations: ZMQ, Shared Memory, Mock (for testing)
    """

    @abstractmethod
    def send_image(self, image: np.ndarray, metadata: dict) -> None:
        """
        Send image for detection.

        Args:
            image: RGB image array
            metadata: Additional metadata (frame_id, timestamp, etc.)
        """
        pass

    @abstractmethod
    def receive_detection(self, timeout: float) -> Optional[object]:
        """
        Receive detection result.

        Args:
            timeout: Timeout in seconds

        Returns:
            DetectionMessage or None if timeout
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close channel and cleanup resources."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if channel is connected and ready."""
        pass
