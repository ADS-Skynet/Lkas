"""
Abstract base classes and interfaces for lane detection system.

Defines contracts that all implementations must follow.
"""

from abc import ABC, abstractmethod
import numpy as np

from .models import Lane, DetectionResult


class LaneDetector(ABC):
    """
    Abstract base class for all lane detectors.

    All lane detection implementations (CV, DL, etc.) must inherit from this class
    and implement the detect() method with a consistent interface.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect lanes in the given image.

        Args:
            image: RGB input image as numpy array (H, W, 3)

        Returns:
            DetectionResult containing:
                - left_lane: Left lane line or None
                - right_lane: Right lane line or None
                - debug_image: Visualization for debugging
                - processing_time_ms: Time taken for detection
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this detector."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current detector parameters as dict."""
        pass


class SensorInterface(ABC):
    """
    Abstract interface for sensor/camera input.

    Allows swapping between CARLA simulator, PiRacer camera, or recorded video.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the sensor.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def get_latest_image(self) -> np.ndarray | None:
        """
        Get the most recent image from sensor.

        Returns:
            RGB image as numpy array or None if not available
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if sensor is connected and operational.

        Returns:
            True if sensor is ready
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up sensor resources."""
        pass


class ProcessorInterface(ABC):
    """
    Abstract interface for frame processors.

    Processors handle one step in the pipeline (detection, analysis, visualization, etc.)
    """

    @abstractmethod
    def process(self, *args, **kwargs):
        """
        Process input and return output.

        Signature varies by processor type.
        """
        pass


class ControllerInterface(ABC):
    """
    Abstract interface for control algorithms.

    Generates steering/throttle commands based on lane metrics.
    """

    @abstractmethod
    def compute_control(self, *args, **kwargs) -> float:
        """
        Compute control output.

        Returns:
            Control value (e.g., steering angle)
        """
        pass
