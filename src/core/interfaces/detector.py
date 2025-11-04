"""
Lane Detector Interface

Abstract interface for lane detection implementations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class ILaneDetector(ABC):
    """
    Abstract interface for lane detectors.

    Implementations: CV-based, DL-based, Hybrid
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> object:
        """
        Detect lanes in image.

        Args:
            image: RGB image array

        Returns:
            DetectionResult with lane information
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get detector name."""
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Get detector parameters."""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before detection.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        pass
