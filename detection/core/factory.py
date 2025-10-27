"""
Factory pattern for creating lane detectors.

Centralizes detector instantiation and configuration.
"""

from typing import Optional
from .interfaces import LaneDetector
from .config import Config


class DetectorFactory:
    """
    Factory for creating lane detector instances.

    Usage:
        factory = DetectorFactory(config)
        detector = factory.create('cv')
    """

    def __init__(self, config: Config):
        """
        Initialize factory with configuration.

        Args:
            config: System configuration
        """
        self.config = config

    def create(self, detector_type: Optional[str] = None, **kwargs) -> LaneDetector:
        """
        Create a lane detector instance.

        Args:
            detector_type: Type of detector ('cv', 'dl', or None for config default)
            **kwargs: Additional parameters to override config

        Returns:
            LaneDetector instance

        Raises:
            ValueError: If detector_type is invalid
        """
        if detector_type is None:
            detector_type = self.config.detection_method

        detector_type = detector_type.lower()

        if detector_type == "cv":
            return self._create_cv_detector(**kwargs)
        elif detector_type == "dl":
            return self._create_dl_detector(**kwargs)
        else:
            raise ValueError(
                f"Unknown detector type: {detector_type}. Use 'cv' or 'dl'."
            )

    def _create_cv_detector(self, **kwargs) -> LaneDetector:
        """Create Computer Vision detector."""
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from method.computer_vision.cv_lane_detector import CVLaneDetector

        cfg = self.config.cv_detector

        params = {
            "canny_low": kwargs.get("canny_low", cfg.canny_low),
            "canny_high": kwargs.get("canny_high", cfg.canny_high),
            "hough_threshold": kwargs.get("hough_threshold", cfg.hough_threshold),
            "hough_min_line_len": kwargs.get(
                "hough_min_line_len", cfg.hough_min_line_len
            ),
            "hough_max_line_gap": kwargs.get(
                "hough_max_line_gap", cfg.hough_max_line_gap
            ),
            "smoothing_factor": kwargs.get("smoothing_factor", cfg.smoothing_factor),
        }

        return CVLaneDetector(**params)

    def _create_dl_detector(self, **kwargs) -> LaneDetector:
        """Create Deep Learning detector."""
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from method.deep_learning.lane_net import DLLaneDetector

        cfg = self.config.dl_detector

        params = {
            "model_type": kwargs.get("model_type", cfg.model_type),
            "input_size": kwargs.get("input_size", cfg.input_size),
            "threshold": kwargs.get("threshold", cfg.threshold),
            "device": kwargs.get("device", cfg.device),
            "model_path": kwargs.get("model_path", None),
        }

        return DLLaneDetector(**params)

    @staticmethod
    def list_available_detectors() -> list:
        """
        List all available detector types.

        Returns:
            List of detector type strings
        """
        return ["cv", "dl"]
