"""
Factory pattern for creating lane detectors.

Centralizes detector instantiation and configuration.
"""

from typing import Any
from .interfaces import LaneDetector


class DetectorFactory:
    """
    Factory for creating lane detector instances.

    Usage:
        factory = DetectorFactory(config)
        detector = factory.create('dl')
    """

    def __init__(self, config: Any):
        """
        Initialize factory with configuration.

        Args:
            config: System configuration
        """
        self.config = config

    def create(self, detector_type: str | None = None, **kwargs) -> LaneDetector:
        """
        Create a lane detector instance.

        Args:
            detector_type: Type of detector ('dl', or None for config default)
            **kwargs: Additional parameters to override config

        Returns:
            LaneDetector instance

        Raises:
            ValueError: If detector_type is invalid
        """
        if detector_type is None:
            detector_type = self.config.detection_method

        detector_type = detector_type.lower()

        if detector_type == "dl":
            return self._create_dl_detector(**kwargs)
        else:
            raise ValueError(
                f"Unknown detector type: {detector_type}. Use 'dl'."
            )

    def _create_dl_detector(self, **kwargs) -> LaneDetector:
        """Create Deep Learning detector (BiSeNet V2 with PyTorch)."""
        cfg = self.config.dl_detector

        from lkas.detection.method.deep_learning.lane_net import DLLaneDetector

        params = {
            "model_type": kwargs.get("model_type", cfg.model_type),
            "input_size": kwargs.get("input_size", cfg.input_size),
            "threshold": kwargs.get("threshold", cfg.threshold),
            "device": kwargs.get("device", cfg.device),
            "model_path": kwargs.get("model_path", cfg.model_path),
            "n_classes": kwargs.get("n_classes", cfg.n_classes),
            "smoothing_factor": kwargs.get("smoothing_factor", cfg.smoothing_factor),
            "use_fp16": kwargs.get("use_fp16", cfg.use_fp16),
        }

        return DLLaneDetector(**params)

    @staticmethod
    def list_available_detectors() -> list:
        """
        List all available detector types.

        Returns:
            List of detector type strings
        """
        return ["dl"]
