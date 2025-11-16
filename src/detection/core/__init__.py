"""
Core data structures and interfaces for lane detection system.
"""

from .models import Lane, LaneMetrics, VehicleTelemetry, DetectionResult
from .config import Config, ConfigManager
from .interfaces import LaneDetector, SensorInterface

__all__ = [
    'Lane',
    'LaneMetrics',
    'VehicleTelemetry',
    'DetectionResult',
    'Config',
    'ConfigManager',
    'LaneDetector',
    'SensorInterface',
]
