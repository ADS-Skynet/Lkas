"""
Core Interfaces

Abstract interfaces following Dependency Inversion Principle.
High-level modules depend on these abstractions, not concrete implementations.
"""

from .communication import IDetectionChannel
from .vehicle import IVehicleController
from .detector import ILaneDetector
from .config import IConfigRepository

__all__ = [
    'IDetectionChannel',
    'IVehicleController',
    'ILaneDetector',
    'IConfigRepository',
]
