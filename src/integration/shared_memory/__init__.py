"""
Shared Memory Communication Channels

High-performance IPC using shared memory for LKAS system.
"""

from .channels import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel,
    SharedMemoryControlChannel,
    SharedObstacleData,
)
from .messages import (
    ImageMessage,
    DetectionMessage,
    LaneMessage,
    ControlMessage,
    ControlMode,
    ObstacleMessage,
    ObstacleAction,
    SystemStatus,
    PerformanceMetrics,
)

__all__ = [
    'SharedMemoryImageChannel',
    'SharedMemoryDetectionChannel',
    'SharedMemoryControlChannel',
    'SharedObstacleData',
    'ImageMessage',
    'DetectionMessage',
    'LaneMessage',
    'ControlMessage',
    'ControlMode',
    'ObstacleMessage',
    'ObstacleAction',
    'SystemStatus',
    'PerformanceMetrics',
]
