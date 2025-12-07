"""
Shared Memory Communication Channels

High-performance IPC using shared memory for LKAS system.
"""

from .channels import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel,
    SharedMemoryControlChannel,
)
from .messages import (
    ImageMessage,
    DetectionMessage,
    LaneMessage,
    ControlMessage,
    ControlMode,
    SystemStatus,
    PerformanceMetrics,
)

__all__ = [
    'SharedMemoryImageChannel',
    'SharedMemoryDetectionChannel',
    'SharedMemoryControlChannel',
    'ImageMessage',
    'DetectionMessage',
    'LaneMessage',
    'ControlMessage',
    'ControlMode',
    'SystemStatus',
    'PerformanceMetrics',
]
