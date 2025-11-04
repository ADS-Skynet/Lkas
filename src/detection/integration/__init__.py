"""
Detection Integration Module

Provides inter-module communication infrastructure:
- Message definitions for all modules
- Shared memory channels for ultra-low latency IPC
"""

from .messages import (
    ImageMessage,
    DetectionMessage,
    LaneMessage,
    ControlMessage,
    ControlMode,
    SystemStatus,
    PerformanceMetrics
)

from .shared_memory_detection import (
    SharedMemoryDetectionServer,
    SharedMemoryDetectionClient,
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel
)

from .shared_memory_control import (
    SharedMemoryControlChannel,
)

__all__ = [
    # Messages
    'ImageMessage',
    'DetectionMessage',
    'LaneMessage',
    'ControlMessage',
    'ControlMode',
    'SystemStatus',
    'PerformanceMetrics',

    # Shared Memory - Detection
    'SharedMemoryDetectionServer',
    'SharedMemoryDetectionClient',
    'SharedMemoryImageChannel',
    'SharedMemoryDetectionChannel',

    # Shared Memory - Control
    'SharedMemoryControlChannel',
]
