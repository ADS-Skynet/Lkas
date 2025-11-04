"""
Integration layer for coordinating CARLA, Detection, and Decision modules.

Note: Message definitions and shared memory channels have been moved to
detection.integration to better reflect their ownership by the detection module.
This module maintains backwards compatibility by re-exporting them.
"""

# Re-export messages from their new location in detection.integration
from detection.integration.messages import (
    ImageMessage,
    LaneMessage,
    DetectionMessage,
    ControlMessage,
    ControlMode,
    SystemStatus,
    PerformanceMetrics
)

# Re-export shared memory from detection.integration
from detection.integration.shared_memory_detection import (
    SharedMemoryDetectionServer,
    SharedMemoryDetectionClient,
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel
)

# Local simulation-specific communication (ZMQ)
from .communication import DetectionClient, DetectionServer

# Orchestrators are imported directly where needed to avoid circular imports
# Use: from simulation.integration.distributed_orchestrator import DistributedOrchestrator

__all__ = [
    # Messages
    'ImageMessage',
    'LaneMessage',
    'DetectionMessage',
    'ControlMessage',
    'ControlMode',
    'SystemStatus',
    'PerformanceMetrics',

    # Shared Memory
    'SharedMemoryDetectionServer',
    'SharedMemoryDetectionClient',
    'SharedMemoryImageChannel',
    'SharedMemoryDetectionChannel',

    # ZMQ Communication
    'DetectionClient',
    'DetectionServer',
]
