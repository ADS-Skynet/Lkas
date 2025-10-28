"""Integration layer for coordinating CARLA, Detection, and Decision modules."""

# Import only messages and communication (no circular deps)
from .messages import (
    ImageMessage,
    LaneMessage,
    DetectionMessage,
    ControlMessage,
    ControlMode,
    SystemStatus,
    PerformanceMetrics
)
from .communication import DetectionClient, DetectionServer

# Orchestrators are imported directly where needed to avoid circular imports
# Use: from integration.orchestrator import SystemOrchestrator
# Use: from integration.distributed_orchestrator import DistributedOrchestrator

__all__ = [
    'ImageMessage',
    'LaneMessage',
    'DetectionMessage',
    'ControlMessage',
    'ControlMode',
    'SystemStatus',
    'PerformanceMetrics',
    'DetectionClient',
    'DetectionServer',
]
