"""
Data models for lane detection system.

This module re-exports models from skynet-common for backward compatibility.
New code should import directly from skynet_common.types.
"""

# Re-export everything from skynet-common for backward compatibility
from skynet_common.types.models import (
    LaneDepartureStatus,
    Lane,
    LaneMetrics,
    VehicleTelemetry,
    DetectionResult,
)

__all__ = [
    "LaneDepartureStatus",
    "Lane",
    "LaneMetrics",
    "VehicleTelemetry",
    "DetectionResult",
]
