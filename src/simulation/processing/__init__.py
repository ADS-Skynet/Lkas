"""
Processing components for simulation-specific pipeline.

Note: PDController has been moved to decision.pd_controller as it contains
vehicle-agnostic control logic. This module maintains backwards compatibility
by re-exporting it.
"""

from .frame_processor import FrameProcessor
from .metrics_logger import MetricsLogger

# Re-export from decision module for backwards compatibility
from decision.pd_controller import PDController

__all__ = [
    'FrameProcessor',
    'PDController',
    'MetricsLogger',
]
