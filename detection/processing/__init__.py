"""
Processing components for lane detection pipeline.

These components handle the business logic between detection and control.
"""

from .frame_processor import FrameProcessor
from .pd_controller import PDController
from .metrics_logger import MetricsLogger

__all__ = [
    'FrameProcessor',
    'PDController',
    'MetricsLogger',
]
