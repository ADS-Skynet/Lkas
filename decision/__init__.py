"""
Decision Module

Handles decision-making and control logic based on lane detection:
- Lane analysis and metrics computation
- Control command generation (steering, throttle, brake)
- Path planning and trajectory control
"""

from .analyzer import LaneAnalyzer
from .controller import DecisionController

__all__ = ['LaneAnalyzer', 'DecisionController']
