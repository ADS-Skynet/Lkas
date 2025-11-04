"""
Decision Module

Vehicle-agnostic decision-making and control logic:
- Lane analysis and position estimation
- PD control for steering computation
- Adaptive throttle policy
- Control command generation (steering, throttle, brake)

This module produces generic control commands that work for any vehicle
(CARLA simulation, real vehicle, etc.) without platform-specific code.
"""

from .controller import DecisionController
from .pd_controller import PDController
from .lane_analyzer import LaneAnalyzer

__all__ = [
    'DecisionController',
    'PDController',
    'LaneAnalyzer',
]
