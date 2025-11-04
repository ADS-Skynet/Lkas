"""
Decision Module

Handles decision-making and control logic based on lane detection:
- Control command generation (steering, throttle, brake)
- Path planning and trajectory control

Note: LaneAnalyzer moved to simulation.utils.lane_analyzer to avoid circular dependencies
"""

from .controller import DecisionController

__all__ = ['DecisionController']
