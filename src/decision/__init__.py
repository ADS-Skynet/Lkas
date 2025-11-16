"""
Decision Module

Vehicle-agnostic decision-making and control logic:
- Lane analysis and position estimation
- PD control for steering computation
- Adaptive throttle policy
- Control command generation (steering, throttle, brake)

This module produces generic control commands that work for any vehicle
(CARLA simulation, real vehicle, etc.) without platform-specific code.

Public API:
- DecisionServer: Run decision process (reads detections, writes controls)
- DecisionClient: Read control commands
- DecisionController: Core decision logic (typically used internally by DecisionServer)

Simple Usage:
    # Simulation/Vehicle side
    client = DecisionClient(shm_name="control_commands")
    control = client.get_control()
    vehicle.apply_control(control.steering, control.throttle, control.brake)
"""

from .controller import DecisionController
from .pd_controller import PDController
from .lane_analyzer import LaneAnalyzer
from .client import DecisionClient

__all__ = [
    'DecisionController',
    'PDController',
    'LaneAnalyzer',
    'DecisionClient',
]
