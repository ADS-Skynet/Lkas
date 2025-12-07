"""
Decision Module

Vehicle-agnostic decision-making and control logic:
- Lane analysis and position estimation
- Multiple steering controllers (PD, PID, MPC, etc.)
- Adaptive throttle policy
- Control command generation (steering, throttle, brake)

This module produces generic control commands that work for any vehicle
(CARLA simulation, real vehicle, etc.) without platform-specific code.

Public API:
- DecisionServer: Run decision process (reads detections, writes controls)
- DecisionClient: Read control commands
- DecisionController: Core decision logic (uses factory to create controllers)
- ControllerFactory: Factory for creating steering controllers
- SteeringController: Abstract base class for all controllers

Simple Usage:
    # Create a decision controller with PID
    controller = DecisionController(
        image_width=800,
        image_height=600,
        controller_method='pid',
        kp=0.5, ki=0.01, kd=0.1
    )

    # Or use factory directly
    from lkas.decision.core import ControllerFactory
    factory = ControllerFactory()
    steering_controller = factory.create('mpc', prediction_horizon=15)
"""

from .controller import DecisionController
from .lane_analyzer import LaneAnalyzer
from .client import DecisionClient
from .core import ControllerFactory, SteeringController

# Expose method-level controllers for backward compatibility
from .method import PDController, PIDController

__all__ = [
    'DecisionController',
    'ControllerFactory',
    'SteeringController',
    'PDController',
    'PIDController',
    'LaneAnalyzer',
    'DecisionClient',
]
