"""
Decision Module

Vehicle-agnostic decision-making and control logic:
- Segmentation lane parsing from DL detection masks
- Pure Pursuit steering controller
- Adaptive throttle policy
- Control command generation (steering, throttle, brake)

Public API:
- DecisionServer: Run decision process (reads detections, writes controls)
- DecisionClient: Read control commands
- DecisionController: Core decision logic
- ControllerFactory: Factory for creating steering controllers
- SteeringController: Abstract base class for all controllers
"""

from .controller import DecisionController
from .lane_analyzer import LaneAnalyzer
from .segmentation_lane_parser import SegmentationLaneParser
from .client import DecisionClient
from .core import ControllerFactory, SteeringController
from .method import PurePursuitController

__all__ = [
    'DecisionController',
    'ControllerFactory',
    'SteeringController',
    'PurePursuitController',
    'LaneAnalyzer',
    'SegmentationLaneParser',
    'DecisionClient',
]
