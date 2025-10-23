"""
Lane Detection Models
This file serves as a compatibility layer for the old structure.

For the actual implementations, see:
- method/computer_vision/cv_lane_detector.py - OpenCV-based lane detection
- method/deep_learning/lane_net.py - CNN-based lane detection models

Usage:
    from method.computer_vision.cv_lane_detector import CVLaneDetector
    from method.deep_learning.lane_net import LaneNet, SimpleLaneNet, DLLaneDetector
"""

# Import all models for easy access
from method.computer_vision.cv_lane_detector import CVLaneDetector
from method.deep_learning.lane_net import LaneNet, SimpleLaneNet, DLLaneDetector

__all__ = ["CVLaneDetector", "LaneNet", "SimpleLaneNet", "DLLaneDetector"]
