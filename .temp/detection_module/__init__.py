"""
Detection Module

Standalone lane detection module that processes images and returns lane information.
Wraps existing CV and DL detection methods.
"""

from ..detector import LaneDetection

__all__ = ['LaneDetection']
