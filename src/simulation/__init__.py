"""
CARLA Module

Handles all CARLA simulator interactions:
- Connection management
- Vehicle spawning and control
- Sensor setup and data collection
"""

from .connection import CARLAConnection
from .vehicle import VehicleManager
from .sensors import CameraSensor

__all__ = ['CARLAConnection', 'VehicleManager', 'CameraSensor']
