"""
ZMQ Message Types

Defines the structure of messages exchanged over ZMQ.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time


@dataclass
class ParameterUpdate:
    """
    Parameter update message.

    Sent from viewer to broker, then forwarded to detection/decision servers.
    """
    category: str  # 'detection' or 'decision'
    parameter: str  # Parameter name
    value: float  # New value
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionRequest:
    """
    Action request message.

    Sent from viewer to broker, then routed to appropriate handler.
    """
    action: str  # Action name (e.g., 'respawn', 'pause', 'resume')
    params: Dict[str, Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VehicleState:
    """
    Vehicle state message.

    Broadcast from LKAS to viewers.
    """
    timestamp: float
    frame_id: int

    # Vehicle kinematics
    x: float
    y: float
    yaw: float
    velocity: float

    # Control state
    steering_angle: float
    target_steering: Optional[float] = None

    # Detection results
    left_lane_detected: bool = False
    right_lane_detected: bool = False
    lateral_offset: Optional[float] = None
    heading_error: Optional[float] = None

    # Simulation state
    paused: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
