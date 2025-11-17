"""
ZMQ Integration for LKAS

Clean, encapsulated ZMQ communication layer.

Public API:
    Broker-side (runs in LKAS main process):
        - LKASBroker: Main broker that manages all ZMQ communication

    Client-side (used by detection/decision servers):
        - ParameterClient: Subscribe to parameter updates

    Messages:
        - VehicleState: Vehicle state message
        - ParameterUpdate: Parameter update message
        - ActionRequest: Action request message
"""

from .broker import LKASBroker, create_broker_from_config
from .client import ParameterClient
from .messages import VehicleState, ParameterUpdate, ActionRequest

__all__ = [
    "LKASBroker",
    "create_broker_from_config",
    "ParameterClient",
    "VehicleState",
    "ParameterUpdate",
    "ActionRequest",
]
