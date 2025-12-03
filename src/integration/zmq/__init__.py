"""
ZMQ Integration for LKAS

Clean, encapsulated ZMQ communication layer.

Public API:
    Broker-side (runs in LKAS main process):
        - LKASBroker: Main broker that manages all ZMQ communication

    Client-side (used by detection/decision servers):
        - ParameterClient: DEPRECATED - Use skynet_common.communication.ParameterSubscriber

    Messages:
        - VehicleState: Vehicle state message
        - ParameterUpdate: Parameter update message
        - ActionRequest: Action request message

DEPRECATION NOTICE:
    ParameterClient has been renamed to ParameterSubscriber and moved to
    skynet_common.communication for consistent naming (pairs with ParameterPublisher).

    Please update your imports:
        Old: from lkas.integration.zmq import ParameterClient
        New: from skynet_common.communication import ParameterSubscriber
"""

from .broker import LKASBroker, create_broker_from_config
from .messages import VehicleState, ParameterUpdate, ActionRequest

__all__ = [
    "LKASBroker",
    "create_broker_from_config",
    "VehicleState",
    "ParameterUpdate",
    "ActionRequest",
]
