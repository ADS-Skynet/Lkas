"""
ZMQ Client API

Easy-to-use API for detection and decision servers to subscribe to parameter updates.

Usage:
    # In detection server
    client = ParameterClient(category='detection')
    client.register_callback(on_parameter_update)

    # In main loop
    while running:
        client.poll()  # Non-blocking, processes any pending updates
        # ... do your work ...

    client.close()
"""

import zmq
import json
from typing import Callable, Optional


class ParameterClient:
    """
    Client for subscribing to parameter updates.

    Provides a clean, simple API for detection and decision servers.
    """

    def __init__(
        self,
        category: str,
        broker_url: str = "tcp://localhost:5560",
        context: Optional[zmq.Context] = None,
    ):
        """
        Initialize parameter client.

        Args:
            category: 'detection' or 'decision' (filters messages for this category)
            broker_url: ZMQ URL of the parameter broker
            context: ZMQ context (optional, will create if not provided)
        """
        self.category = category
        self.broker_url = broker_url

        # Create or use provided context
        self.context = context if context else zmq.Context()
        self.owns_context = context is None

        # Create subscriber socket
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(broker_url)

        # Subscribe only to our category
        self.socket.setsockopt(zmq.SUBSCRIBE, category.encode('utf-8'))

        # Non-blocking mode with timeout
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        # Callback function
        self.callback: Optional[Callable[[str, float], None]] = None

    def register_callback(self, callback: Callable[[str, float], None]):
        """
        Register callback for parameter updates.

        Args:
            callback: Function that takes (parameter_name: str, value: float)

        Example:
            def on_update(param_name: str, value: float):
                print(f"Parameter {param_name} updated to {value}")

            client.register_callback(on_update)
        """
        self.callback = callback

    def poll(self) -> bool:
        """
        Poll for parameter updates (non-blocking).

        Call this regularly in your main loop.

        Returns:
            True if a message was received, False otherwise
        """
        try:
            # Receive multipart message: [category, json_data]
            parts = self.socket.recv_multipart(zmq.NOBLOCK)

            category = parts[0].decode('utf-8')
            data = json.loads(parts[1].decode('utf-8'))

            # Extract parameter info
            parameter = data['parameter']
            value = data['value']

            # Call user callback
            if self.callback:
                self.callback(parameter, value)

            return True

        except zmq.Again:
            # No message available (this is normal)
            return False
        except Exception as e:
            # Unexpected error (log but don't crash)
            print(f"[ParameterClient] Error receiving message: {e}")
            return False

    def close(self):
        """Close the client and cleanup resources."""
        if self.socket:
            self.socket.close()
        if self.owns_context and self.context:
            self.context.term()


# =============================================================================
# Usage Example
# =============================================================================

"""
# Detection Server Example
# ========================

from lkas.integration.zmq import ParameterClient

class DetectionServer:
    def __init__(self):
        # Create parameter client
        self.param_client = ParameterClient(category='detection')
        self.param_client.register_callback(self._on_parameter_update)

    def _on_parameter_update(self, param_name: str, value: float):
        # Update detector parameters
        if hasattr(self.detector, 'update_parameter'):
            self.detector.update_parameter(param_name, value)

    def run(self):
        while self.running:
            # Poll for parameter updates (non-blocking)
            self.param_client.poll()

            # Do your detection work
            # ...

    def stop(self):
        self.param_client.close()


# Decision Server Example
# ========================

from lkas.integration.zmq import ParameterClient

class DecisionServer:
    def __init__(self):
        # Create parameter client
        self.param_client = ParameterClient(category='decision')
        self.param_client.register_callback(self._on_parameter_update)

    def _on_parameter_update(self, param_name: str, value: float):
        # Update controller parameters
        if hasattr(self.controller, 'update_parameter'):
            self.controller.update_parameter(param_name, value)

    def run(self):
        while self.running:
            # Poll for parameter updates (non-blocking)
            self.param_client.poll()

            # Do your decision work
            # ...

    def stop(self):
        self.param_client.close()
"""
