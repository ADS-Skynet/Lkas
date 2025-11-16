"""
Decision Client

Client for reading control commands from decision server.
Used by simulation/vehicle processes to receive control outputs.

Architecture:
    Decision Server (decision/run.py)
    ↓ writes control commands
    DecisionClient (decision/client.py)
    ↓ reads controls
    Simulation/Vehicle Process
"""

from typing import Optional
from lkas.integration.shared_memory import SharedMemoryControlChannel
from lkas.integration.messages import ControlMessage


class DecisionClient:
    """
    Client for reading control commands from decision server.
    Pairs with DecisionServer for unified architecture.

    Usage:
        client = DecisionClient(shm_name="control_commands")
        control = client.get_control()
        vehicle.apply_control(control.steering, control.throttle, control.brake)
        client.close()
    """

    def __init__(self, shm_name: str, retry_count: int = 20, retry_delay: float = 0.5):
        """
        Initialize decision client.

        Args:
            shm_name: Name of control shared memory
            retry_count: Number of retry attempts for connection (default: 20)
            retry_delay: Delay between retries in seconds (default: 0.5)
        """
        self.shm_name = shm_name

        # Connect to control shared memory (reader) with retry
        self._channel = SharedMemoryControlChannel(
            name=shm_name,
            create=False,
            retry_count=retry_count,
            retry_delay=retry_delay
        )

    def get_control(self, timeout: float = 1.0) -> Optional[ControlMessage]:
        """
        Get latest control commands.

        Args:
            timeout: Maximum wait time (not implemented yet)

        Returns:
            ControlMessage or None
        """
        return self._channel.read()

    def close(self):
        """Close connection."""
        self._channel.close()
