"""
Shared Memory Communication Channels

High-performance IPC using shared memory for LKAS system.
"""

from .channels import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel,
    SharedMemoryControlChannel,
)

__all__ = [
    'SharedMemoryImageChannel',
    'SharedMemoryDetectionChannel',
    'SharedMemoryControlChannel',
]
