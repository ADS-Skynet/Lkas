"""
Lane Keeping Assist System (LKAS) Module

Complete autonomous lane keeping system with detection and decision-making.

Architecture:
    lkas/
    ├── system.py          - LKAS, LKASSimple (high-level wrappers)
    ├── detection/         - Lane detection subsystem
    │   ├── server.py      - DetectionServer
    │   ├── client.py      - DetectionClient
    │   └── detector.py    - Core detection algorithms
    └── decision/          - Decision-making subsystem
        ├── run.py         - DecisionServer
        ├── client.py      - DecisionClient
        └── controller.py  - Core control logic

Usage Levels:

    Level 1 (Highest) - Production Use:
        from lkas import LKASClient

        lkas = LKASClient(image_shape=(600, 800, 3))
        lkas.send_image(image, timestamp, frame_id)
        control = lkas.get_control()

    Level 2 - Component Access:
        from lkas.detection import DetectionClient
        from lkas.decision import DecisionClient

        detection = DetectionClient(...)
        decision = DecisionClient(...)

    Level 3 - Server Processes:
        # Terminal 1
        lane-detection

        # Terminal 2
        decision-server

Public API:
- LKASClient: Complete lane keeping system (recommended)
- LKASSimple: Minimal interface (send/receive only)

Submodules (for advanced use):
- lkas.detection: Lane detection components
- lkas.decision: Decision-making components
"""

from .client import LKASClient, LKASSimple

__version__ = "0.1.0"

__all__ = ['LKASClient', 'LKASSimple']
