"""
Integration layer for coordinating CARLA with LKAS (Lane Keeping Assist System).

Note: LKAS components are in the lkas module.
This module maintains backwards compatibility by re-exporting them.

For communication with LKAS, use the clean API:
    from lkas import LKAS

    lkas = LKAS(image_shape=(600, 800, 3))
    lkas.send_image(image, timestamp, frame_id)
    control = lkas.get_control()

Or use components directly:
    from lkas.detection import DetectionClient

    client = DetectionClient(
        detection_shm_name="detection_results",
        image_shm_name="camera_feed",
        image_shape=(600, 800, 3)
    )
"""

# Re-export messages from lkas.detection.integration
from lkas.detection.integration.messages import (
    ImageMessage,
    LaneMessage,
    DetectionMessage,
    ControlMessage,
    ControlMode,
    SystemStatus,
    PerformanceMetrics
)

# Re-export detection communication API (for backwards compatibility)
from lkas.detection import DetectionClient

__all__ = [
    # Messages
    'ImageMessage',
    'LaneMessage',
    'DetectionMessage',
    'ControlMessage',
    'ControlMode',
    'SystemStatus',
    'PerformanceMetrics',

    # Detection Communication (clean API)
    'DetectionClient',
]
