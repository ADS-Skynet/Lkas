"""
Lane Detection Module
Lane keeping assist system implementation for CARLA and PiRacer.

Public API:
- DetectionServer: Run detection process (reads images, writes detections)
- DetectionClient: Bidirectional client (write images, read detections)
- LaneDetection: Core detection algorithm (typically used internally by DetectionServer)

Simple Usage:
    # Simulation/Camera side
    client = DetectionClient(
        detection_shm_name="detection_results",
        image_shm_name="camera_feed",
        image_shape=(600, 800, 3)
    )
    client.send_image(image, timestamp=time.time(), frame_id=frame_count)
    detection = client.get_detection()
"""

from .detector import LaneDetection
from .server import DetectionServer
from .client import DetectionClient

__version__ = "0.1.0"

__all__ = ['LaneDetection', 'DetectionServer', 'DetectionClient']
