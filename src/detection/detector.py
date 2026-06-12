"""
Lane Detection Module

Standalone detector that processes images and returns lane detection results.
"""

import numpy as np
import time

from lkas.integration.shared_memory.messages import ImageMessage, DetectionMessage, LaneMessage, LaneContour
from lkas.detection.core.factory import DetectorFactory
from common.config import Config


class LaneDetection:
    """
    Standalone lane detection module.

    Responsibility:
    - Accept image data
    - Run lane detection algorithm (CV or DL)
    - Return structured detection results
    """

    def __init__(self, config: Config, method: str = "cv"):
        """
        Initialize lane detection module.

        Args:
            config: System configuration
            method: Detection method ('cv' or 'dl')
        """
        self.config = config
        self.method = method

        # Create detector using factory
        factory = DetectorFactory(config)
        self.detector = factory.create(method)

    def process_image(self, image_msg: ImageMessage) -> DetectionMessage:
        """
        Process an image and detect lanes.

        Args:
            image_msg: Image message from CARLA

        Returns:
            Detection message with lane results
        """
        start_time = time.time()

        # Run detection
        result = self.detector.detect(image_msg.image)

        # Convert Lane objects to LaneMessage objects
        left_lane_msg = None
        right_lane_msg = None

        if result.left_lane:
            left_lane_msg = LaneMessage(
                x1=result.left_lane.x1,
                y1=result.left_lane.y1,
                x2=result.left_lane.x2,
                y2=result.left_lane.y2,
                confidence=result.left_lane.confidence,
            )

        if result.right_lane:
            right_lane_msg = LaneMessage(
                x1=result.right_lane.x1,
                y1=result.right_lane.y1,
                x2=result.right_lane.x2,
                y2=result.right_lane.y2,
                confidence=result.right_lane.confidence,
            )

        # Get segmentation mask and pre-computed lane grid if available (DL detection)
        segmentation_mask = None
        if hasattr(self.detector, 'get_last_mask'):
            segmentation_mask = self.detector.get_last_mask()

        lane_grid = None
        if hasattr(self.detector, 'get_last_lane_grid'):
            lane_grid = self.detector.get_last_lane_grid()

        # Convert lane contours from DetectionResult to LaneContour messages
        lanes_msg = None
        if result.lanes:
            lanes_msg = [
                LaneContour(
                    points=lane.points,
                    class_id=lane.class_id,
                    confidence=lane.confidence
                )
                for lane in result.lanes
            ]

        # Create detection message
        # Note: No debug_image - viewer handles all overlays on laptop side
        detection_msg = DetectionMessage(
            left_lane=left_lane_msg,
            right_lane=right_lane_msg,
            processing_time_ms=result.processing_time_ms,
            frame_id=image_msg.frame_id,
            timestamp=image_msg.timestamp,
            debug_image=None,  # Viewer draws overlays, not LKAS
            segmentation_mask=segmentation_mask,
            detection_method=self.method,
            lanes=lanes_msg,
            lane_grid=lane_grid,
        )

        return detection_msg

    def get_detector_name(self) -> str:
        """Get detector name."""
        return self.detector.get_name()

    def get_detector_params(self) -> dict:
        """Get detector parameters."""
        return self.detector.get_parameters()
