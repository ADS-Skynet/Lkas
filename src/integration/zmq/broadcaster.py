"""
ZMQ Broadcaster

Broadcasts vehicle state, frames, and detection results to remote viewers.
"""

import zmq
import json
import cv2
import numpy as np
import time
from typing import Optional, Dict, Any

from .messages import VehicleState


class VehicleBroadcaster:
    """
    Publisher: Broadcasts vehicle data to remote viewers.

    Publishes:
    - Video frames (topic: 'frame')
    - Detection results visualization (topic: 'detection')
    - Vehicle state (topic: 'state')
    """

    def __init__(self, bind_url: str = "tcp://*:5557", context: Optional[zmq.Context] = None):
        """
        Initialize broadcaster.

        Args:
            bind_url: ZMQ URL to bind publisher socket
            context: ZMQ context (optional, will create if not provided)
        """
        self.bind_url = bind_url

        # Create or use provided context
        self.context = context if context else zmq.Context()
        self.owns_context = context is None

        # Create publisher socket
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(bind_url)

        # High water mark (max queued messages)
        # Drop old frames if viewer is slow (we want real-time, not buffered)
        self.socket.setsockopt(zmq.SNDHWM, 10)

        # Stats
        self.frame_count = 0
        self.last_frame_time = time.time()

    def send_frame(
        self,
        image: np.ndarray,
        frame_id: int,
        jpeg_quality: int = 85,
        raw_rgb: bool = False
    ):
        """
        Send video frame to viewers.

        Args:
            image: Image array (RGB or BGR)
            frame_id: Frame sequence number
            jpeg_quality: JPEG compression quality (0-100), ignored if raw_rgb=True
            raw_rgb: Send raw RGB instead of JPEG (faster for localhost, larger bandwidth)
        """
        if raw_rgb:
            # Send raw RGB data (no encoding overhead, ~10x larger)
            # Convert to contiguous RGB if needed
            if image.shape[2] == 3:
                image_rgb = image if image.flags['C_CONTIGUOUS'] else np.ascontiguousarray(image)
            else:
                image_rgb = image

            message = {
                'timestamp': time.time(),
                'frame_id': frame_id,
                'width': image.shape[1],
                'height': image.shape[0],
                'format': 'raw_rgb',
                'size': image_rgb.nbytes,
            }

            # Send multipart: [topic, metadata_json, raw_rgb_data]
            self.socket.send_multipart([
                b'frame',
                json.dumps(message).encode('utf-8'),
                image_rgb.tobytes(),
            ])
        else:
            # Compress to JPEG (10x smaller for network transfer)
            if image.shape[2] == 3:
                # Assume RGB, convert to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            success, buffer = cv2.imencode('.jpg', image_bgr, encode_param)

            if not success:
                return

            # Create message metadata
            message = {
                'timestamp': time.time(),
                'frame_id': frame_id,
                'width': image.shape[1],
                'height': image.shape[0],
                'format': 'jpeg',
                'jpeg_size': len(buffer),
            }

            # Send multipart: [topic, metadata_json, jpeg_data]
            self.socket.send_multipart([
                b'frame',
                json.dumps(message).encode('utf-8'),
                buffer.tobytes(),
            ])

        self.frame_count += 1

    def send_detection(
        self,
        detection_data: Dict[str, Any],
        frame_id: int = None
    ):
        """
        Send detection data to viewers (no image, just lane data).

        Args:
            detection_data: Detection data (left_lane, right_lane, etc.)
            frame_id: Frame sequence number (optional, included in detection_data)
        """
        # Send detection data as-is (viewer expects exact format)
        # Detection data should include: left_lane, right_lane, processing_time_ms, frame_id
        # Do NOT add extra fields like timestamp (breaks viewer's DetectionData deserialization)

        # Send as JSON only (no image data)
        self.socket.send_multipart([
            b'detection',
            json.dumps(detection_data).encode('utf-8'),
        ])

    def send_state(self, state: VehicleState):
        """
        Send vehicle state to viewers.

        Args:
            state: Vehicle state message
        """
        message = state.to_dict()

        # Send multipart: [topic, json_data]
        self.socket.send_multipart([
            b'state',
            json.dumps(message).encode('utf-8'),
        ])

    def close(self):
        """Close the broadcaster and cleanup resources."""
        if self.socket:
            self.socket.close()
        if self.owns_context and self.context:
            self.context.term()

    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        elapsed = time.time() - self.last_frame_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            'frame_count': self.frame_count,
            'fps': fps,
            'bind_url': self.bind_url,
        }


# =============================================================================
# Usage Example
# =============================================================================

"""
# LKAS Main Process Example
# ==========================

from lkas.integration.zmq import LKASBroker
from lkas.integration.zmq.messages import VehicleState

# Create broker (includes broadcaster)
broker = LKASBroker()

# In main loop
while running:
    # Broadcast vehicle state
    state = VehicleState(
        timestamp=time.time(),
        frame_id=frame_id,
        x=vehicle.x,
        y=vehicle.y,
        yaw=vehicle.yaw,
        velocity=vehicle.velocity,
        steering_angle=vehicle.steering,
        left_lane_detected=detection.left_lane is not None,
        right_lane_detected=detection.right_lane is not None,
    )
    broker.broadcast_state(state)

    # Broadcast frame (optional)
    broker.broadcast_frame(image, frame_id)

    # Poll for incoming parameter updates and actions
    broker.poll()

# Cleanup
broker.close()
"""
