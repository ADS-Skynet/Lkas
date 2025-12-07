"""
Broadcast Management Module

Handles ZMQ broadcasting and shared memory reading for remote viewers.
Separates broadcasting concerns from main server logic.
"""

import sys
import io
import time
from typing import Optional, Dict


class BroadcastManager:
    """
    Manages ZMQ broadcasting of frames and detection data.

    Responsibilities:
    - Setup ZMQ broker
    - Connect to shared memory channels (lazy)
    - Broadcast frames and detections
    - Track FPS statistics
    """

    def __init__(
        self,
        system_config,
        image_shm_name: str,
        detection_shm_name: str,
        control_shm_name: str,
        jpeg_quality: int,
        raw_rgb: bool,
        broadcast_log_interval: int,
        verbose: bool,
        terminal,
    ):
        """
        Initialize broadcast manager.

        Args:
            system_config: System configuration
            image_shm_name: Shared memory name for images
            detection_shm_name: Shared memory name for detections
            control_shm_name: Shared memory name for controls
            jpeg_quality: JPEG compression quality
            raw_rgb: Whether to send raw RGB data
            broadcast_log_interval: Interval for logging broadcasts
            verbose: Enable verbose logging
            terminal: Terminal display for messages
        """
        self.system_config = system_config
        self.image_shm_name = image_shm_name
        self.detection_shm_name = detection_shm_name
        self.control_shm_name = control_shm_name
        self.jpeg_quality = jpeg_quality
        self.raw_rgb = raw_rgb
        self.broadcast_log_interval = broadcast_log_interval
        self.verbose = verbose
        self.terminal = terminal

        # Broker
        self.broker = None

        # Shared memory channels (lazy initialization)
        self.image_channel = None
        self.detection_channel = None
        self.control_channel = None

        # Tracking
        self.last_broadcast_frame_id = -1
        self.last_broadcast_detection_id = -1

        # FPS tracking
        self.broadcast_frame_count = 0
        self.broadcast_detection_count = 0
        self.last_fps_update_time = time.time()
        self.current_frame_fps = 0.0
        self.current_detection_fps = 0.0

    def setup(self) -> bool:
        """
        Setup ZMQ broker.

        Returns:
            True if successful, False otherwise
        """
        try:
            from lkas.integration.zmq import LKASBroker

            self.terminal.print("Initializing ZMQ broker (routing & broadcasting)...")
            self.broker = LKASBroker(verbose=self.verbose)
            self.terminal.print("")
            return True

        except Exception as e:
            self.terminal.print(f" Failed to initialize ZMQ broker: {e}")
            self.terminal.print("  Continuing without broker...")
            self.broker = None
            return False

    def announce_ready(self):
        """Announce that broadcasting is ready."""
        self.terminal.print("\n Broadcasting enabled - will connect to shared memory when available")

    def _connect_image_channel(self) -> bool:
        """Lazy connection to image channel."""
        try:
            from lkas.integration.shared_memory import SharedMemoryImageChannel

            # Suppress stdout during connection
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self.image_channel = SharedMemoryImageChannel(
                    name=self.image_shm_name,
                    shape=(self.system_config.camera.height, self.system_config.camera.width, 3),
                    create=False,  # Reader mode
                    retry_count=1,
                    retry_delay=0.0,
                )
            finally:
                sys.stdout = old_stdout

            return True

        except Exception:
            return False

    def _connect_detection_channel(self) -> bool:
        """Lazy connection to detection channel."""
        try:
            from lkas.integration.shared_memory import SharedMemoryDetectionChannel

            self.detection_channel = SharedMemoryDetectionChannel(
                name=self.detection_shm_name,
                create=False,  # Reader mode
                retry_count=1,
                retry_delay=0.0,
            )
            return True

        except Exception:
            return False

    def _connect_control_channel(self) -> bool:
        """Lazy connection to control channel."""
        try:
            from lkas.integration.shared_memory import SharedMemoryControlChannel

            self.control_channel = SharedMemoryControlChannel(
                name=self.control_shm_name,
                create=False,  # Reader mode
                retry_count=1,
                retry_delay=0.0,
            )
            return True

        except Exception:
            return False

    def _broadcast_frame(self):
        """Read and broadcast frame from shared memory."""
        # Lazy connect
        if self.image_channel is None:
            if not self._connect_image_channel():
                return

        try:
            image_msg = self.image_channel.read(copy=False)
            if image_msg is not None and image_msg.frame_id != self.last_broadcast_frame_id:
                self.broker.broadcast_frame(
                    image_msg.image,
                    image_msg.frame_id,
                    jpeg_quality=self.jpeg_quality,
                    raw_rgb=self.raw_rgb
                )
                self.last_broadcast_frame_id = image_msg.frame_id
                self.broadcast_frame_count += 1

        except Exception:
            pass  # Silently ignore

    def _broadcast_detection(self):
        """Read and broadcast detection from shared memory."""
        # Lazy connect
        if self.detection_channel is None:
            if not self._connect_detection_channel():
                return

        if self.control_channel is None:
            self._connect_control_channel()

        try:
            detection_msg = self.detection_channel.read()
            control_msg = None

            if self.control_channel:
                control_msg = self.control_channel.read()

            if detection_msg is not None and detection_msg.frame_id != self.last_broadcast_detection_id:
                # Convert to viewer format
                detection_data = self._build_detection_data(detection_msg, control_msg)

                self.broker.broadcast_detection(detection_data, detection_msg.frame_id)
                self.last_broadcast_detection_id = detection_msg.frame_id
                self.broadcast_detection_count += 1

                # Log if verbose
                if self.verbose and detection_msg.frame_id % self.broadcast_log_interval == 0:
                    self._log_broadcast(detection_msg, detection_data, control_msg)

        except Exception as e:
            self.terminal.print(f"Warning: Failed to broadcast detection: {e}")

    def _build_detection_data(self, detection_msg, control_msg) -> Dict:
        """Build detection data for broadcasting."""
        detection_data = {
            'left_lane': {
                'x1': float(detection_msg.left_lane.x1),
                'y1': float(detection_msg.left_lane.y1),
                'x2': float(detection_msg.left_lane.x2),
                'y2': float(detection_msg.left_lane.y2),
                'confidence': float(detection_msg.left_lane.confidence)
            } if detection_msg.left_lane is not None else None,
            'right_lane': {
                'x1': float(detection_msg.right_lane.x1),
                'y1': float(detection_msg.right_lane.y1),
                'x2': float(detection_msg.right_lane.x2),
                'y2': float(detection_msg.right_lane.y2),
                'confidence': float(detection_msg.right_lane.confidence)
            } if detection_msg.right_lane is not None else None,
            'processing_time_ms': detection_msg.processing_time_ms,
            'frame_id': detection_msg.frame_id,
        }

        # Add metrics from control message
        if control_msg is not None:
            detection_data['lateral_offset_meters'] = control_msg.lateral_offset_meters
            detection_data['heading_angle_deg'] = control_msg.heading_angle
            detection_data['lane_width_pixels'] = control_msg.lane_width_pixels
            detection_data['departure_status'] = control_msg.departure_status
        else:
            detection_data['lateral_offset_meters'] = None
            detection_data['heading_angle_deg'] = None
            detection_data['lane_width_pixels'] = None
            detection_data['departure_status'] = None

            if self.verbose and detection_msg.frame_id % self.broadcast_log_interval == 0:
                self.terminal.print(f"[Broker] Warning: No control data for frame {detection_msg.frame_id}")

        return detection_data

    def _log_broadcast(self, detection_msg, detection_data, control_msg):
        """Log broadcast details."""
        has_metrics = control_msg is not None
        if has_metrics:
            self.terminal.print(
                f"[Broker] Frame {detection_msg.frame_id}: "
                f"L:{detection_msg.left_lane is not None}, R:{detection_msg.right_lane is not None} | "
                f"Metrics: offset={detection_data['lateral_offset_meters']:.3f}m, "
                f"status={detection_data['departure_status']}"
            )
        else:
            self.terminal.print(
                f"[Broker] Frame {detection_msg.frame_id}: "
                f"L:{detection_msg.left_lane is not None}, R:{detection_msg.right_lane is not None} | "
                f"Metrics: N/A"
            )

    def _update_fps(self):
        """Update FPS statistics."""
        now = time.time()
        if now - self.last_fps_update_time >= 1.0:
            elapsed = now - self.last_fps_update_time
            self.current_frame_fps = self.broadcast_frame_count / elapsed
            self.current_detection_fps = self.broadcast_detection_count / elapsed

            # Update footer with FPS stats
            self.terminal.update_footer(fps_stats={
                'Frame': self.current_frame_fps,
                'Detection': self.current_detection_fps
            })

            self.broadcast_frame_count = 0
            self.broadcast_detection_count = 0
            self.last_fps_update_time = now

    def broadcast_data(self):
        """Main broadcasting loop - call periodically."""
        if not self.broker:
            return

        self._broadcast_frame()
        self._broadcast_detection()
        self._update_fps()

    def poll_broker(self):
        """Poll broker for parameter updates and action requests."""
        if self.broker:
            self.broker.poll()

    def close(self):
        """Cleanup resources."""
        # Close shared memory readers
        if self.image_channel:
            try:
                self.image_channel.close()
            except Exception as e:
                if self.verbose:
                    self.terminal.print(f"  (Image channel close: {e})")

        if self.detection_channel:
            try:
                self.detection_channel.close()
            except Exception as e:
                if self.verbose:
                    self.terminal.print(f"  (Detection channel close: {e})")

        # Stop broker
        if self.broker:
            self.broker.close()
            self.broker = None
