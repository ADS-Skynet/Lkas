"""
Communication Protocol for Distributed Architecture

Uses ZMQ (ZeroMQ) for fast, lightweight message passing between:
- CARLA Client (sends images)
- Detection Server (returns lane detections)

ZMQ is chosen because:
- Fast: Near-native socket performance
- Simple: No broker needed
- Flexible: Multiple transport layers (TCP, IPC, inproc)
- Reliable: Built-in reconnection
"""

import zmq
import numpy as np
import time
import json

from dataclasses import asdict

from detection.integration.messages import ImageMessage, DetectionMessage, LaneMessage


class InputSerial:
    """
    Efficient image serialization for network transfer.

    Optimizations:
    - JPEG compression for RGB images (10x smaller)
    - Metadata in JSON (fast parsing)
    - Single network packet when possible
    """

    @staticmethod
    def serialize(image_msg: ImageMessage) -> bytes:
        """
        Serialize image message to bytes.

        Format:
        - First 4 bytes: Metadata length (uint32)
        - Next N bytes: JSON metadata
        - Remaining bytes: JPEG compressed image

        Args:
            image_msg: Image message to serialize

        Returns:
            Serialized bytes
        """
        import cv2

        # Compress image to JPEG (much smaller for network transfer)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, image_buffer = cv2.imencode('.jpg', image_msg.image, encode_param)
        image_bytes = image_buffer.tobytes()

        # Create metadata
        metadata = {
            'timestamp': image_msg.timestamp,
            'frame_id': image_msg.frame_id,
            'width': image_msg.width,
            'height': image_msg.height,
            'compressed': True,
            'format': 'jpeg'
        }
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_len = len(metadata_json)

        # Pack: [metadata_len (4 bytes)][metadata][image]
        message = (
            metadata_len.to_bytes(4, 'little') +
            metadata_json +
            image_bytes
        )

        return message

    @staticmethod
    def deserialize(message_bytes: bytes) -> ImageMessage:
        """
        Deserialize bytes to image message.

        Args:
            message_bytes: Serialized message

        Returns:
            ImageMessage instance
        """
        import cv2

        # Unpack metadata length
        metadata_len = int.from_bytes(message_bytes[:4], 'little')

        # Unpack metadata
        metadata_json = message_bytes[4:4+metadata_len]
        metadata = json.loads(metadata_json.decode('utf-8'))

        # Unpack image
        image_bytes = message_bytes[4+metadata_len:]

        # Decompress JPEG
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR

        # Create message
        return ImageMessage(
            image=image,
            timestamp=metadata['timestamp'],
            frame_id=metadata['frame_id']
        )


class OutputSerial:
    """
    Serialization for detection results.

    Much simpler than images - just lane coordinates in JSON.
    """

    @staticmethod
    def serialize(detection_msg: DetectionMessage) -> bytes:
        """Serialize detection message to JSON bytes."""
        data = {
            'frame_id': detection_msg.frame_id,
            'timestamp': detection_msg.timestamp,
            'processing_time_ms': detection_msg.processing_time_ms,
            'left_lane': None,
            'right_lane': None
        }

        if detection_msg.left_lane:
            data['left_lane'] = {
                'x1': detection_msg.left_lane.x1,
                'y1': detection_msg.left_lane.y1,
                'x2': detection_msg.left_lane.x2,
                'y2': detection_msg.left_lane.y2,
                'confidence': detection_msg.left_lane.confidence
            }

        if detection_msg.right_lane:
            data['right_lane'] = {
                'x1': detection_msg.right_lane.x1,
                'y1': detection_msg.right_lane.y1,
                'x2': detection_msg.right_lane.x2,
                'y2': detection_msg.right_lane.y2,
                'confidence': detection_msg.right_lane.confidence
            }

        return json.dumps(data).encode('utf-8')

    @staticmethod
    def deserialize(message_bytes: bytes) -> DetectionMessage:
        """Deserialize JSON bytes to detection message."""
        data = json.loads(message_bytes.decode('utf-8'))

        left_lane = None
        if data['left_lane']:
            left_lane = LaneMessage(**data['left_lane'])

        right_lane = None
        if data['right_lane']:
            right_lane = LaneMessage(**data['right_lane'])

        return DetectionMessage(
            left_lane=left_lane,
            right_lane=right_lane,
            processing_time_ms=data['processing_time_ms'],
            frame_id=data['frame_id'],
            timestamp=data['timestamp'],
            debug_image=None  # Don't send debug image over network (too large)
        )


class DetectionClient:
    """
    Client for sending images to remote detection server.

    Used by CARLA client to communicate with detection server.
    """

    def __init__(self, server_url: str = "tcp://localhost:5555", timeout_ms: int = 1000,
                 max_retries: int = 5, retry_delay: float = 2.0):
        """
        Initialize detection client.

        Args:
            server_url: ZMQ URL of detection server
            timeout_ms: Request timeout in milliseconds
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retry attempts (seconds)
        """
        self.server_url = server_url
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create ZMQ context
        self.context = zmq.Context()
        self.socket = None

        # Try to connect with retries
        self._connect_with_retry()

    def detect(self, image_msg: ImageMessage) -> DetectionMessage | None:
        """
        Send image to server and receive detection results.

        Args:
            image_msg: Image to process

        Returns:
            Detection results or None if timeout/error
        """
        try:
            # Serialize and send image
            message_bytes = InputSerial.serialize(image_msg)
            self.socket.send(message_bytes)

            # Receive detection results
            response_bytes = self.socket.recv()
            detection = OutputSerial.deserialize(response_bytes)

            return detection

        except zmq.Again:
            # Timeout
            print(f"⚠ Detection timeout ({self.timeout_ms}ms)")
            # Need to recreate socket after timeout
            self._reconnect()
            return None

        except Exception as e:
            print(f"⚠ Detection error: {e}")
            self._reconnect()
            return None

    def _create_socket(self):
        """Create and configure a new ZMQ socket."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.server_url)

    def _test_connection(self) -> bool:
        """
        Test if server is actually reachable by sending a ping message.

        Returns:
            True if server responds, False otherwise
        """
        try:
            # Send a small test message (ping)
            test_msg = json.dumps({"type": "ping"}).encode('utf-8')
            self.socket.send(test_msg)

            # Wait for response
            response = self.socket.recv()
            response_data = json.loads(response.decode('utf-8'))

            if response_data.get("type") == "pong":
                return True
            else:
                print(f"  ⚠ Unexpected response: {response_data}")
                return False

        except zmq.Again:
            # Timeout - server not responding
            return False
        except Exception as e:
            print(f"  ⚠ Connection test error: {e}")
            return False

    def _connect_with_retry(self):
        """Connect to server with retry logic."""
        print(f"\n[DetectionClient] Connecting to detection server...")
        print(f"  URL: {self.server_url}")
        print(f"  Timeout: {self.timeout_ms}ms")
        print(f"  Max retries: {self.max_retries}")

        for attempt in range(1, self.max_retries + 1):
            print(f"\n  Attempt {attempt}/{self.max_retries}:")
            print(f"    Creating socket...")

            # Create new socket
            self._create_socket()
            print(f"    ✓ Socket created")

            # Test connection
            print(f"    Testing connection (sending ping)...")
            if self._test_connection():
                print(f"\n✓ Successfully connected to detection server!")
                print(f"  Server responded on attempt {attempt}")
                return

            # Failed - socket is now in bad state, will recreate on next attempt
            print(f"    ✗ No response from server")

            if attempt < self.max_retries:
                print(f"    Waiting {self.retry_delay}s before retry...")
                time.sleep(self.retry_delay)

        # All retries exhausted
        raise ConnectionError(
            f"\n✗ Failed to connect to detection server after {self.max_retries} attempts\n"
            f"  URL: {self.server_url}\n"
            f"  Make sure detection server is running:\n"
            f"    python detection_server.py --port {self.server_url.split(':')[-1]}\n"
        )

    def _reconnect(self):
        """Reconnect socket after error."""
        print("  [DetectionClient] Reconnecting socket...")
        try:
            self._create_socket()
            print("  ✓ Socket recreated")
        except Exception as e:
            print(f"  ✗ Reconnect failed: {e}")

    def close(self):
        """Close connection."""
        self.socket.close()
        self.context.term()


class DetectionServer:
    """
    Server that receives images and returns lane detections.

    Used by standalone detection process.
    """

    def __init__(self, bind_url: str = "tcp://*:5555"):
        """
        Initialize detection server.

        Args:
            bind_url: ZMQ URL to bind to
        """
        self.bind_url = bind_url

        # Create ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # REP side of REQ-REP

        print(f"Starting detection server on {bind_url}...")
        self.socket.bind(bind_url)
        print(f"✓ Detection server listening on {bind_url}")

        self.running = False
        self.request_count = 0

    def serve(self, detection_callback):
        """
        Start serving detection requests.

        Args:
            detection_callback: Function that takes ImageMessage and returns DetectionMessage
        """
        self.running = True
        print("Detection server ready to receive requests...")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                # Receive message
                message_bytes = self.socket.recv()

                # Check if this is a ping message (connection test)
                try:
                    # Try to decode as JSON first (ping messages are plain JSON)
                    test_data = json.loads(message_bytes.decode('utf-8'))
                    if test_data.get("type") == "ping":
                        # Respond with pong
                        pong_response = json.dumps({"type": "pong"}).encode('utf-8')
                        self.socket.send(pong_response)
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Not a ping message, continue with normal image processing
                    pass

                self.request_count += 1

                # Deserialize image message
                image_msg = InputSerial.deserialize(message_bytes)

                # Process detection (call user's detection function)
                detection_msg = detection_callback(image_msg)

                # Serialize and send response
                response_bytes = OutputSerial.serialize(detection_msg)
                self.socket.send(response_bytes)

                # Print periodic status (refreshing)
                if self.request_count % 30 == 0:
                    print(f"\rProcessed {self.request_count} frames | "
                          f"Last: {detection_msg.processing_time_ms:.1f}ms | "
                          f"Lanes: {'L' if detection_msg.left_lane else '-'}"
                          f"{'R' if detection_msg.right_lane else '-'}",
                          end="", flush=True)

        except KeyboardInterrupt:
            print("\n\nStopping server...")
        finally:
            self.stop()

    def stop(self):
        """Stop server."""
        self.running = False
        self.socket.close()
        self.context.term()
        print("✓ Detection server stopped")
