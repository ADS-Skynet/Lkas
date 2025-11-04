"""
ZMQ Pub-Sub Broadcasting for Web Viewer

Uses ZMQ PUB-SUB pattern for non-blocking communication between:
- Vehicle/Simulation (Publisher) → Web Viewer (Subscriber)

Benefits:
- Non-blocking: Vehicle doesn't wait for viewer
- Fire-and-forget: Works even if viewer disconnects
- Multiple subscribers: Can have multiple viewers
- Network-capable: Works across machines (WiFi/Ethernet)

Topics:
- 'frame': Video frames with overlay metadata
- 'state': Vehicle state (steering, speed, etc.)
- 'detection': Lane detection results
- 'action': Commands from viewer (respawn, pause, etc.)
"""

import zmq
import numpy as np
import json
import time
import cv2
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable
from threading import Thread


@dataclass
class FrameData:
    """Frame data sent to viewer."""
    image_jpeg: bytes  # JPEG compressed image
    timestamp: float
    frame_id: int
    width: int
    height: int


@dataclass
class DetectionData:
    """Lane detection results."""
    left_lane: Optional[Dict[str, float]]  # {x1, y1, x2, y2, confidence}
    right_lane: Optional[Dict[str, float]]
    processing_time_ms: float
    frame_id: int


@dataclass
class VehicleState:
    """Vehicle/simulation state."""
    steering: float  # -1.0 to 1.0
    throttle: float  # 0.0 to 1.0
    brake: float     # 0.0 to 1.0
    speed_kmh: float
    position: Optional[tuple] = None  # (x, y, z)
    rotation: Optional[tuple] = None  # (pitch, yaw, roll)


class VehicleBroadcaster:
    """
    Publisher: Broadcasts vehicle data to remote viewers.

    Runs on vehicle/simulation. Sends frames, detections, and state.
    """

    def __init__(self, bind_url: str = "tcp://*:5557"):
        """
        Initialize broadcaster.

        Args:
            bind_url: ZMQ URL to bind publisher socket
        """
        self.bind_url = bind_url

        # Create ZMQ context and publisher socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(bind_url)

        # High water mark (max queued messages)
        self.socket.setsockopt(zmq.SNDHWM, 10)  # Drop old frames if viewer is slow

        print(f"✓ Vehicle broadcaster started on {bind_url}")
        print(f"  Topics: frame, detection, state, action")

        # Stats
        self.frame_count = 0
        self.last_print_time = time.time()

        # Give ZMQ time to establish connection (slow joiner problem)
        time.sleep(0.1)

    def send_frame(self, image: np.ndarray, frame_id: int, jpeg_quality: int = 85):
        """
        Send frame to viewers.

        Args:
            image: RGB image array
            frame_id: Frame sequence number
            jpeg_quality: JPEG compression quality (0-100)
        """
        # Compress to JPEG (10x smaller for network transfer)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        success, buffer = cv2.imencode('.jpg', image_bgr, encode_param)

        if not success:
            print("⚠ Failed to encode frame")
            return

        # Create message
        message = {
            'timestamp': time.time(),
            'frame_id': frame_id,
            'width': image.shape[1],
            'height': image.shape[0],
            'jpeg_size': len(buffer)
        }

        # Send topic + JSON metadata + JPEG data
        self.socket.send_multipart([
            b'frame',
            json.dumps(message).encode('utf-8'),
            buffer.tobytes()
        ])

        self.frame_count += 1

        # Print stats every 3 seconds
        if time.time() - self.last_print_time > 3.0:
            fps = self.frame_count / (time.time() - self.last_print_time)
            print(f"\r[Broadcaster] {fps:.1f} FPS | Frame {frame_id} | {len(buffer)/1024:.1f} KB", end="", flush=True)
            self.frame_count = 0
            self.last_print_time = time.time()

    def send_detection(self, detection: DetectionData):
        """Send detection results to viewers."""
        message = asdict(detection)

        self.socket.send_multipart([
            b'detection',
            json.dumps(message).encode('utf-8')
        ])

    def send_state(self, state: VehicleState):
        """Send vehicle state to viewers."""
        message = asdict(state)

        self.socket.send_multipart([
            b'state',
            json.dumps(message).encode('utf-8')
        ])

    def close(self):
        """Close broadcaster."""
        self.socket.close()
        self.context.term()
        print("✓ Vehicle broadcaster stopped")


class ViewerSubscriber:
    """
    Subscriber: Receives vehicle data in web viewer.

    Runs on laptop. Receives frames, detections, and state.
    """

    def __init__(self, connect_url: str = "tcp://localhost:5557"):
        """
        Initialize subscriber.

        Args:
            connect_url: ZMQ URL to connect to publisher
        """
        self.connect_url = connect_url

        # Create ZMQ context and subscriber socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(connect_url)

        # Subscribe to all topics
        self.socket.setsockopt(zmq.SUBSCRIBE, b'frame')
        self.socket.setsockopt(zmq.SUBSCRIBE, b'detection')
        self.socket.setsockopt(zmq.SUBSCRIBE, b'state')

        # Non-blocking receive (don't wait if no data)
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        # High water mark
        self.socket.setsockopt(zmq.RCVHWM, 10)

        print(f"✓ Viewer subscriber connected to {connect_url}")

        # Callbacks
        self.frame_callback: Optional[Callable] = None
        self.detection_callback: Optional[Callable] = None
        self.state_callback: Optional[Callable] = None

        # Latest data
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detection: Optional[DetectionData] = None
        self.latest_state: Optional[VehicleState] = None

        # Stats
        self.frame_count = 0
        self.last_print_time = time.time()

    def register_frame_callback(self, callback: Callable[[np.ndarray, Dict], None]):
        """Register callback for frame updates."""
        self.frame_callback = callback

    def register_detection_callback(self, callback: Callable[[DetectionData], None]):
        """Register callback for detection updates."""
        self.detection_callback = callback

    def register_state_callback(self, callback: Callable[[VehicleState], None]):
        """Register callback for state updates."""
        self.state_callback = callback

    def poll(self) -> bool:
        """
        Poll for new messages (non-blocking).

        Returns:
            True if received message, False if no data
        """
        try:
            # Receive topic and message
            parts = self.socket.recv_multipart(zmq.NOBLOCK)

            topic = parts[0].decode('utf-8')

            if topic == 'frame':
                metadata = json.loads(parts[1].decode('utf-8'))
                jpeg_data = parts[2]

                # Decode JPEG
                image_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                self.latest_frame = image_rgb
                self.frame_count += 1

                # Print stats
                if time.time() - self.last_print_time > 3.0:
                    fps = self.frame_count / (time.time() - self.last_print_time)
                    print(f"[Subscriber] Receiving {fps:.1f} FPS | Frame {metadata['frame_id']}", end='\r', flush=True)
                    print('\n')
                    self.frame_count = 0
                    self.last_print_time = time.time()

                # Call callback
                if self.frame_callback:
                    self.frame_callback(image_rgb, metadata)

            elif topic == 'detection':
                data = json.loads(parts[1].decode('utf-8'))
                detection = DetectionData(**data)
                self.latest_detection = detection

                if self.detection_callback:
                    self.detection_callback(detection)

            elif topic == 'state':
                data = json.loads(parts[1].decode('utf-8'))
                state = VehicleState(**data)
                self.latest_state = state

                if self.state_callback:
                    self.state_callback(state)

            return True

        except zmq.Again:
            # No message available
            return False
        except Exception as e:
            print(f"⚠ Error receiving message: {e}")
            return False

    def run_loop(self):
        """Run polling loop in current thread."""
        print("Subscriber loop started (Ctrl+C to stop)")
        try:
            while True:
                self.poll()
                time.sleep(0.001)  # Small sleep to prevent busy-wait
        except KeyboardInterrupt:
            print("\nStopping subscriber...")

    def close(self):
        """Close subscriber."""
        self.socket.close()
        self.context.term()
        print("✓ Viewer subscriber stopped")


class ActionPublisher:
    """
    Publisher: Sends actions from web viewer to vehicle.

    Runs on laptop. Sends commands like respawn, pause, etc.
    """

    def __init__(self, connect_url: str = "tcp://localhost:5558"):
        """
        Initialize action publisher.

        Args:
            connect_url: ZMQ URL to connect to action subscriber
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(connect_url)

        # Give ZMQ time to establish
        time.sleep(0.1)

        print(f"✓ Action publisher connected to {connect_url}")

    def send_action(self, action: str, params: Optional[Dict[str, Any]] = None):
        """
        Send action command.

        Args:
            action: Action name (e.g., 'respawn', 'pause')
            params: Optional parameters
        """
        message = {
            'action': action,
            'params': params or {},
            'timestamp': time.time()
        }

        self.socket.send_multipart([
            b'action',
            json.dumps(message).encode('utf-8')
        ])

        print(f"[Action] Sent: {action}")

    def close(self):
        """Close publisher."""
        self.socket.close()
        self.context.term()


class ActionSubscriber:
    """
    Subscriber: Receives actions on vehicle/simulation.

    Runs on vehicle. Receives commands from web viewer.
    """

    def __init__(self, bind_url: str = "tcp://*:5558"):
        """Initialize action subscriber."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(bind_url)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'action')
        self.socket.setsockopt(zmq.RCVTIMEO, 100)

        print(f"✓ Action subscriber listening on {bind_url}")

        # Callbacks
        self.action_callbacks: Dict[str, Callable] = {}

    def register_action(self, action: str, callback: Callable):
        """Register callback for action."""
        self.action_callbacks[action] = callback
        print(f"  Registered action: {action}")

    def poll(self) -> bool:
        """Poll for action commands."""
        try:
            parts = self.socket.recv_multipart(zmq.NOBLOCK)
            topic = parts[0].decode('utf-8')
            data = json.loads(parts[1].decode('utf-8'))

            if topic == 'action':
                action = data['action']
                params = data.get('params', {})

                print(f"[Action] Received: {action}")

                if action in self.action_callbacks:
                    self.action_callbacks[action](**params)
                else:
                    print(f"  ⚠ Unknown action: {action}")

            return True

        except zmq.Again:
            return False
        except Exception as e:
            print(f"⚠ Error receiving action: {e}")
            return False

    def close(self):
        """Close subscriber."""
        self.socket.close()
        self.context.term()


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python zmq_broadcast.py broadcaster  # Run on vehicle")
        print("  python zmq_broadcast.py viewer       # Run on laptop")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "broadcaster":
        # Simulate vehicle broadcasting
        broadcaster = VehicleBroadcaster()

        try:
            frame_id = 0
            while True:
                # Generate test frame
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.putText(image, f"Frame {frame_id}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Broadcast
                broadcaster.send_frame(image, frame_id)

                # Simulate detection
                detection = DetectionData(
                    left_lane={'x1': 100, 'y1': 400, 'x2': 200, 'y2': 100, 'confidence': 0.9},
                    right_lane={'x1': 500, 'y1': 400, 'x2': 600, 'y2': 100, 'confidence': 0.85},
                    processing_time_ms=15.5,
                    frame_id=frame_id
                )
                broadcaster.send_detection(detection)

                # Simulate state
                state = VehicleState(
                    steering=0.1,
                    throttle=0.5,
                    brake=0.0,
                    speed_kmh=25.0
                )
                broadcaster.send_state(state)

                frame_id += 1
                time.sleep(0.033)  # 30 FPS

        except KeyboardInterrupt:
            broadcaster.close()

    elif mode == "viewer":
        # Simulate viewer receiving
        subscriber = ViewerSubscriber()

        def on_frame(image, metadata):
            print(f"  Received frame {metadata['frame_id']}: {image.shape}")

        def on_detection(detection):
            print(f"  Detection: {detection.processing_time_ms:.1f}ms")

        def on_state(state):
            print(f"  State: speed={state.speed_kmh:.1f} km/h, steering={state.steering:.2f}")

        subscriber.register_frame_callback(on_frame)
        subscriber.register_detection_callback(on_detection)
        subscriber.register_state_callback(on_state)

        try:
            subscriber.run_loop()
        except KeyboardInterrupt:
            subscriber.close()
