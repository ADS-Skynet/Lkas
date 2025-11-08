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
from rich.console import Console
from rich.live import Live
from rich.table import Table


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
    paused: Optional[bool] = None  # Simulation paused state


@dataclass
class ParameterUpdate:
    """Parameter update message for real-time tuning."""
    category: str  # 'detection' or 'decision'
    parameter: str  # Parameter name (e.g., 'canny_low', 'kp')
    value: float    # New parameter value
    timestamp: float = 0.0


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
        self.current_fps = 0.0
        self.current_frame_id = 0
        self.current_kb = 0.0

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

        # Update stats every 3 seconds
        if time.time() - self.last_print_time > 3.0:
            fps = self.frame_count / (time.time() - self.last_print_time)
            self.current_fps = fps
            self.current_frame_id = frame_id
            self.current_kb = len(buffer) / 1024.0
            # Suppress print - orchestrator will display in footer
            # print(f"\r[Broadcaster] {fps:.1f} FPS | Frame {frame_id} | {len(buffer)/1024:.1f} KB", end="", flush=True)
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

    def get_stats(self) -> dict:
        """Get current broadcaster statistics."""
        return {
            'fps': self.current_fps,
            'frame_id': self.current_frame_id,
            'kb': self.current_kb
        }

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
        self.current_fps = 0.0
        self.current_frame_id = 0
        self.paused = False
        self.state_received = False  # Track if we've received any state yet

        # Rich console for footer
        self.console = Console()
        self.live_display: Optional[Live] = None

        # Initialize footer immediately
        self._init_footer()

    def _init_footer(self):
        """Initialize rich live footer display."""
        if self.live_display is None:
            # Create live display with auto-refresh
            self.live_display = Live(
                self._generate_footer_table(),
                console=self.console,
                refresh_per_second=2,
                vertical_overflow="visible"
            )
            self.live_display.start()

    def _generate_footer_table(self) -> Table:
        """Generate footer display as a rich Table."""
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)

        if not self.state_received:
            # Initial state - waiting for first state from simulation
            table.add_row(
                f"[bold dim]○ Waiting for stream...[/bold dim]",
            )
        elif self.paused:
            # Simulation is paused
            table.add_row(
                f"[bold yellow]■ PAUSED[/bold yellow]",
            )
        else:
            # Simulation is running
            table.add_row(
                f"[bold green]● LIVE[/bold green] [bold cyan]{self.current_fps:.1f} FPS[/bold cyan]",
            )

        return table

    def _update_footer(self, fps: float, frame_id: int):
        """Update footer with new stats."""
        self.current_fps = fps
        self.current_frame_id = frame_id

        if self.live_display is not None:
            self.live_display.update(self._generate_footer_table())

    def set_paused(self, paused: bool):
        """Set pause status and update footer immediately."""
        self.paused = paused
        if self.live_display is not None:
            self.live_display.update(self._generate_footer_table())

    def _clear_footer(self):
        """Stop and clear the footer display."""
        if self.live_display is not None:
            try:
                self.live_display.stop()
            except Exception:
                pass
            finally:
                self.live_display = None
                # Print newline to ensure clean terminal state
                print()

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

                # Update footer stats
                if time.time() - self.last_print_time > 0.5:  # Update every 500ms
                    elapsed = time.time() - self.last_print_time
                    fps = self.frame_count / elapsed

                    # Update footer with new stats
                    self._update_footer(fps, metadata['frame_id'])

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
                self.state_received = True

                # Update pause state from vehicle if provided
                if state.paused is not None and state.paused != self.paused:
                    self.set_paused(state.paused)

                if self.state_callback:
                    self.state_callback(state)

            return True

        except zmq.Again:
            # No message available
            return False
        except Exception as e:
            # print(f"⚠ Error receiving message: {e}")
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
        finally:
            self._clear_footer()

    def close(self):
        """Close subscriber."""
        # Ensure footer is cleared before closing
        if self.live_display is not None:
            try:
                self.live_display.stop()
                self.live_display = None
            except Exception:
                pass

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

    def close(self):
        """Close publisher."""
        self.socket.close()
        self.context.term()


class ActionSubscriber:
    """
    Subscriber: Receives actions on vehicle/simulation.

    Runs on vehicle. Receives commands from web viewer or LKAS broker.
    """

    def __init__(self, bind_url: str = "tcp://*:5558", connect_mode: bool = False):
        """
        Initialize action subscriber.

        Args:
            bind_url: ZMQ URL to bind/connect to
            connect_mode: If True, connect as client (receive from LKAS broker).
                         If False, bind as server (old architecture, for backward compatibility).
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        if connect_mode:
            # Connect to LKAS broker (new architecture)
            # Convert tcp://*:5558 to tcp://localhost:5561 (LKAS action forward port)
            connect_url = "tcp://localhost:5561"  # LKAS broker forwards to this port
            self.socket.connect(connect_url)
            print(f"✓ Action subscriber connected to {connect_url}")
        else:
            # Bind as server (old architecture, for backward compatibility)
            self.socket.bind(bind_url)
            print(f"✓ Action subscriber listening on {bind_url}")

        self.socket.setsockopt(zmq.SUBSCRIBE, b'action')
        self.socket.setsockopt(zmq.RCVTIMEO, 100)

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

                if action in self.action_callbacks:
                    result = self.action_callbacks[action](**params)
                else:
                    print(f"[ActionSubscriber] ⚠ Unknown action: {action}")
                    print(f"[ActionSubscriber] Available actions: {list(self.action_callbacks.keys())}")

            return True

        except zmq.Again:
            return False

        except Exception as e:
            print(f"[ActionSubscriber] ⚠ Error receiving action: {e}")

            import traceback
            traceback.print_exc()
            return False

    def close(self):
        """Close subscriber."""
        self.socket.close()
        self.context.term()


class ParameterPublisher:
    """
    Publisher: Sends parameter updates from viewer to vehicle/simulation.

    Runs in viewer process. Publishes real-time parameter adjustments.
    """

    def __init__(self, bind_url: str = "tcp://*:5559", connect_mode: bool = False):
        """
        Initialize parameter publisher.

        Args:
            bind_url: ZMQ URL to bind/connect to
            connect_mode: If True, connect as client. If False, bind as server (default)
                         Use connect_mode=True when LKAS broker is running (new architecture)
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)

        if connect_mode:
            # Connect to LKAS broker (new architecture)
            # Convert tcp://*:5559 to tcp://localhost:5559 for connect
            connect_url = bind_url.replace("tcp://*:", "tcp://localhost:")
            self.socket.connect(connect_url)
        else:
            # Bind as server (old architecture, for backward compatibility)
            self.socket.bind(bind_url)

        # Give ZMQ time to establish
        time.sleep(0.1)

    def send_parameter(self, category: str, parameter: str, value: float):
        """
        Send parameter update.

        Args:
            category: 'detection' or 'decision'
            parameter: Parameter name
            value: New value
        """
        try:
            if self.socket.closed:
                return

            update = ParameterUpdate(
                category=category,
                parameter=parameter,
                value=value,
                timestamp=time.time()
            )

            message = asdict(update)

            self.socket.send_multipart([
                b'parameter',
                json.dumps(message).encode('utf-8')
            ], flags=zmq.NOBLOCK)
        except zmq.error.Again:
            # Socket is full, skip this message
            pass
        except Exception as e:
            # Don't crash on send errors
            print(f"[ParameterPublisher] Error sending parameter: {e}")

    def close(self):
        """Close publisher."""
        try:
            if not self.socket.closed:
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.close()
            self.context.term()
        except:
            pass


class ParameterSubscriber:
    """
    Subscriber: Receives parameter updates in detection/decision servers.

    Runs in detection and decision servers. Applies parameter changes in real-time.
    """

    def __init__(self, connect_url: str = "tcp://localhost:5559"):
        """
        Initialize parameter subscriber.

        Args:
            connect_url: ZMQ URL to connect to parameter broker
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(connect_url)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'parameter')
        self.socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout for non-blocking

        print(f"✓ Parameter subscriber connected to {connect_url}")

        # Callbacks for different categories
        self.parameter_callbacks: Dict[str, Callable[[str, float], None]] = {}

    def register_callback(self, category: str, callback: Callable[[str, float], None]):
        """
        Register callback for parameter updates.

        Args:
            category: Category to listen for ('detection' or 'decision')
            callback: Function(parameter_name, value) to handle updates
        """
        self.parameter_callbacks[category] = callback
        print(f"  Registered parameter callback for: {category}")

    def poll(self) -> bool:
        """
        Poll for parameter updates (non-blocking).

        Returns:
            True if received update, False otherwise
        """
        try:
            parts = self.socket.recv_multipart(zmq.NOBLOCK)
            topic = parts[0].decode('utf-8')
            data = json.loads(parts[1].decode('utf-8'))

            if topic == 'parameter':
                update = ParameterUpdate(**data)

                # Call registered callback for this category
                if update.category in self.parameter_callbacks:
                    self.parameter_callbacks[update.category](
                        update.parameter,
                        update.value
                    )
                    return True

            return False

        except zmq.Again:
            # No message available
            return False
        except Exception as e:
            # print(f"⚠ Error receiving parameter: {e}")
            return False

    def close(self):
        """Close subscriber."""
        self.socket.close()
        self.context.term()


class VehicleStatusPublisher:
    """
    Publisher: Sends vehicle status TO the LKAS broker.

    Runs in simulation/orchestrator. Sends vehicle state to LKAS broker
    which then broadcasts to all viewers.
    """

    def __init__(self, lkas_broker_url: str = "tcp://localhost:5562"):
        """
        Initialize vehicle status publisher.

        Args:
            lkas_broker_url: URL to send vehicle status to LKAS broker
        """
        self.context = zmq.Context()

        # Publisher socket (connects to LKAS broker's subscriber)
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(lkas_broker_url)
        self.pub_socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
        self.pub_socket.setsockopt(zmq.SNDHWM, 10)  # High water mark

        time.sleep(0.1)  # Let socket establish

        print(f"✓ Vehicle status publisher connected to LKAS broker: {lkas_broker_url}")

    def send_state(self, state: VehicleState):
        """
        Send vehicle state to LKAS broker.

        Args:
            state: Vehicle state data
        """
        try:
            if self.pub_socket.closed:
                return

            message = asdict(state)

            # Send multipart: [topic, json_data]
            self.pub_socket.send_multipart([
                b'vehicle_status',
                json.dumps(message).encode('utf-8')
            ], flags=zmq.NOBLOCK)

        except zmq.Again:
            # Socket full, skip message (prefer real-time over buffering)
            pass
        except Exception as e:
            # Silently ignore send errors (broker might not be ready yet)
            pass

    def close(self):
        """Close publisher."""
        if self.pub_socket:
            self.pub_socket.close()
        if self.context:
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
