"""
Distributed System Orchestrator

CARLA client that uses a remote detection server.
This is the production-ready architecture for ML deployment.

Architecture:
    This Process (CARLA Client) <--ZMQ--> Detection Server Process
    - CARLA connection              - Lane detection
    - Vehicle control                - ML models
    - Decision making                - GPU processing
    - Fast control loop              - Can be on different machine
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from typing import Optional
import cv2

from simulation import CARLAConnection, VehicleManager, CameraSensor
from decision import DecisionController
from detection.integration.communication import DetectionClient
from detection.integration.messages import (
    ImageMessage,
    DetectionMessage,
    ControlMessage,
    SystemStatus,
    PerformanceMetrics,
    ControlMode,
)
from detection.core.config import Config


class DistributedOrchestrator:
    """
    Orchestrator for distributed architecture.

    Coordinates:
    1. CARLA Module: Local (provides images, accepts control)
    2. Detection Module: Remote (processes images via network)
    3. Decision Module: Local (generates control commands)

    Data flow:
        CARLA (image) --network--> Remote Detection
                                          ↓
                                    (lanes)
                                          ↓
        Decision (control) <-------- (lanes)
                ↓
        CARLA (actuate)
    """

    def __init__(self, config: Config, detector_url: str = "tcp://localhost:5555"):
        """
        Initialize distributed orchestrator.

        Args:
            config: System configuration
            detector_url: URL of remote detection server
        """
        self.config = config
        self.detector_url = detector_url

        # System status
        self.status = SystemStatus()
        self.running = False

        # Performance metrics
        self.metrics = PerformanceMetrics()
        self._last_time = time.time()
        self._frame_times = []

        # Module instances
        self.carla_connection: Optional[CARLAConnection] = None
        self.vehicle_manager: Optional[VehicleManager] = None
        self.camera_sensor: Optional[CameraSensor] = None
        self.detection_client: Optional[DetectionClient] = None  # Remote!
        self.decision_controller: Optional[DecisionController] = None

        # Network statistics
        self.network_timeouts = 0
        self.network_errors = 0

    def initialize(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        spawn_point: Optional[int] = None,
        detector_timeout_ms: int = 1000,
    ) -> bool:
        """
        Initialize all components.

        Args:
            carla_host: CARLA server host
            carla_port: CARLA server port
            spawn_point: Vehicle spawn point index
            detector_timeout_ms: Detection request timeout

        Returns:
            True if all components initialized successfully
        """
        print("\n" + "=" * 60)
        print("Initializing Distributed Lane Keeping System")
        print("=" * 60)
        print(f"Architecture: CARLA Client <--> Detection Server")
        print(f"Detection Server: {self.detector_url}")

        # 1. Initialize CARLA connection
        print("\n[1/5] Connecting to CARLA...")
        self.carla_connection = CARLAConnection(carla_host, carla_port)
        if not self.carla_connection.connect():
            return False
        self.status.carla_connected = True

        # 2. Spawn vehicle
        print("\n[2/5] Spawning vehicle...")
        self.vehicle_manager = VehicleManager(self.carla_connection.get_world())
        if not self.vehicle_manager.spawn_vehicle(
            vehicle_type=self.config.carla.vehicle_type, spawn_point_index=spawn_point
        ):
            return False
        self.status.vehicle_spawned = True

        # 3. Setup camera
        print("\n[3/5] Setting up camera...")
        self.camera_sensor = CameraSensor(
            self.carla_connection.get_world(), self.vehicle_manager.get_vehicle()
        )
        if not self.camera_sensor.setup_camera(
            width=self.config.camera.width,
            height=self.config.camera.height,
            fov=self.config.camera.fov,
            position=self.config.camera.position,
            rotation=self.config.camera.rotation,
        ):
            return False
        self.status.camera_ready = True

        # 4. Connect to remote detection server
        print("\n[4/5] Connecting to detection server...")
        try:
            self.detection_client = DetectionClient(
                server_url=self.detector_url, timeout_ms=detector_timeout_ms
            )
            self.status.detector_ready = True
        except Exception as e:
            print(f"✗ Failed to connect to detection server: {e}")
            print(f"  Make sure detection server is running:")
            print(
                f"  python detection_server.py --port {self.detector_url.split(':')[-1]}"
            )
            return False

        # 5. Initialize decision controller
        print("\n[5/5] Initializing decision controller...")
        self.decision_controller = DecisionController(
            image_width=self.config.camera.width,
            image_height=self.config.camera.height,
            kp=self.config.controller.kp,
            kd=self.config.controller.kd,
        )
        self.status.controller_ready = True

        print("\n" + "=" * 60)
        print("✓ All components initialized successfully")
        print("=" * 60)
        print(f"\n📡 Network: CARLA Client <--> Detection Server")
        print(f"   Detection: {self.detector_url}")
        print(f"   Timeout: {detector_timeout_ms}ms")

        return True

    def run(self, enable_autopilot: bool = True, show_visualization: bool = True):
        """
        Run the main control loop.

        Args:
            enable_autopilot: Enable CARLA autopilot for vehicle movement
            show_visualization: Show visualization window
        """
        if not self.status.is_ready:
            print("✗ System not ready. Call initialize() first.")
            return

        # Enable autopilot if requested
        if enable_autopilot:
            self.vehicle_manager.set_autopilot(True)
            print("\n✓ Autopilot enabled")

        self.running = True
        print("\n" + "=" * 60)
        print("System Running - Press 'q' to quit")
        print("=" * 60 + "\n")

        try:
            while self.running:
                # Process one cycle
                self._process_cycle(show_visualization)

                # Update FPS
                self._update_fps()

                # Handle keyboard
                if show_visualization:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\nQuitting...")
                        break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self.shutdown()

    def _process_cycle(self, show_visualization: bool = True):
        """Process one cycle of the control loop."""
        # Step 1: Get image from CARLA
        image = self.camera_sensor.get_latest_image()
        if image is None:
            return

        # Create image message
        image_msg = ImageMessage(
            image=image, timestamp=time.time(), frame_id=self.metrics.total_frames
        )

        # Step 2: Send to remote detection server and receive lanes
        detection_start = time.time()
        detection_msg = self.detection_client.detect(image_msg)
        detection_time = (time.time() - detection_start) * 1000

        # Handle detection failure (timeout or error)
        if detection_msg is None:
            self.network_timeouts += 1
            # Use safe default: no steering, apply brake
            control_msg = ControlMessage(
                steering=0.0, throttle=0.0, brake=0.3, mode=ControlMode.LANE_KEEPING
            )
        else:
            self.metrics.detection_time_ms = detection_time

            # Step 3: Generate control commands
            control_start = time.time()
            control_msg = self.decision_controller.process_detection(detection_msg)
            control_time = (time.time() - control_start) * 1000
            self.metrics.control_time_ms = control_time

        # Step 4: Apply control to vehicle (only if not in autopilot mode)
        if not self.vehicle_manager.is_autopilot_enabled():
            self.vehicle_manager.apply_control(
                steering=control_msg.steering,
                throttle=control_msg.throttle,
                brake=control_msg.brake,
            )

        # Step 5: Visualization (local only, don't send over network)
        if show_visualization:
            # Create simple visualization since we don't have debug image from server
            vis_image = image.copy()

            # Draw detected lanes if available
            if detection_msg and detection_msg.left_lane:
                cv2.line(
                    vis_image,
                    (detection_msg.left_lane.x1, detection_msg.left_lane.y1),
                    (detection_msg.left_lane.x2, detection_msg.left_lane.y2),
                    (255, 0, 0),
                    5,
                )

            if detection_msg and detection_msg.right_lane:
                cv2.line(
                    vis_image,
                    (detection_msg.right_lane.x1, detection_msg.right_lane.y1),
                    (detection_msg.right_lane.x2, detection_msg.right_lane.y2),
                    (0, 0, 255),
                    5,
                )

            # Add metrics
            cv2.putText(
                vis_image,
                f"FPS: {self.metrics.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis_image,
                f"Network: {detection_time:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis_image,
                f"Steering: {control_msg.steering:.3f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if self.network_timeouts > 0:
                cv2.putText(
                    vis_image,
                    f"Timeouts: {self.network_timeouts}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Lane Keeping System (Distributed)", vis_image)

        # Update metrics
        self.metrics.total_frames += 1

        # Print periodic status
        if self.metrics.total_frames % 30 == 0:
            self._print_status(detection_msg, control_msg)

    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        delta = current_time - self._last_time
        self._frame_times.append(delta)

        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        avg_time = sum(self._frame_times) / len(self._frame_times)
        self.metrics.fps = 1.0 / avg_time if avg_time > 0 else 0.0
        self._last_time = current_time

    def _print_status(
        self, detection: Optional[DetectionMessage], control: ControlMessage
    ):
        """Print periodic status update."""
        lanes_str = (
            "TIMEOUT"
            if detection is None
            else (
                f"{'L' if detection.left_lane else '-'}"
                f"{'R' if detection.right_lane else '-'}"
            )
        )
        print(
            f"Frame {self.metrics.total_frames:5d} | "
            f"FPS: {self.metrics.fps:5.1f} | "
            f"Lanes: {lanes_str} | "
            f"Network: {self.metrics.detection_time_ms:5.1f}ms | "
            f"Steering: {control.steering:+.3f} | "
            f"Timeouts: {self.network_timeouts}"
        )

    def shutdown(self):
        """Shutdown all components and cleanup resources."""
        print("\n" + "=" * 60)
        print("Shutting down distributed system...")
        print("=" * 60)

        if self.detection_client:
            self.detection_client.close()

        if self.camera_sensor:
            self.camera_sensor.destroy_camera()

        if self.vehicle_manager:
            self.vehicle_manager.destroy_vehicle()

        if self.carla_connection:
            self.carla_connection.disconnect()

        cv2.destroyAllWindows()

        print(f"\nNetwork Statistics:")
        print(f"  Total frames: {self.metrics.total_frames}")
        print(f"  Timeouts: {self.network_timeouts}")
        print(
            f"  Success rate: {(1 - self.network_timeouts/max(1, self.metrics.total_frames))*100:.1f}%"
        )

        print("\n✓ System shutdown complete")
