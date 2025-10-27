"""
System Orchestrator

Coordinates the three modules: CARLA, Detection, and Decision.
Manages the data flow: CARLA → Detection → Decision → CARLA
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from typing import Optional
import cv2

from simulation import CARLAConnection, VehicleManager, CameraSensor
from detection.detection_module import LaneDetectionModule
from decision import DecisionController
from detection.integration.messages import (
    ImageMessage,
    DetectionMessage,
    ControlMessage,
    SystemStatus,
    PerformanceMetrics,
    ControlMode,
)
from detection.core.config import Config


class SystemOrchestrator:
    """
    Main orchestrator for the lane keeping system.

    Coordinates the three modules:
    1. CARLA Module: Provides images, accepts control commands
    2. Detection Module: Processes images, detects lanes
    3. Decision Module: Analyzes lanes, generates control commands

    Data flow:
        CARLA (image) → Detection (lanes) → Decision (control) → CARLA (actuate)
    """

    def __init__(self, config: Config, detection_method: str = "cv"):
        """
        Initialize system orchestrator.

        Args:
            config: System configuration
            detection_method: Lane detection method ('cv' or 'dl')
        """
        self.config = config
        self.detection_method = detection_method

        # System status
        self.status = SystemStatus()
        self.running = False

        # Performance metrics
        self.metrics = PerformanceMetrics()
        self._last_time = time.time()
        self._frame_times = []

        # Module instances (initialized later)
        self.carla_connection: Optional[CARLAConnection] = None
        self.vehicle_manager: Optional[VehicleManager] = None
        self.camera_sensor: Optional[CameraSensor] = None
        self.detection_module: Optional[LaneDetectionModule] = None
        self.decision_controller: Optional[DecisionController] = None

    def initialize(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        spawn_point: Optional[int] = None,
    ) -> bool:
        """
        Initialize all modules.

        Args:
            carla_host: CARLA server host
            carla_port: CARLA server port
            spawn_point: Vehicle spawn point index

        Returns:
            True if all modules initialized successfully
        """
        print("\n" + "=" * 60)
        print("Initializing Lane Keeping System")
        print("=" * 60)

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

        # 4. Initialize detection module
        print("\n[4/5] Initializing detection module...")
        self.detection_module = LaneDetectionModule(self.config, self.detection_method)
        self.status.detector_ready = True

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
        print("✓ All modules initialized successfully")
        print("=" * 60)

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

        # Enable autopilot if requested (for automatic driving)
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

        # Step 2: Detect lanes
        detection_start = time.time()
        detection_msg = self.detection_module.process_image(image_msg)
        detection_time = (time.time() - detection_start) * 1000
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

        # Step 5: Visualization
        if show_visualization and detection_msg.debug_image is not None:
            vis_image = detection_msg.debug_image.copy()

            # Add FPS and metrics
            cv2.putText(
                vis_image,
                f"FPS: {self.metrics.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis_image,
                f"Detection: {detection_time:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis_image,
                f"Steering: {control_msg.steering:.3f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Lane Keeping System", vis_image)

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

        # Keep last 30 frames for averaging
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        avg_time = sum(self._frame_times) / len(self._frame_times)
        self.metrics.fps = 1.0 / avg_time if avg_time > 0 else 0.0
        self._last_time = current_time

    def _print_status(self, detection: DetectionMessage, control: ControlMessage):
        """Print periodic status update."""
        print(
            f"Frame {self.metrics.total_frames:5d} | "
            f"FPS: {self.metrics.fps:5.1f} | "
            f"Lanes: {'L' if detection.left_lane else '-'}"
            f"{'R' if detection.right_lane else '-'} | "
            f"Steering: {control.steering:+.3f} | "
            f"Offset: {control.lateral_offset:.3f if control.lateral_offset else 'N/A'}"
        )

    def shutdown(self):
        """Shutdown all modules and cleanup resources."""
        print("\n" + "=" * 60)
        print("Shutting down system...")
        print("=" * 60)

        if self.camera_sensor:
            self.camera_sensor.destroy_camera()

        if self.vehicle_manager:
            self.vehicle_manager.destroy_vehicle()

        if self.carla_connection:
            self.carla_connection.disconnect()

        cv2.destroyAllWindows()

        print("✓ System shutdown complete")

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        return self.status

    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self.metrics
