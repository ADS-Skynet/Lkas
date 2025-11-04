"""
Simulation Orchestrator

Clean, testable orchestrator that coordinates all simulation components.
Follows Single Responsibility Principle and Dependency Inversion.
"""

import time
import signal
import sys
from typing import Optional, Callable
from dataclasses import dataclass

from detection.core.config import ConfigManager
from simulation import CARLAConnection, VehicleManager, CameraSensor
from decision import DecisionController
from detection.integration.messages import ControlMessage, ControlMode
from simulation.integration.zmq_broadcast import (
    VehicleBroadcaster,
    ActionSubscriber,
    DetectionData,
    VehicleState,
)
from detection.integration.shared_memory_detection import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionClient,
)
from core.constants import SimulationConstants


@dataclass
class SimulationConfig:
    """Configuration for simulation orchestrator."""
    carla_host: str
    carla_port: int
    spawn_point: Optional[int]
    image_shm_name: str
    detection_shm_name: str
    detector_timeout: int
    enable_broadcast: bool
    broadcast_url: str
    action_url: str
    enable_autopilot: bool
    enable_sync_mode: bool
    base_throttle: float
    warmup_frames: int
    enable_latency_tracking: bool


class SimulationOrchestrator:
    """
    Orchestrates the lane keeping simulation.

    Responsibilities:
    - Initialize all subsystems (CARLA, detection, control)
    - Coordinate main simulation loop
    - Handle cleanup on shutdown

    This class is testable by injecting dependencies.
    """

    def __init__(self, config: SimulationConfig, system_config: object):
        """
        Initialize orchestrator.

        Args:
            config: Simulation configuration
            system_config: System-wide configuration (camera, controller, etc.)
        """
        self.config = config
        self.system_config = system_config

        # Subsystems (initialized in setup methods)
        self.carla_conn: Optional[CARLAConnection] = None
        self.vehicle_mgr: Optional[VehicleManager] = None
        self.camera: Optional[CameraSensor] = None
        self.image_writer: Optional[SharedMemoryImageChannel] = None
        self.detector: Optional[SharedMemoryDetectionClient] = None
        self.controller: Optional[DecisionController] = None
        self.broadcaster: Optional[VehicleBroadcaster] = None
        self.action_subscriber: Optional[ActionSubscriber] = None

        # State
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.timeouts = 0

    def setup(self) -> bool:
        """
        Setup all subsystems.

        Returns:
            True if setup successful
        """
        if not self._setup_carla():
            return False

        if not self._setup_vehicle():
            return False

        if not self._setup_camera():
            return False

        if not self._setup_detection():
            return False

        self._setup_controller()

        if self.config.enable_broadcast:
            self._setup_broadcasting()

        self._setup_event_handlers()

        return True

    def _setup_carla(self) -> bool:
        """Setup CARLA connection."""
        print("\n[1/5] Connecting to CARLA...")
        self.carla_conn = CARLAConnection(self.config.carla_host, self.config.carla_port)

        if not self.carla_conn.connect():
            return False

        # Setup synchronous mode
        if self.config.enable_sync_mode:
            self.carla_conn.setup_synchronous_mode(
                enabled=True,
                fixed_delta_seconds=SimulationConstants.FIXED_DELTA_SECONDS
            )
        else:
            print("✓ Running in asynchronous mode")

        return True

    def _setup_vehicle(self) -> bool:
        """Setup vehicle."""
        print("\n[2/5] Spawning vehicle...")
        self.vehicle_mgr = VehicleManager(self.carla_conn.get_world())

        if not self.vehicle_mgr.spawn_vehicle(
            self.system_config.carla.vehicle_type,
            self.config.spawn_point
        ):
            return False

        if self.config.enable_autopilot:
            self.vehicle_mgr.set_autopilot(True)
            print("✓ Autopilot enabled")

        return True

    def _setup_camera(self) -> bool:
        """Setup camera sensor."""
        print("\n[3/5] Setting up camera...")
        self.camera = CameraSensor(self.carla_conn.get_world(), self.vehicle_mgr.get_vehicle())

        return self.camera.setup_camera(
            width=self.system_config.camera.width,
            height=self.system_config.camera.height,
            fov=self.system_config.camera.fov,
            position=self.system_config.camera.position,
            rotation=self.system_config.camera.rotation,
        )

    def _setup_detection(self) -> bool:
        """Setup detection communication."""
        print("\n[4/5] Setting up detection communication...")

        try:
            print("Setting up shared memory communication...")
            print("(Both processes can start in any order - will retry if needed)")

            # Create image writer
            print("\nCreating shared memory image writer...")
            self.image_writer = SharedMemoryImageChannel(
                name=self.config.image_shm_name,
                shape=(
                    self.system_config.camera.height,
                    self.system_config.camera.width,
                    3
                ),
                create=True,
                retry_count=SimulationConstants.DEFAULT_RETRY_COUNT,
                retry_delay=SimulationConstants.RETRY_DELAY_SECONDS
            )

            # Connect to detection results
            print("\nConnecting to detection results...")
            self.detector = SharedMemoryDetectionClient(
                detection_shm_name=self.config.detection_shm_name,
                retry_count=SimulationConstants.DEFAULT_RETRY_COUNT,
                retry_delay=SimulationConstants.RETRY_DELAY_SECONDS
            )

            print("\n✓ Shared memory communication ready")
            return True

        except Exception as e:
            print(f"\n✗ Failed to setup shared memory: {e}")
            print(f"  Tip: Make sure detection server is running")
            print(f"       Both processes can start in any order, but both must be running.")
            return False

    def _setup_controller(self):
        """Setup decision controller."""
        print("\n[5/5] Setting up controller...")

        throttle_policy = {
            "base": self.system_config.throttle_policy.base,
            "min": self.system_config.throttle_policy.min,
            "steer_threshold": self.system_config.throttle_policy.steer_threshold,
            "steer_max": self.system_config.throttle_policy.steer_max,
        }

        self.controller = DecisionController(
            image_width=self.system_config.camera.width,
            image_height=self.system_config.camera.height,
            kp=self.system_config.controller.kp,
            kd=self.system_config.controller.kd,
            throttle_policy=throttle_policy,
        )

        print(f"✓ Controller ready")
        print(f"  Adaptive throttle: base={self.system_config.throttle_policy.base}, "
              f"min={self.system_config.throttle_policy.min}")

    def _setup_broadcasting(self):
        """Setup ZMQ broadcasting for remote viewer."""
        print("\nInitializing ZMQ broadcaster for remote viewer...")
        self.broadcaster = VehicleBroadcaster(bind_url=self.config.broadcast_url)
        self.action_subscriber = ActionSubscriber(bind_url=self.config.action_url)
        print("✓ ZMQ broadcaster ready")

    def _setup_event_handlers(self):
        """Setup event handlers for actions."""
        if not self.action_subscriber:
            return

        # Register action handlers
        self.action_subscriber.register_action('respawn', self._handle_respawn)
        self.action_subscriber.register_action('pause', self._handle_pause)
        self.action_subscriber.register_action('resume', self._handle_resume)

        print("\n✓ Action handlers registered")
        print("  Actions: respawn, pause, resume")

    def _handle_respawn(self) -> bool:
        """Handle respawn action."""
        print("\n🔄 Respawn requested")
        try:
            if self.vehicle_mgr.teleport_to_spawn_point(self.config.spawn_point):
                print("✓ Vehicle respawned successfully")
                return True
            else:
                print("✗ Failed to respawn vehicle")
                return False
        except Exception as e:
            print(f"✗ Respawn error: {e}")
            return False

    def _handle_pause(self) -> bool:
        """Handle pause action."""
        self.paused = True
        print("\n⏸ Paused - simulation loop will freeze")
        return True

    def _handle_resume(self) -> bool:
        """Handle resume action."""
        self.paused = False
        print("\n▶️ Resumed - simulation loop continues")
        return True

    def run(self):
        """
        Run main simulation loop.

        This is the main entry point after setup.
        """
        self.running = True
        self._register_signal_handlers()

        print("\n" + "=" * 60)
        print("System Running")
        print("Press Ctrl+C to quit")
        print("=" * 60 + "\n")

        last_print = time.time()

        try:
            while self.running:
                # Poll for actions
                if self.action_subscriber:
                    self.action_subscriber.poll()

                # Check if paused
                if self.paused:
                    time.sleep(SimulationConstants.PAUSE_SLEEP_SECONDS)
                    continue

                # Tick world (if sync mode)
                if self.config.enable_sync_mode:
                    self.carla_conn.get_world().tick()

                # Get image from camera
                image = self.camera.get_latest_image()
                if image is None:
                    continue

                # Send to detector and get result
                self.image_writer.write(image, timestamp=time.time(), frame_id=self.frame_count)
                detection = self.detector.get_detection(
                    timeout=self.config.detector_timeout / 1000.0
                )

                # Process detection and compute control
                control = self._process_detection(detection)

                # Apply control
                if not self.vehicle_mgr.is_autopilot_enabled():
                    self.vehicle_mgr.apply_control(
                        control.steering,
                        control.throttle,
                        control.brake
                    )

                # Broadcast to remote viewers
                if self.broadcaster:
                    self._broadcast_data(image, detection, control)

                self.frame_count += 1

                # Print status periodically
                if self.frame_count % SimulationConstants.STATUS_PRINT_INTERVAL_FRAMES == 0:
                    self._print_status(last_print, detection, control)
                    last_print = time.time()

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.cleanup()

    def _process_detection(self, detection):
        """Process detection result and compute control."""
        # Check if we have valid detection
        has_valid_detection = detection is not None and (
            detection.left_lane is not None or detection.right_lane is not None
        )

        if not has_valid_detection:
            self.timeouts += 1
            # No detection: use base throttle to keep moving
            return ControlMessage(
                steering=0.0,
                throttle=self.config.base_throttle,
                brake=0.0,
                mode=ControlMode.LANE_KEEPING,
            )
        else:
            # Valid detection: use controller
            return self.controller.process_detection(detection)

    def _broadcast_data(self, image, detection, control):
        """Broadcast data to remote viewers."""
        # Send image
        self.broadcaster.send_frame(image, self.frame_count)

        # Send detection
        if detection:
            detection_data = DetectionData(
                left_lane={
                    'x1': float(detection.left_lane.x1),
                    'y1': float(detection.left_lane.y1),
                    'x2': float(detection.left_lane.x2),
                    'y2': float(detection.left_lane.y2),
                    'confidence': float(detection.left_lane.confidence)
                } if detection.left_lane else None,
                right_lane={
                    'x1': float(detection.right_lane.x1),
                    'y1': float(detection.right_lane.y1),
                    'x2': float(detection.right_lane.x2),
                    'y2': float(detection.right_lane.y2),
                    'confidence': float(detection.right_lane.confidence)
                } if detection.right_lane else None,
                processing_time_ms=detection.processing_time_ms if hasattr(detection, 'processing_time_ms') else 0.0,
                frame_id=self.frame_count
            )
            self.broadcaster.send_detection(detection_data)

        # Send vehicle state
        velocity = self.vehicle_mgr.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 if velocity else 0.0

        vehicle_state = VehicleState(
            steering=float(control.steering),
            throttle=float(control.throttle),
            brake=float(control.brake),
            speed_kmh=float(speed_ms * 3.6),
            position=None,
            rotation=None
        )
        self.broadcaster.send_state(vehicle_state)

    def _print_status(self, last_print, detection, control):
        """Print status line."""
        fps = SimulationConstants.STATUS_PRINT_INTERVAL_FRAMES / (time.time() - last_print)

        # Lane status
        if detection is None:
            lanes = "TIMEOUT"
        else:
            lanes = f"{'L' if detection.left_lane else '-'}{'R' if detection.right_lane else '-'}"

        # Detection info
        detection_info = ""
        if detection is not None and hasattr(detection, 'processing_time_ms'):
            detection_info = f" | Det: {detection.processing_time_ms:.1f}ms"

        status_line = (
            f"Frame {self.frame_count:5d} | FPS: {fps:5.1f} | Lanes: {lanes}{detection_info} | "
            f"Steering: {control.steering:+.3f} | Throttle: {control.throttle:.2f} | Timeouts: {self.timeouts}"
        )

        print(f"\r{status_line}", end="", flush=True)

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            print("\n\nReceived interrupt signal")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def cleanup(self):
        """Cleanup all resources."""
        print("\nCleaning up...")

        if self.detector:
            self.detector.close()

        if self.image_writer:
            self.image_writer.close()
            self.image_writer.unlink()

        if self.camera:
            self.camera.destroy_camera()

        if self.vehicle_mgr:
            self.vehicle_mgr.destroy_vehicle()

        if self.carla_conn:
            self.carla_conn.disconnect()

        print("✓ Shutdown complete")
