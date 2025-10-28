"""
Main Integration Script for CARLA Lane Detection (OOP VERSION)

This is the CLEAN version using proper OOP architecture.

KEY IMPROVEMENTS vs original main.py:
✅ Configuration-driven (no hardcoded values)
✅ Factory pattern for detector creation
✅ Single Responsibility Principle (each class does ONE thing)
✅ Type-safe data models (Lane, LaneMetrics)
✅ Separation of concerns (processing, UI, control)
✅ Easy to test and maintain
✅ Well-documented with Python OOP explanations

For C++ developers:
    This demonstrates clean architecture in Python.
    Each class is like a C++ class with single responsibility.
    Uses composition (has-a) instead of god class anti-pattern.
"""

import cv2
import numpy as np
import argparse
import sys

# Import OOP components
from core.config import Config, ConfigManager
from core.factory import DetectorFactory
from core.models import LaneDepartureStatus

# Import processing components
from processing.frame_processor import FrameProcessor
from processing.pd_controller import PDController
from processing.metrics_logger import MetricsLogger

# Import UI components
from ui.keyboard_handler import KeyboardHandler
from ui.video_recorder import VideoRecorder

# Import existing components (still used)
from carla_interface import CARLAInterface
from utils.lane_analyzer import LaneAnalyzer
from utils.visualizer import LKASVisualizer
from utils.spectator_overlay import SpectatorOverlay


class LaneKeepingAssist:
    """
    Lane Keeping Assist System.

    CLEAN ARCHITECTURE:
        This class is SIMPLE - just orchestrates components!
        No longer a "God Class" doing everything.

    COMPOSITION:
        - HAS A frame processor (not IS A processor)
        - HAS A controller (not IS A controller)
        - HAS A keyboard handler
        - HAS A video recorder

    SINGLE RESPONSIBILITY:
        - Initialize components from config
        - Run main loop
        - Coordinate components
        That's it! Much cleaner than 500+ line original.
    """

    def __init__(self, config: Config, args: argparse.Namespace):
        """
        Initialize LKAS with configuration.

        DEPENDENCY INJECTION:
            Config is passed in (not created here)
            Makes testing easier!

        Args:
            config: System configuration
            args: Command line arguments
        """
        self.config = config
        self.args = args

        # Initialize CARLA interface
        print("Initializing CARLA interface...")
        # Use args if provided, otherwise use config
        carla_host = args.host if args.host else config.carla.host
        carla_port = args.port if args.port else config.carla.port

        self.carla = CARLAInterface(host=carla_host, port=carla_port)

        # Create lane detector using factory pattern
        print(f"Creating {args.method.upper()} detector...")
        factory = DetectorFactory(config)
        detector = factory.create(
            args.method, model_path=args.model if args.method == "dl" else None
        )
        print(f"✓ Detector created: {detector.get_name()}")

        # Create analysis and visualization components
        analyzer = LaneAnalyzer(
            image_width=config.camera.width,
            image_height=config.camera.height,
            drift_threshold=config.analyzer.drift_threshold,
            departure_threshold=config.analyzer.departure_threshold,
        )

        visualizer = LKASVisualizer(
            image_width=config.camera.width, image_height=config.camera.height
        )

        # Create processing components (NEW!)
        self.processor = FrameProcessor(detector, analyzer, visualizer)
        self.controller = PDController(kp=config.controller.kp, kd=config.controller.kd)
        self.metrics_logger = MetricsLogger()

        # Create UI components (NEW!)
        self.keyboard = KeyboardHandler()
        self.video_recorder = VideoRecorder(output_path=args.save_video, fps=30.0)

        # Spectator overlay (initialized after CARLA connection)
        self.spectator_overlay = None

        # State variables
        self.running = False
        self.show_spectator_overlay = config.visualization.show_spectator_overlay
        self.follow_with_spectator = config.visualization.follow_with_spectator

    def setup_carla(self) -> bool:
        """
        Setup CARLA simulation.

        Returns:
            True if successful
        """
        # Connect to CARLA
        if not self.carla.connect():
            return False

        # Spawn vehicle
        if not self.carla.spawn_vehicle(
            self.config.carla.vehicle_type, spawn_point_index=self.args.spawn_point
        ):
            return False

        # Setup camera using config (no hardcoded values!)
        if not self.carla.setup_camera(
            image_width=self.config.camera.width,
            image_height=self.config.camera.height,
            fov=self.config.camera.fov,
            position=self.config.camera.position,
            rotation=self.config.camera.rotation,
        ):
            return False

        # Enable autopilot for testing
        self.carla.set_vehicle_autopilot(True)

        # Initialize spectator overlay
        self.spectator_overlay = SpectatorOverlay(self.carla.world)

        # Start video recording if requested
        if self.args.save_video:
            self.video_recorder.start(
                self.config.camera.width, self.config.camera.height
            )

        print("✓ CARLA setup complete!")
        return True

    def setup_keyboard_controls(self):
        """
        Setup keyboard controls using callback pattern.

        CLEAN SEPARATION:
            Keyboard handling is separate from main logic!
            Easy to add/remove/modify controls.
        """
        # Register keyboard actions
        self.keyboard.register("q", self._quit, "Quit")
        self.keyboard.register("s", self._toggle_autopilot, "Toggle autopilot")
        self.keyboard.register("o", self._toggle_spectator, "Toggle spectator overlay")
        self.keyboard.register("f", self._toggle_follow, "Toggle spectator follow")
        self.keyboard.register("r", self._respawn_vehicle, "Respawn vehicle")
        self.keyboard.register("t", self._teleport, "Teleport to spawn point")

        # Print help
        # print("\n" + "=" * 60)
        # print("Keyboard Controls:")
        # print("=" * 60)
        # print("  'q' → Quit")
        # print("  's' → Toggle autopilot")
        # print("  'o' → Toggle spectator overlay")
        # print("  'f' → Toggle spectator follow mode")
        # print("  'r' → Respawn vehicle")
        # print("  't' → Teleport to next spawn point")
        # print("=" * 60 + "\n")

    def run(self):
        """
        Run the LKAS main loop.

        CLEAN LOOP:
            Much simpler than original!
            Each step delegates to responsible component.
        """
        self.running = True
        self.setup_keyboard_controls()

        print("Lane Keeping Assist System - Running")
        print("Starting main loop...\n")

        try:
            while self.running:
                # Get image from CARLA
                image = self.carla.get_latest_image()
                if image is None:
                    continue

                # Process frame (detection → analysis → visualization)
                # All done by FrameProcessor!
                vis_image, metrics, _ = self.processor.process(image)

                # Compute steering correction
                # Done by PDController!
                steering = self.controller.compute_steering(metrics)

                # Update spectator overlay (if enabled)
                if self.show_spectator_overlay and self.spectator_overlay:
                    self._update_spectator_overlay(metrics, steering)

                # Update metrics logger
                self.metrics_logger.update_fps()
                self.metrics_logger.log_frame(metrics)

                # Add FPS to display
                cv2.putText(
                    vis_image,
                    f"FPS: {self.metrics_logger.get_fps():.1f}",
                    (self.config.camera.width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Display visualization
                if not self.args.no_display:
                    cv2.imshow("Lane Keeping Assist", vis_image)

                # Record video if enabled
                if self.video_recorder.is_recording:
                    self.video_recorder.write(vis_image)

                # Handle keyboard input
                self.keyboard.handle(wait_ms=1)

                # Print metrics periodically
                if self.metrics_logger.total_frames % 30 == 0:
                    self.metrics_logger.print_metrics(metrics, steering)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self._cleanup()

    def _update_spectator_overlay(self, metrics, steering):
        """Update CARLA spectator camera overlay."""
        # Get color based on departure status
        departure_color = self._get_departure_color(metrics.departure_status)

        # Draw vehicle position
        self.spectator_overlay.draw_vehicle_position(
            self.carla.vehicle, color=departure_color, lifetime=0.1
        )

        # Add info text
        info_text = f"Lane: {metrics.departure_status.value}"
        self.spectator_overlay.draw_vehicle_info_text(
            self.carla.vehicle, additional_info=info_text, lifetime=0.1
        )

        # Update spectator camera if follow mode enabled
        if self.follow_with_spectator:
            self.spectator_overlay.update_spectator_camera(
                self.carla.vehicle, follow_distance=20.0, height=10.0
            )

    def _get_departure_color(self, status: LaneDepartureStatus):
        """Get CARLA color for departure status."""
        import carla

        if status == LaneDepartureStatus.CENTERED:
            return carla.Color(0, 255, 0)  # Green
        elif status in [
            LaneDepartureStatus.LEFT_DRIFT,
            LaneDepartureStatus.RIGHT_DRIFT,
        ]:
            return carla.Color(255, 255, 0)  # Yellow
        elif status in [
            LaneDepartureStatus.LEFT_DEPARTURE,
            LaneDepartureStatus.RIGHT_DEPARTURE,
        ]:
            return carla.Color(255, 0, 0)  # Red
        else:
            return carla.Color(255, 255, 255)  # White

    def _cleanup(self):
        """Cleanup resources."""
        print("\n\nShutting down...")
        self.video_recorder.stop()
        cv2.destroyAllWindows()
        self.carla.cleanup()
        print("✓ System shutdown complete")

    # =========================================================================
    # KEYBOARD ACTION CALLBACKS
    # These are called by KeyboardHandler when keys are pressed
    # =========================================================================

    def _quit(self):
        """Quit the application."""
        print("\nQuitting...")
        self.running = False

    def _toggle_autopilot(self):
        """Toggle CARLA autopilot."""
        current = self.carla.is_autopilot_enabled()
        self.carla.set_vehicle_autopilot(not current)
        status = "ON" if not current else "OFF"
        print(f"\nAutopilot: {status}")

    def _toggle_spectator(self):
        """Toggle spectator overlay."""
        self.show_spectator_overlay = not self.show_spectator_overlay
        status = "ON" if self.show_spectator_overlay else "OFF"
        print(f"\nSpectator overlay: {status}")

    def _toggle_follow(self):
        """Toggle spectator follow mode."""
        self.follow_with_spectator = not self.follow_with_spectator
        status = "ON" if self.follow_with_spectator else "OFF"
        print(f"\nSpectator follow mode: {status}")

    def _respawn_vehicle(self):
        """Respawn vehicle at current location."""
        print("\nRespawning vehicle...")
        if self.carla.respawn_vehicle():
            # Reattach camera
            self.carla.setup_camera(
                image_width=self.config.camera.width,
                image_height=self.config.camera.height,
                fov=self.config.camera.fov,
                position=self.config.camera.position,
                rotation=self.config.camera.rotation,
            )
            print("✓ Vehicle respawned")
        else:
            print("✗ Respawn failed")

    def _teleport(self):
        """Teleport to next spawn point."""
        print("\nTeleporting...")
        if self.carla.teleport_to_spawn_point():
            print("✓ Teleport successful")
        else:
            print("✗ Teleport failed")


def main():
    """
    Main entry point.

    CLEAN ENTRY POINT:
        1. Parse arguments
        2. Load configuration
        3. Create LKAS
        4. Run

    Much simpler than original!
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Lane Keeping Assist System for CARLA (OOP Version)"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="cv",
        choices=["cv", "dl"],
        help="Lane detection method (cv=Computer Vision, dl=Deep Learning)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--host", type=str, default=None, help="CARLA server host (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="CARLA server port (overrides config)"
    )
    parser.add_argument(
        "--spawn-point", type=int, default=None, help="Spawn point index"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to pretrained DL model"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable visualization display"
    )
    parser.add_argument(
        "--save-video", type=str, default=None, help="Path to save output video"
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = ConfigManager.load(args.config)

    # Override config with command line arguments
    if args.host:
        config.carla.host = args.host
    if args.port:
        config.carla.port = args.port

    # Set detection method in config
    config.detection_method = args.method

    # Print configuration
    print(f"✓ Configuration loaded")
    print(f"  CARLA: {config.carla.host}:{config.carla.port}")
    print(f"  Camera: {config.camera.width}x{config.camera.height}")
    print(f"  Detection method: {config.detection_method}")
    print(f"  Controller gains: Kp={config.controller.kp}, Kd={config.controller.kd}")

    # Create LKAS instance
    print("\nCreating Lane Keeping Assist System...")
    lkas = LaneKeepingAssist(config, args)

    # Setup CARLA
    print("\nConnecting to CARLA...")
    if not lkas.setup_carla():
        print("✗ Failed to setup CARLA. Exiting.")
        return 1

    # Run system
    lkas.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
