"""
Main Integration Script for CARLA Lane Detection
Combines all components for real-time lane keeping assist.
"""

import cv2
import numpy as np
import argparse
import time
from typing import Optional

from carla_interface import CARLAInterface
from method.computer_vision.cv_lane_detector import CVLaneDetector
from method.deep_learning.lane_net import DLLaneDetector
from utils.lane_analyzer import LaneAnalyzer, LaneDepartureStatus
from utils.visualizer import LKASVisualizer
from utils.spectator_overlay import SpectatorOverlay


class LaneKeepingAssist:
    """Main Lane Keeping Assist System."""

    def __init__(
        self,
        detection_method: str = "cv",
        image_width: int = 800,
        image_height: int = 600,
        carla_host: str = "carla-server",
        carla_port: int = 2000,
        model_path: Optional[str] = None,
    ):
        """
        Initialize LKAS.

        Args:
            detection_method: 'cv' for traditional CV or 'dl' for deep learning
            image_width: Camera image width
            image_height: Camera image height
            carla_host: CARLA server host
            carla_port: CARLA server port
            model_path: Path to pretrained DL model (if using DL)
        """
        self.detection_method = detection_method
        self.image_width = image_width
        self.image_height = image_height

        # Initialize CARLA interface
        print("Initializing CARLA interface...")
        self.carla = CARLAInterface(host=carla_host, port=carla_port)

        # Initialize lane detector
        print(f"Initializing lane detector ({detection_method})...")
        if detection_method == "cv":
            self.detector = CVLaneDetector()
        else:
            # Use pre-trained model by default (no model_path needed)
            # model_type options: 'pretrained', 'simple', 'full'
            self.detector = DLLaneDetector(
                model_path=model_path,
                model_type="pretrained",  # Use pre-trained U-Net
                input_size=(256, 256)
            )

        # Initialize lane analyzer
        self.analyzer = LaneAnalyzer(
            image_width=image_width,
            image_height=image_height,
            drift_threshold=0.15,
            departure_threshold=0.35,
        )

        # Initialize visualizer
        self.visualizer = LKASVisualizer(
            image_width=image_width, image_height=image_height
        )

        # Spectator overlay (will be initialized after CARLA connection)
        self.spectator_overlay = None

        # State variables
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        # Alert state
        self.alert_blink_counter = 0

        # Feature toggles
        self.show_spectator_overlay = True
        self.follow_with_spectator = False

        # Store camera configuration for respawn
        self.camera_config = {
            "width": image_width,
            "height": image_height,
            "position": (2.0, 0.0, 1.5),
            "rotation": (-10.0, 0.0, 0.0),
            "fov": 90.0,
        }

    def setup_carla(
        self,
        vehicle_type: str = "vehicle.tesla.model3",
        spawn_point: Optional[int] = None,
    ) -> bool:
        """
        Setup CARLA simulation.

        Args:
            vehicle_type: Type of vehicle to spawn
            spawn_point: Specific spawn point index (None for auto-select)

        Returns:
            True if successful, False otherwise
        """
        # Connect to CARLA
        if not self.carla.connect():
            return False

        # Spawn vehicle
        if not self.carla.spawn_vehicle(vehicle_type, spawn_point_index=spawn_point):
            return False

        # Setup camera
        camera_position = (2.0, 0.0, 1.5)  # Forward, right, up from vehicle center
        camera_rotation = (-10.0, 0.0, 0.0)  # Pitch down slightly

        if not self.carla.setup_camera(
            image_width=self.image_width,
            image_height=self.image_height,
            fov=90.0,
            position=camera_position,
            rotation=camera_rotation,
        ):
            return False

        # Enable autopilot for testing
        self.carla.set_vehicle_autopilot(True)

        # Initialize spectator overlay
        self.spectator_overlay = SpectatorOverlay(self.carla.world)

        print("CARLA setup complete!")
        return True

    def process_frame(self, image: np.ndarray) -> tuple:
        """
        Process a single frame.

        Args:
            image: Input camera image

        Returns:
            Tuple of (visualization_image, metrics, steering_correction)
        """
        # Detect lanes
        # Both CV and DL methods now return the same format: (left_lane, right_lane, debug_image)
        left_lane, right_lane, debug_image = self.detector.detect(image)

        # Analyze lanes
        metrics = self.analyzer.get_metrics(left_lane, right_lane)

        # Calculate steering correction
        steering_correction = self.analyzer.get_steering_correction(
            left_lane, right_lane, kp=0.5, kd=0.1
        )

        # Get vehicle telemetry
        telemetry = self.carla.get_vehicle_telemetry()

        # Create visualization
        vis_image = self.visualizer.draw_lanes(image, left_lane, right_lane)
        vis_image = self.visualizer.draw_vehicle_position(
            vis_image,
            metrics["vehicle_center_x"],
            metrics["lane_center_x"],
            metrics["departure_status"],
        )
        vis_image = self.visualizer.draw_hud(
            vis_image,
            metrics,
            show_steering=True,
            steering_value=steering_correction,
            vehicle_telemetry=telemetry,
        )

        # Add alert overlay if needed
        departure_status = metrics["departure_status"]
        if departure_status in [
            LaneDepartureStatus.LEFT_DEPARTURE,
            LaneDepartureStatus.RIGHT_DEPARTURE,
        ]:
            blink = (self.alert_blink_counter % 10) < 5
            vis_image = self.visualizer.create_alert_overlay(
                vis_image, departure_status, blink=blink
            )
            self.alert_blink_counter += 1
        else:
            self.alert_blink_counter = 0

        return vis_image, metrics, steering_correction

    def update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        elapsed = current_time - self.last_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

        self.frame_count += 1

    def run(self, display: bool = True, save_video: Optional[str] = None):
        """
        Run the lane keeping assist system.

        Args:
            display: Whether to display visualization
            save_video: Path to save output video (optional)
        """
        self.running = True

        # Setup video writer if needed
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                save_video, fourcc, 30.0, (self.image_width, self.image_height)
            )

        print("\n" + "=" * 60)
        print("Lane Keeping Assist System - Running")
        print("=" * 60)
        print("Press 'q' to quit")
        print("Press 's' to toggle autopilot")
        print("Press 'o' to toggle spectator overlay")
        print("Press 'f' to toggle spectator follow mode")
        print("Press 'r' to respawn vehicle (if stuck)")
        print("Press 't' to teleport to next spawn point")
        print("=" * 60 + "\n")

        try:
            while self.running:
                # Get image from CARLA
                image = self.carla.get_latest_image()

                if image is None:
                    continue

                # Process frame
                vis_image, metrics, steering = self.process_frame(image)

                # Update spectator overlay in CARLA world
                if self.show_spectator_overlay and self.spectator_overlay:
                    departure_color = self._get_departure_color(
                        metrics["departure_status"]
                    )
                    self.spectator_overlay.draw_vehicle_position(
                        self.carla.vehicle, color=departure_color, lifetime=0.1
                    )

                    # Add telemetry text
                    telemetry = self.carla.get_vehicle_telemetry()
                    info_text = f"Lane: {metrics['departure_status'].value}"
                    self.spectator_overlay.draw_vehicle_info_text(
                        self.carla.vehicle, additional_info=info_text, lifetime=0.1
                    )

                # Update spectator follow mode
                if self.follow_with_spectator and self.spectator_overlay:
                    self.spectator_overlay.update_spectator_camera(
                        self.carla.vehicle, follow_distance=20.0, height=10.0
                    )

                # Update FPS
                self.update_fps()

                # Add FPS to display
                cv2.putText(
                    vis_image,
                    f"FPS: {self.fps:.1f}",
                    (self.image_width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Display
                if display:
                    cv2.imshow("Lane Keeping Assist", vis_image)

                # Save video
                if video_writer:
                    video_writer.write(vis_image)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                elif key == ord("s"):
                    # Toggle autopilot
                    current_autopilot = self.carla.is_autopilot_enabled()
                    self.carla.set_vehicle_autopilot(not current_autopilot)
                elif key == ord("o"):
                    # Toggle spectator overlay
                    self.show_spectator_overlay = not self.show_spectator_overlay
                    status = "ON" if self.show_spectator_overlay else "OFF"
                    print(f"\nSpectator overlay: {status}")
                elif key == ord("f"):
                    # Toggle spectator follow mode
                    self.follow_with_spectator = not self.follow_with_spectator
                    status = "ON" if self.follow_with_spectator else "OFF"
                    print(f"\nSpectator follow mode: {status}")
                elif key == ord("r"):
                    # Respawn vehicle
                    print("\n\nRespawning vehicle...")
                    if self._respawn_vehicle():
                        print("Vehicle respawned successfully!")
                    else:
                        print("Failed to respawn vehicle")
                elif key == ord("t"):
                    # Teleport to next spawn point
                    print("\n\nTeleporting to next spawn point...")
                    if self.carla.teleport_to_spawn_point():
                        print("Teleport successful!")
                    else:
                        print("Teleport failed")

                # Print metrics periodically
                if self.frame_count % 30 == 0:
                    self._print_metrics(metrics, steering)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            self.carla.cleanup()
            print("System shutdown complete")

    def _print_metrics(self, metrics: dict, steering: Optional[float]):
        """Print current metrics to console."""
        print(
            f"\rStatus: {metrics['departure_status'].value:<20} | "
            f"Offset: {metrics['lateral_offset_meters'] if metrics['lateral_offset_meters'] else 'N/A':>6s} m | "
            f"Heading: {metrics['heading_angle_deg'] if metrics['heading_angle_deg'] else 'N/A':>6s}° | "
            f"Steering: {steering if steering else 'N/A':>6}",
            end="",
        )

    def _get_departure_color(self, status: LaneDepartureStatus):
        """
        Get CARLA color based on departure status.

        Args:
            status: Departure status

        Returns:
            carla.Color object
        """
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

    def _respawn_vehicle(self) -> bool:
        """
        Handle vehicle respawn and camera reattachment.

        Returns:
            True if successful, False otherwise
        """
        # Store autopilot state before respawn
        autopilot_was_enabled = self.carla.is_autopilot_enabled()

        # Respawn vehicle
        if not self.carla.respawn_vehicle():
            return False

        # Wait a moment for vehicle to stabilize
        time.sleep(0.5)

        # Reattach camera
        if not self.carla.setup_camera(
            image_width=self.camera_config["width"],
            image_height=self.camera_config["height"],
            fov=self.camera_config["fov"],
            position=self.camera_config["position"],
            rotation=self.camera_config["rotation"],
        ):
            print("Warning: Failed to reattach camera after respawn")
            return False

        # Restore autopilot if it was enabled
        if autopilot_was_enabled:
            self.carla.set_vehicle_autopilot(True)

        # Reinitialize spectator overlay with new vehicle
        if self.spectator_overlay:
            self.spectator_overlay = SpectatorOverlay(self.carla.world)

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lane Keeping Assist System for CARLA")

    parser.add_argument(
        "--method",
        type=str,
        default="cv",
        choices=["cv", "dl"],
        help="Lane detection method (cv=Computer Vision, dl=Deep Learning)",
    )
    parser.add_argument("--width", type=int, default=800, help="Camera image width")
    parser.add_argument("--height", type=int, default=600, help="Camera image height")
    parser.add_argument(
        "--host", type=str, default="carla-server.local", help="CARLA server host"
    )
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--vehicle",
        type=str,
        default="vehicle.tesla.model3",
        help="Vehicle type to spawn",
    )
    parser.add_argument(
        "--spawn-point",
        type=int,
        default=None,
        help="Spawn point index (default: auto-select random free point)",
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

    # Create LKAS instance
    lkas = LaneKeepingAssist(
        detection_method=args.method,
        image_width=args.width,
        image_height=args.height,
        carla_host=args.host,
        carla_port=args.port,
        model_path=args.model,
    )

    # Setup CARLA
    if not lkas.setup_carla(vehicle_type=args.vehicle, spawn_point=args.spawn_point):
        print("Failed to setup CARLA. Exiting.")
        return 1

    # Run system
    lkas.run(display=not args.no_display, save_video=args.save_video)

    return 0


if __name__ == "__main__":
    exit(main())
