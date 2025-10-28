#!/usr/bin/env python3
"""
Distributed Lane Keeping System - CARLA Client (Enhanced Visualization)

NEW FEATURES:
- Multiple visualization backends (OpenCV, Pygame, Web)
- Auto-detection of best viewer for environment
- Web viewer for remote/Docker (no X11 needed!)
- Better support for XQuartz and remote development

Usage:
    # Auto-select best viewer
    python main.py --detector-url tcp://localhost:5555

    # Force specific viewer
    python main.py --viewer opencv --detector-url tcp://localhost:5555
    python main.py --viewer pygame --detector-url tcp://localhost:5555
    python main.py --viewer web --detector-url tcp://localhost:5555

    # Web viewer (best for remote/Docker!)
    python main.py --viewer web --web-port 8080 --detector-url tcp://localhost:5555
    # Then open: http://localhost:8080 in browser
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
from detection.core.config import ConfigManager
from simulation.integration.visualization import (
    VisualizationManager,
    auto_select_viewer,
)
from simulation import CARLAConnection, VehicleManager, CameraSensor
from decision import DecisionController
from simulation.integration.communication import DetectionClient
from simulation.integration.messages import (
    ImageMessage,
    DetectionMessage,
    ControlMessage,
    ControlMode,
)


def main():
    """Main entry point for distributed CARLA client with enhanced visualization."""
    parser = argparse.ArgumentParser(
        description="Distributed Lane Keeping System - Enhanced Visualization"
    )

    # System options
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--host", type=str, default=None, help="CARLA server host (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="CARLA server port (overrides config)"
    )
    parser.add_argument("--spawn-point", type=int, default=None)

    # Detection server
    parser.add_argument("--detector-url", type=str, default="tcp://localhost:5555")
    parser.add_argument("--detector-timeout", type=int, default=1000)

    # Visualization options ⭐ NEW
    parser.add_argument(
        "--viewer",
        type=str,
        choices=["auto", "opencv", "pygame", "web", "none"],
        default="auto",
        help="Visualization backend (auto=auto-detect best)",
    )
    parser.add_argument(
        "--web-port", type=int, default=8080, help="Port for web viewer"
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-autopilot", action="store_true")

    args = parser.parse_args()

    # Load configuration
    print("\nLoading configuration...")
    config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded from {args.config}")

    carla_host = args.host if args.host else config.carla.host
    carla_port = args.port if args.port else config.carla.port

    # Print banner
    print("\n" + "=" * 60)
    print("DISTRIBUTED LANE KEEPING SYSTEM - ENHANCED")
    print("=" * 60)
    print(f"CARLA Server: {carla_host}:{carla_port}")
    print(f"Detection Server: {args.detector_url}")
    print(f"Camera: {config.camera.width}x{config.camera.height}")

    # Determine viewer type
    if args.no_display:
        viewer_type = "none"
    elif args.viewer == "auto":
        viewer_type = auto_select_viewer()
        print(f"Auto-selected viewer: {viewer_type}")
    else:
        viewer_type = args.viewer

    print(f"Visualization: {viewer_type}")
    print("=" * 60)

    # Initialize visualization
    if not args.no_display:
        print(f"\nInitializing {viewer_type} viewer...")
        viz = VisualizationManager(
            viewer_type=viewer_type,
            width=config.camera.width,
            height=config.camera.height,
            web_port=args.web_port,
        )

    # Initialize CARLA
    print("\n[1/4] Connecting to CARLA...")
    carla_conn = CARLAConnection(carla_host, carla_port)
    if not carla_conn.connect():
        return 1

    print("\n[2/4] Spawning vehicle...")
    vehicle_mgr = VehicleManager(carla_conn.get_world())
    if not vehicle_mgr.spawn_vehicle(config.carla.vehicle_type, args.spawn_point):
        return 1

    print("\n[3/4] Setting up camera...")
    camera = CameraSensor(carla_conn.get_world(), vehicle_mgr.get_vehicle())
    if not camera.setup_camera(
        width=config.camera.width,
        height=config.camera.height,
        fov=config.camera.fov,
        position=config.camera.position,
        rotation=config.camera.rotation,
    ):
        return 1

    print("\n[4/4] Connecting to detection server...")
    try:
        detector = DetectionClient(args.detector_url, args.detector_timeout)
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return 1

    # Initialize decision controller
    controller = DecisionController(
        image_width=config.camera.width,
        image_height=config.camera.height,
        kp=config.controller.kp,
        kd=config.controller.kd,
    )

    # Enable autopilot
    if not args.no_autopilot:
        vehicle_mgr.set_autopilot(True)
        print("\n✓ Autopilot enabled")

    # Main loop
    print("\n" + "=" * 60)
    print("System Running")
    if viewer_type == "web":
        print(f"View at: http://localhost:{args.web_port}")
    print("Press Ctrl+C to quit")
    print("=" * 60 + "\n")

    frame_count = 0
    timeouts = 0
    last_print = time.time()

    try:
        while True:
            # Get image
            image = camera.get_latest_image()
            if image is None:
                continue

            # Send to detector
            image_msg = ImageMessage(
                image=image, timestamp=time.time(), frame_id=frame_count
            )
            detection = detector.detect(image_msg)

            if detection is None:
                timeouts += 1
                control = ControlMessage(
                    steering=0.0, throttle=0.0, brake=0.3, mode=ControlMode.LANE_KEEPING
                )
            else:
                control = controller.process_detection(detection)

            # Apply control
            if not vehicle_mgr.is_autopilot_enabled():
                vehicle_mgr.apply_control(
                    control.steering, control.throttle, control.brake
                )

            # Visualize
            if not args.no_display:
                # Use debug image from detector if available, otherwise use raw image
                if detection is not None and detection.debug_image is not None:
                    vis_image = detection.debug_image.copy()
                else:
                    vis_image = image.copy()

                    # Draw lanes manually if debug_image not available
                    if detection is not None:
                        if detection.left_lane:
                            cv2.line(
                                vis_image,
                                (detection.left_lane.x1, detection.left_lane.y1),
                                (detection.left_lane.x2, detection.left_lane.y2),
                                (255, 0, 0),
                                5,
                            )
                        if detection.right_lane:
                            cv2.line(
                                vis_image,
                                (detection.right_lane.x1, detection.right_lane.y1),
                                (detection.right_lane.x2, detection.right_lane.y2),
                                (0, 0, 255),
                                5,
                            )

                # Add text overlays
                cv2.putText(
                    vis_image,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis_image,
                    f"Steering: {control.steering:.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis_image,
                    f"Timeouts: {timeouts}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0) if timeouts > 0 else (0, 255, 0),
                    2,
                )

                # Show detection status
                status_text = (
                    "Detection: OK" if detection is not None else "Detection: TIMEOUT"
                )
                status_color = (0, 255, 0) if detection is not None else (255, 0, 0)
                cv2.putText(
                    vis_image,
                    status_text,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                )

                # Show image
                if not viz.show(vis_image):
                    print("\nViewer closed")
                    break

            frame_count += 1

            # Print status every 30 frames
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - last_print)
                lanes = (
                    "TIMEOUT"
                    if detection is None
                    else (
                        f"{'L' if detection.left_lane else '-'}{'R' if detection.right_lane else '-'}"
                    )
                )
                print(
                    f"Frame {frame_count:5d} | FPS: {fps:5.1f} | Lanes: {lanes} | "
                    f"Steering: {control.steering:+.3f} | Timeouts: {timeouts}"
                )
                last_print = time.time()

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # Cleanup
        if not args.no_display:
            viz.close()
        detector.close()
        camera.destroy_camera()
        vehicle_mgr.destroy_vehicle()
        carla_conn.disconnect()
        print("✓ Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
