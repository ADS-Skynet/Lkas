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
    parser.add_argument("--detector-url", type=str, default="tcp://localhost:5556")
    parser.add_argument("--detector-timeout", type=int, default=1000)

    # Visualization options
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
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument(
        "--no-sync", action="store_true", help="Disable synchronous mode"
    )
    parser.add_argument(
        "--force-throttle",
        type=float,
        default=None,
        help="Force constant throttle (for testing)",
    )
    parser.add_argument(
        "--base-throttle",
        type=float,
        default=0.3,
        help="Base throttle during initialization/failures (default: 0.3)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=50,
        help="Frames to use base throttle before full control (default: 50)",
    )

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
    print("\n[1/5] Connecting to CARLA...")
    carla_conn = CARLAConnection(carla_host, carla_port)
    if not carla_conn.connect():
        return 1

    # Setup world (synchronous mode, cleanup, traffic lights)
    print("\n[2/5] Setting up world environment...")
    sync_mode = not args.no_sync
    if sync_mode:
        carla_conn.setup_synchronous_mode(enabled=True, fixed_delta_seconds=0.05)
    else:
        print("✓ Running in asynchronous mode (--no-sync)")

    # World cleanup (uncomment if needed)
    # carla_conn.cleanup_world()
    # carla_conn.set_all_traffic_lights_green()

    print("\n[3/5] Spawning vehicle...")
    vehicle_mgr = VehicleManager(carla_conn.get_world())
    if not vehicle_mgr.spawn_vehicle(config.carla.vehicle_type, args.spawn_point):
        return 1

    print("\n[4/5] Setting up camera...")
    camera = CameraSensor(carla_conn.get_world(), vehicle_mgr.get_vehicle())
    if not camera.setup_camera(
        width=config.camera.width,
        height=config.camera.height,
        fov=config.camera.fov,
        position=config.camera.position,
        rotation=config.camera.rotation,
    ):
        return 1

    print("\n[5/5] Connecting to detection server...")
    try:
        detector = DetectionClient(args.detector_url, args.detector_timeout)
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return 1

    # Initialize decision controller with adaptive throttle
    throttle_policy = {
        "base": config.throttle_policy.base,
        "min": config.throttle_policy.min,
        "steer_threshold": config.throttle_policy.steer_threshold,
        "steer_max": config.throttle_policy.steer_max,
    }

    controller = DecisionController(
        image_width=config.camera.width,
        image_height=config.camera.height,
        kp=config.controller.kp,
        kd=config.controller.kd,
        throttle_policy=throttle_policy,
    )
    print(
        f"✓ Adaptive throttle enabled: base={config.throttle_policy.base}, min={config.throttle_policy.min}"
    )

    # Enable autopilot
    if args.autopilot:
        vehicle_mgr.set_autopilot(True)
        print("\n✓ Autopilot enabled")

    # Main loop
    print("\n" + "=" * 60)
    print("System Running")
    # URL already printed by web_viewer.py, don't print again
    print("Press Ctrl+C to quit")
    print("=" * 60 + "\n")

    frame_count = 0
    timeouts = 0
    last_print = time.time()
    warmup_complete = False

    # Print initialization strategy
    print(f"\n🚀 Initialization Strategy:")
    print(f"   Warmup: {args.warmup_frames} frames (~{args.warmup_frames/20:.1f} seconds)")
    print(f"   During warmup:")
    print(f"     - Steering: LOCKED at 0.0 (go straight, ignore detections)")
    print(f"     - Throttle: Fixed at {args.base_throttle}")
    print(f"     - Reason: Early detections are unstable!")
    print(f"   After warmup:")
    print(f"     - Steering: From lane detection (PD controller)")
    print(f"     - Throttle: Adaptive ({config.throttle_policy.min}-{config.throttle_policy.base})")
    print(f"     - Full lane-keeping control")

    try:
        while True:
            # Tick the world (required in synchronous mode)
            if sync_mode:
                carla_conn.get_world().tick()

            # Get image
            image = camera.get_latest_image()
            if image is None:
                continue

            # Send to detector
            image_msg = ImageMessage(
                image=image, timestamp=time.time(), frame_id=frame_count
            )
            detection = detector.detect(image_msg)

            # Determine if we're still in warmup phase
            in_warmup = frame_count < args.warmup_frames

            if detection is None:
                timeouts += 1
                # During warmup or timeout: use base throttle to keep moving
                control = ControlMessage(
                    steering=0.0,
                    throttle=args.base_throttle,
                    brake=0.0,
                    mode=ControlMode.LANE_KEEPING,
                )
                if frame_count < 5:  # Print first few timeouts
                    print(
                        f"⚠ Detection timeout on frame {frame_count} - using base throttle"
                    )
            else:
                # Detection available, but might be unstable during warmup
                control = controller.process_detection(detection)

            # Apply control
            if not vehicle_mgr.is_autopilot_enabled():
                # Determine final steering and throttle
                if args.force_throttle is not None:
                    # Override for testing
                    steering = control.steering
                    throttle = args.force_throttle
                elif in_warmup:
                    # During warmup: IGNORE unstable steering, go straight with base throttle
                    steering = 0.0  # GO STRAIGHT - don't trust early detections!
                    throttle = args.base_throttle
                    if frame_count == args.warmup_frames - 1:
                        warmup_complete = True
                        print(
                            f"\n✅ Warmup complete! Detections stabilized. Switching to full lane-keeping control.\n"
                        )
                else:
                    # After warmup: use full control (both steering and adaptive throttle)
                    steering = control.steering
                    throttle = control.throttle

                vehicle_mgr.apply_control(steering, throttle, control.brake)
                if frame_count < 5 or (
                    in_warmup and frame_count % 10 == 0
                ):  # Print during warmup
                    mode = "WARMUP" if in_warmup else "ACTIVE"
                    if in_warmup:
                        print(
                            f"[{mode}] Frame {frame_count:3d}: steering={steering:+.3f} (forced=0.0), "
                            f"throttle={throttle:.3f}, detected_steering={control.steering:+.3f} (ignored)"
                        )
                    else:
                        print(
                            f"[{mode}] Frame {frame_count:3d}: steering={steering:+.3f}, throttle={throttle:.3f}, brake={control.brake:.3f}"
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
                    f"Steering: {control.steering:+.3f} | Throttle: {control.throttle:.3f}",
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
