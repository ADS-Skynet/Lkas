#!/usr/bin/env python3
"""
Distributed Lane Keeping System - CARLA Client

Features:
- Multiple visualization backends (OpenCV, Pygame, Web)
- Auto-detection of best viewer for environment
- Web viewer for remote/Docker (no X11 needed!)
- Better support for XQuartz and remote development

Usage:
    simulation --viewer web

"""

import argparse
import sys
from pathlib import Path

import cv2
import time
from collections import deque
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
from simulation.integration.zmq_broadcast import (
    VehicleBroadcaster,
    ActionSubscriber,
    DetectionData,
    VehicleState,
)
from simulation.integration.shared_memory_detection import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionClient,
)


class LatencyStats:
    """Track and analyze latency at different pipeline stages."""

    def __init__(self, window_size: int = 100):
        """Initialize latency tracker.

        Args:
            window_size: Number of samples to keep for rolling statistics
        """
        self.window_size = window_size

        # Latency components (in milliseconds)
        self.capture_to_request = deque(
            maxlen=window_size
        )  # Camera capture to ZMQ send
        self.network_roundtrip = deque(maxlen=window_size)  # ZMQ request-reply time
        self.detection_processing = deque(
            maxlen=window_size
        )  # Detector's internal processing
        self.control_processing = deque(maxlen=window_size)  # Control computation time
        self.total_latency = deque(maxlen=window_size)  # End-to-end latency

        # Timestamps for current frame
        self.t_capture: float | None = None
        self.t_request_sent: float | None = None
        self.t_response_received: float | None = None
        self.t_control_applied: float | None = None

    def mark_capture(self):
        """Mark image capture timestamp."""
        self.t_capture = time.time()

    def mark_request_sent(self):
        """Mark detection request sent timestamp."""
        self.t_request_sent = time.time()

    def mark_response_received(self, detection_processing_ms: float):
        """Mark detection response received timestamp.

        Args:
            detection_processing_ms: Processing time reported by detector
        """
        self.t_response_received = time.time()

        # Record detection processing time (from detector)
        self.detection_processing.append(detection_processing_ms)

        # Calculate latencies
        if self.t_capture and self.t_request_sent:
            capture_to_req = (self.t_request_sent - self.t_capture) * 1000
            self.capture_to_request.append(capture_to_req)

        if self.t_request_sent and self.t_response_received:
            network = (self.t_response_received - self.t_request_sent) * 1000
            self.network_roundtrip.append(network)

    def mark_control_applied(self):
        """Mark control applied timestamp."""
        self.t_control_applied = time.time()

        # Calculate control processing time
        if self.t_response_received and self.t_control_applied:
            control = (self.t_control_applied - self.t_response_received) * 1000
            self.control_processing.append(control)

        # Calculate total end-to-end latency
        if self.t_capture and self.t_control_applied:
            total = (self.t_control_applied - self.t_capture) * 1000
            self.total_latency.append(total)

    def get_stats(self) -> dict:
        """Get statistics for all latency components.

        Returns:
            Dictionary with min/avg/max for each component
        """

        def stats(data):
            if not data:
                return {"min": 0.0, "avg": 0.0, "max": 0.0}
            return {
                "min": min(data),
                "avg": sum(data) / len(data),
                "max": max(data),
            }

        return {
            "capture_to_request": stats(self.capture_to_request),
            "network_roundtrip": stats(self.network_roundtrip),
            "detection_processing": stats(self.detection_processing),
            "control_processing": stats(self.control_processing),
            "total_latency": stats(self.total_latency),
            "sample_count": len(self.total_latency),
        }

    def print_report(self, frame_count: int):
        """Print comprehensive latency report.

        Args:
            frame_count: Current frame number
        """
        stats = self.get_stats()

        if stats["sample_count"] == 0:
            return

        print(f"\n{'='*70}")
        print(
            f"LATENCY REPORT - Frame {frame_count} (Last {stats['sample_count']} samples)"
        )
        print(f"{'='*70}")

        # Header
        print(f"{'Component':<25} {'Min (ms)':>12} {'Avg (ms)':>12} {'Max (ms)':>12}")
        print(f"{'-'*70}")

        # Each component
        components = [
            ("Capture → Request", "capture_to_request"),
            ("Network Round-trip", "network_roundtrip"),
            ("Detection Processing", "detection_processing"),
            ("Control Processing", "control_processing"),
            ("─" * 25, None),  # Separator
            ("TOTAL END-TO-END", "total_latency"),
        ]

        for label, key in components:
            if key is None:
                print(f"{label}")
                continue

            data = stats[key]
            print(
                f"{label:<25} {data['min']:>12.2f} {data['avg']:>12.2f} {data['max']:>12.2f}"
            )

        print(f"{'='*70}")

        # Breakdown percentages
        total_avg = stats["total_latency"]["avg"]
        if total_avg > 0:
            print(f"\nLatency Breakdown (% of total {total_avg:.2f}ms):")
            print(
                f"  Capture → Request:   {stats['capture_to_request']['avg']/total_avg*100:5.1f}%"
            )
            print(
                f"  Network Round-trip:  {stats['network_roundtrip']['avg']/total_avg*100:5.1f}%"
            )
            print(
                f"  Detection Processing: {stats['detection_processing']['avg']/total_avg*100:5.1f}%"
            )
            print(
                f"  Control Processing:  {stats['control_processing']['avg']/total_avg*100:5.1f}%"
            )

        # Bottleneck identification
        bottleneck = max(
            [
                ("Network Round-trip", stats["network_roundtrip"]["avg"]),
                ("Detection Processing", stats["detection_processing"]["avg"]),
                ("Control Processing", stats["control_processing"]["avg"]),
            ],
            key=lambda x: x[1],
        )
        print(f"\n⚠ Primary Bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f}ms)")
        print(f"{'='*70}\n")


def main():
    """Main entry point for distributed CARLA client with enhanced visualization."""
    parser = argparse.ArgumentParser(
        description="Distributed Lane Keeping System - Enhanced Visualization"
    )

    # System options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: <project-root>/config.yaml)",
    )
    parser.add_argument(
        "--host", type=str, default=None, help="CARLA server host (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="CARLA server port (overrides config)"
    )
    parser.add_argument("--spawn-point", type=int, default=None)

    # Detection communication mode
    parser.add_argument(
        "--shared-memory",
        action="store_true",
        help="Use shared memory for detection (ultra-low latency, same machine only)",
    )
    parser.add_argument(
        "--image-shm-name",
        type=str,
        default="camera_feed",
        help="Shared memory name for camera images (default: camera_feed)",
    )
    parser.add_argument(
        "--detection-shm-name",
        type=str,
        default="detection_results",
        help="Shared memory name for detection results (default: detection_results)",
    )

    # ZMQ detection server (used if not --shared-memory)
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
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Enable latency tracking and reporting (adds overhead)",
    )

    # ZMQ Broadcasting options (NEW: For remote viewer)
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="ZMQ broadcast mode: 'none' (disabled), 'detection-only' (production, ~9 KB/s), 'with-images' (development, ~1.5 MB/s)",
    )
    parser.add_argument(
        "--broadcast-url",
        type=str,
        default="tcp://*:5557",
        help="ZMQ URL for broadcasting vehicle data (default: tcp://*:5557)",
    )
    parser.add_argument(
        "--action-url",
        type=str,
        default="tcp://*:5558",
        help="ZMQ URL for receiving actions (default: tcp://*:5558)",
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

    # Detection mode
    if args.shared_memory:
        print(f"Detection Mode: SHARED MEMORY (ultra-low latency ~0.001ms)")
        print(f"  Image output: {args.image_shm_name}")
        print(f"  Detection input: {args.detection_shm_name}")
    else:
        print(f"Detection Mode: ZMQ (network-capable ~2ms)")
        print(f"  Detection Server: {args.detector_url}")

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

    # ZMQ Broadcasting (NEW: For production/remote viewer)
    if args.broadcast != "none":
        print(f"ZMQ Broadcasting: ENABLED")
        print(f"  Broadcast URL: {args.broadcast_url}")
        print(f"  Action URL: {args.action_url}")
        # if args.broadcast == "with-images":
        #     print(f"  Mode: DEVELOPMENT (with images, ~1.5 MB/s)")
        # else:
        #     print(f"  Mode: PRODUCTION (detection only, ~9 KB/s) ✅")
    else:
        print(f"ZMQ Broadcasting: DISABLED (use --broadcast detection-only or --broadcast with-images)")

    print("=" * 60)

    # Initialize ZMQ broadcaster (if enabled)
    broadcaster = None
    action_subscriber = None

    if args.broadcast:
        print("\nInitializing ZMQ broadcaster for remote viewer...")
        broadcaster = VehicleBroadcaster(bind_url=args.broadcast_url)
        action_subscriber = ActionSubscriber(bind_url=args.action_url)
        print("✓ ZMQ broadcaster ready")

    # Initialize visualization
    if not args.no_display:
        print(f"\nInitializing {viewer_type} viewer...")
        viz = VisualizationManager(
            viewer_type=viewer_type,
            width=config.camera.width,
            height=config.camera.height,
            web_port=args.web_port,
        )
    else:
        viz = None

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

    # World change (load specified town)
    # carla_conn.set_map("Town03")

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

    print("\n[5/5] Setting up detection communication...")
    detector = None
    image_writer = None

    if args.shared_memory:
        # Shared memory mode: Create image writer, detection reader
        # Both sides will retry if the other isn't ready yet
        try:
            print("Setting up shared memory communication...")
            print("(Both processes can start in any order - will retry if needed)")

            # Create image writer (creates shared memory)
            print("\nCreating shared memory image writer...")
            image_writer = SharedMemoryImageChannel(
                name=args.image_shm_name,
                shape=(config.camera.height, config.camera.width, 3),
                create=True,
                retry_count=20,
                retry_delay=0.5
            )

            # Connect to detection results (waits for detection server)
            print("\nConnecting to detection results...")
            detector = SharedMemoryDetectionClient(
                detection_shm_name=args.detection_shm_name,
                retry_count=20,
                retry_delay=0.5
            )

            print("\n✓ Shared memory communication ready")
        except Exception as e:
            print(f"\n✗ Failed to setup shared memory: {e}")
            print(f"  Tip: Make sure detection server is running with --shared-memory")
            print(f"       Both processes can start in any order, but both must be running.")
            return 1
    else:
        # ZMQ mode: Connect to detection server
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

    # Register action handlers (for both web viewer and ZMQ)
    paused_state = {'is_paused': False}

    # Define action handlers (shared by web viewer and ZMQ action subscriber)
    def handle_respawn():
        """Handle respawn action from any source."""
        print("\n🔄 Respawn requested")
        try:
            # if vehicle_mgr.respawn_vehicle():
            if vehicle_mgr.teleport_to_spawn_point(args.spawn_point):
                print("✓ Vehicle respawned successfully")
                return True
            else:
                print("✗ Failed to respawn vehicle")
                return False
        except Exception as e:
            print(f"✗ Respawn error: {e}")
            return False

    def handle_pause():
        """Handle pause action from any source."""
        paused_state['is_paused'] = True
        print("\n⏸ Paused - simulation loop will freeze")
        return True

    def handle_resume():
        """Handle resume action from any source."""
        paused_state['is_paused'] = False
        print("\n▶️ Resumed - simulation loop continues")
        return True

    # Register with ZMQ action subscriber (if broadcasting enabled)
    if action_subscriber:
        action_subscriber.register_action('respawn', handle_respawn)
        action_subscriber.register_action('pause', handle_pause)
        action_subscriber.register_action('resume', handle_resume)
        print("\n✓ ZMQ action subscriber registered")
        print("  Actions: respawn, pause, resume")

    # Register with web viewer (if using web viewer)
    if viz and viewer_type == 'web':
        if hasattr(viz, 'viewer') and hasattr(viz.viewer, 'register_action'):
            viz.viewer.register_action('respawn', handle_respawn)
            viz.viewer.register_action('pause', handle_pause)
            viz.viewer.register_action('resume', handle_resume)
            print("\n✓ Web viewer actions registered")
            print("  Press 'R' or click 'Respawn' button to respawn vehicle")
            print("  Press 'Space' or click 'Pause' button to pause/resume")
        else:
            print("\n⚠ Warning: Could not register web viewer actions")
            print(f"  viewer type: {type(viz.viewer) if hasattr(viz, 'viewer') else 'N/A'}")

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

    # Initialize latency tracker (optional)
    latency_tracker = LatencyStats(window_size=100) if args.latency else None
    if args.latency:
        print("📊 Latency tracking ENABLED (adds ~0.1ms overhead per frame)")
    else:
        print("📊 Latency tracking DISABLED (use --latency to enable)")

    # Print initialization strategy
    print(f"\n🚀 Initialization Strategy:")
    print(
        f"   Warmup: {args.warmup_frames} frames (~{args.warmup_frames/20:.1f} seconds)"
    )
    print(f"   During warmup:")
    print(f"     - Steering: LOCKED at 0.0 (go straight, ignore detections)")
    print(f"     - Throttle: Fixed at {args.base_throttle}")
    print(f"     - Reason: Early detections are unstable!")
    print(f"   After warmup:")
    print(f"     - Steering: From lane detection (PD controller)")
    print(
        f"     - Throttle: Adaptive ({config.throttle_policy.min}-{config.throttle_policy.base})"
    )
    print(f"     - Full lane-keeping control")

    try:
        while True:
            # Poll for ZMQ actions (non-blocking)
            if action_subscriber:
                action_subscriber.poll()

            # Check if paused
            if paused_state['is_paused']:
                time.sleep(0.1)  # Small delay to reduce CPU usage while paused
                continue

            # Tick the world (required in synchronous mode)
            if sync_mode:
                carla_conn.get_world().tick()

            # Get image
            image = camera.get_latest_image()
            if image is None:
                continue

            # [LATENCY TRACKING] Mark image capture timestamp
            if latency_tracker:
                latency_tracker.mark_capture()

            # Send to detector (different methods for shared memory vs ZMQ)
            image_msg = ImageMessage(
                image=image, timestamp=time.time(), frame_id=frame_count
            )

            # [LATENCY TRACKING] Mark request sent timestamp
            if latency_tracker:
                latency_tracker.mark_request_sent()

            if args.shared_memory:
                # Shared memory mode: Write image, read detection
                image_writer.write(image, timestamp=time.time(), frame_id=frame_count)
                detection = detector.get_detection(timeout=args.detector_timeout / 1000.0)
            else:
                # ZMQ mode: Send image, receive detection
                detection = detector.detect(image_msg)

            # [LATENCY TRACKING] Mark response received (with detection processing time)
            if latency_tracker:
                if detection is not None and hasattr(detection, "processing_time_ms"):
                    latency_tracker.mark_response_received(detection.processing_time_ms)
                elif detection is not None:
                    # Estimate if processing time not available
                    latency_tracker.mark_response_received(0.0)

            # Determine if we're still in warmup phase
            in_warmup = frame_count < args.warmup_frames

            # Check if we have valid detections (not None AND at least one lane exists)
            has_valid_detection = detection is not None and (
                detection.left_lane is not None or detection.right_lane is not None
            )

            if not has_valid_detection:
                timeouts += 1
                # No detection or empty detection: use base throttle to keep moving
                control = ControlMessage(
                    steering=0.0,
                    throttle=args.base_throttle,
                    brake=0.0,
                    mode=ControlMode.LANE_KEEPING,
                )
                if frame_count < 5:  # Print first few timeouts
                    if detection is None:
                        print(
                            f"⚠ Detection timeout on frame {frame_count} - using base throttle"
                        )
                    else:
                        print(
                            f"⚠ No lanes detected on frame {frame_count} - using base throttle"
                        )
            else:
                # Valid detection available, but might be unstable during warmup
                control = controller.process_detection(detection)

            # Apply control
            if not vehicle_mgr.is_autopilot_enabled():
                # Determine final steering and throttle
                if args.force_throttle is not None:
                    # Override for testing
                    steering = control.steering
                    throttle = args.force_throttle
                # elif in_warmup:
                #     # During warmup: IGNORE unstable steering, go straight with base throttle
                #     steering = 0.0  # GO STRAIGHT - don't trust early detections!
                #     throttle = args.base_throttle
                #     if frame_count == args.warmup_frames - 1:
                #         warmup_complete = True
                #         print(
                #             f"\n✅ Warmup complete! Detections stabilized. Switching to full lane-keeping control.\n"
                #         )
                else:
                    # After warmup: use full control (both steering and adaptive throttle)
                    steering = control.steering
                    throttle = control.throttle

                vehicle_mgr.apply_control(steering, throttle, control.brake)

                # [LATENCY TRACKING] Mark control applied timestamp
                if latency_tracker:
                    latency_tracker.mark_control_applied()

                # if frame_count < 5 or (
                #     in_warmup and frame_count % 10 == 0
                # ):  # Print during warmup
                #     mode = "WARMUP" if in_warmup else "ACTIVE"
                #     if in_warmup:
                #         print(
                #             f"[{mode}] Frame {frame_count:3d}: steering={steering:+.3f} (forced=0.0), "
                #             f"throttle={throttle:.3f}, detected_steering={control.steering:+.3f} (ignored)"
                #         )
                #     else:
                #         print(
                #             f"[{mode}] Frame {frame_count:3d}: steering={steering:+.3f}, throttle={throttle:.3f}, brake={control.brake:.3f}"
                #         )

            # Visualize
            if viz:
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
                # cv2.putText(
                #     vis_image,
                #     f"Frame: {frame_count}",
                #     (10, 30),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 255, 0),
                #     2,
                # )
                # cv2.putText(
                #     vis_image,
                #     f"Steering: {control.steering:+.3f} | Throttle: {control.throttle:.3f}",
                #     (10, 60),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (0, 255, 0),
                #     2,
                # )
                # cv2.putText(
                #     vis_image,
                #     f"Timeouts: {timeouts}",
                #     (10, 90),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (255, 0, 0) if timeouts > 0 else (0, 255, 0),
                #     2,
                # )

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

            # Broadcast to ZMQ viewers (if enabled)
            if broadcaster:
                # Send raw image ONLY if explicitly enabled (development mode, high bandwidth!)
                # Production mode: Only send detection data (tiny payload!)
                # if args.broadcast == "with-images":
                if args.broadcast:
                    broadcaster.send_frame(image, frame_count)  # ~50 KB per frame

                # Send detection results (ALWAYS - tiny payload ~200 bytes)
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
                        frame_id=frame_count
                    )
                    broadcaster.send_detection(detection_data)

                # Send vehicle state
                velocity = vehicle_mgr.get_velocity()
                speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 if velocity else 0.0
                vehicle_state = VehicleState(
                    steering=float(control.steering),
                    throttle=float(control.throttle),
                    brake=float(control.brake),
                    speed_kmh=float(speed_ms * 3.6),  # m/s to km/h
                    position=None,  # TODO: Get from vehicle_mgr if needed
                    rotation=None   # TODO: Get from vehicle_mgr if needed
                )
                broadcaster.send_state(vehicle_state)

            frame_count += 1

            # Print status every 30 frames
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - last_print)

                # Lane status
                if detection is None:
                    lanes = "TIMEOUT"
                else:
                    lanes = f"{'L' if detection.left_lane else '-'}{'R' if detection.right_lane else '-'}"

                # Detection info (add processing time if available)
                detection_info = ""
                if detection is not None and hasattr(detection, 'processing_time_ms'):
                    detection_info = f" | Det: {detection.processing_time_ms:.1f}ms"

                # Build status line with optional latency info
                status_line = (
                    f"Frame {frame_count:5d} | FPS: {fps:5.1f} | Lanes: {lanes}{detection_info} | "
                    f"Steering: {control.steering:+.3f} | Throttle: {control.throttle:.2f} | Timeouts: {timeouts}"
                )

                if latency_tracker:
                    stats = latency_tracker.get_stats()
                    total_latency = (
                        stats["total_latency"]["avg"]
                        if stats["sample_count"] > 0
                        else 0.0
                    )
                    status_line += f" | Latency: {total_latency:6.2f}ms"

                print(f"\r{status_line}", end="", flush=True)
                last_print = time.time()

            # Print detailed latency report every 90 frames (after warmup)
            if (
                latency_tracker
                and frame_count > args.warmup_frames
                and frame_count % 90 == 0
            ):
                latency_tracker.print_report(frame_count)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # Cleanup
        if viz:
            viz.close()
        if detector:
            detector.close()
        if image_writer:
            image_writer.close()
            image_writer.unlink()
        camera.destroy_camera()
        vehicle_mgr.destroy_vehicle()
        carla_conn.disconnect()
        print("✓ Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
