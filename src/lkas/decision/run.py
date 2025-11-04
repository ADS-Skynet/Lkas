#!/usr/bin/env python3
"""
Standalone Decision Server

This is a separate process that:
1. Receives lane detection results via Shared Memory
2. Computes control commands (steering, throttle, brake)
3. Sends control commands via Shared Memory
4. Ultra-low latency (~0.001ms) using shared memory IPC

Usage:
    # Start server with default configuration
    decision-server

    # Start server with custom config
    decision-server --config path/to/config.yaml

    # Custom shared memory names
    decision-server --detection-shm-name my_detections --control-shm-name my_controls

Architecture:
    Detection Process       Decision Process        Simulation/Vehicle Process
    ┌──────────────┐       ┌──────────────┐        ┌──────────────┐
    │ Detect Lanes │──────►│ Compute      │───────►│ Apply        │
    │              │ SHM   │ Controls     │ SHM    │ Control      │
    └──────────────┘       └──────────────┘        └──────────────┘
"""

import argparse
import sys
import signal
import time
from pathlib import Path

from lkas.detection.core.config import ConfigManager
from lkas.detection.integration.messages import DetectionMessage, ControlMessage
from lkas.detection.integration.shared_memory_detection import SharedMemoryDetectionChannel
from lkas.detection.integration.shared_memory_control import SharedMemoryControlChannel
from lkas.decision import DecisionController
from simulation.constants import CommunicationConstants


class DecisionServer:
    """
    Standalone server that runs decision module.
    Uses shared memory for ultra-low latency communication.
    Pairs with DecisionClient for clean client-server architecture.
    """

    def __init__(
        self,
        config,
        detection_shm_name: str = "detection_results",
        control_shm_name: str = "control_commands",
    ):
        """
        Initialize decision server.

        Args:
            config: System configuration
            detection_shm_name: Shared memory name for detection input
            control_shm_name: Shared memory name for control output
        """
        print("\n" + "=" * 60)
        print("Decision Server")
        print("=" * 60)

        # Initialize decision controller
        print(f"\nInitializing decision controller...")
        self.controller = DecisionController(
            image_width=config.camera.width,
            image_height=config.camera.height,
            kp=config.controller.kp,
            kd=config.controller.kd,
            throttle_policy={
                "base": config.throttle_policy.base,
                "min": config.throttle_policy.min,
                "steer_threshold": config.throttle_policy.steer_threshold,
                "steer_max": config.throttle_policy.steer_max,
            },
        )
        print(f"✓ Decision controller ready")
        print(f"  PD Gains: Kp={config.controller.kp}, Kd={config.controller.kd}")
        print(f"  Throttle: base={config.throttle_policy.base}, min={config.throttle_policy.min}")

        # Connect to detection shared memory (reader)
        print(f"\nConnecting to detection shared memory '{detection_shm_name}'...")
        self.detection_channel = SharedMemoryDetectionChannel(
            name=detection_shm_name,
            create=False,
            retry_count=20,
            retry_delay=0.5,
        )
        print(f"✓ Connected to detection input")

        # Create control shared memory (writer)
        print(f"Creating control shared memory '{control_shm_name}'...")
        self.control_channel = SharedMemoryControlChannel(
            name=control_shm_name,
            create=True,
            retry_count=20,
            retry_delay=0.5,
        )
        print(f"✓ Created control output")

        self.running = False
        self.frame_count = 0
        self.last_print_time = time.time()

        print("\n" + "=" * 60)
        print("Server initialized successfully!")
        print("=" * 60)


    def run(self):
        """Start serving decision requests."""

        # Register signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nReceived interrupt signal")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("\n" + "=" * 60)
        print("Decision Server Running")
        print("=" * 60)
        print(f"Reading detections from: detection_results")
        print(f"Writing controls to: control_commands")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        self.running = True

        try:
            while self.running:
                # Read detection from shared memory (non-blocking)
                detection = self.detection_channel.read()

                if detection is None:
                    time.sleep(0.0001)  # 0.1ms sleep
                    continue

                # Process detection and compute control
                start_time = time.time()
                control = self.controller.process_detection(detection)
                processing_time_ms = (time.time() - start_time) * 1000.0

                # Write control to shared memory using proper control channel
                self.control_channel.write(
                    control=control,
                    frame_id=detection.frame_id,
                    timestamp=detection.timestamp,
                    processing_time_ms=processing_time_ms
                )

                self.frame_count += 1

                # Stats tracking
                if time.time() - self.last_print_time > 3.0:
                    fps = self.frame_count / (time.time() - self.last_print_time)
                    print(
                        f"\r{fps:.1f} FPS | Frame {detection.frame_id} | "
                        f"Decision: {processing_time_ms:.2f}ms | "
                        f"Steering: {control.steering:+.3f} | "
                        f"Throttle: {control.throttle:.3f}",
                        end="",
                        flush=True,
                    )
                    self.frame_count = 0
                    self.last_print_time = time.time()

        except KeyboardInterrupt:
            print("\n\nStopping decision server...")
        finally:
            self.stop()

    def stop(self):
        """Stop the server and cleanup."""
        self.running = False
        self.detection_channel.close()
        self.control_channel.close()
        self.control_channel.unlink()
        print("✓ Decision server stopped")


def main():
    """Main entry point for decision server."""
    parser = argparse.ArgumentParser(description="Standalone Decision Server")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: <project-root>/config.yaml)",
    )

    # Shared memory options
    parser.add_argument(
        "--detection-shm-name",
        type=str,
        default=CommunicationConstants.DEFAULT_DETECTION_SHM_NAME,
        help=f"Shared memory name for detection input (default: {CommunicationConstants.DEFAULT_DETECTION_SHM_NAME})",
    )
    parser.add_argument(
        "--control-shm-name",
        type=str,
        default="control_commands",
        help="Shared memory name for control output (default: control_commands)",
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded")

    # Create and run server
    server = DecisionServer(
        config=config,
        detection_shm_name=args.detection_shm_name,
        control_shm_name=args.control_shm_name,
    )

    server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
