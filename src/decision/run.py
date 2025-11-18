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

from skynet_common.config import ConfigManager
from lkas.integration.messages import DetectionMessage, ControlMessage
from lkas.integration.shared_memory import SharedMemoryDetectionChannel, SharedMemoryControlChannel
from lkas.decision import DecisionController
# from simulation.constants import CommunicationConstants


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
        retry_count: int = 20,
        retry_delay: float = 0.5,
        enable_parameter_updates: bool = True,
        parameter_broker_url: str = "tcp://localhost:5560",
    ):
        """
        Initialize decision server.

        Args:
            config: System configuration
            detection_shm_name: Shared memory name for detection input
            control_shm_name: Shared memory name for control output
            retry_count: Connection retry attempts
            retry_delay: Delay between retries (seconds)
            enable_parameter_updates: Enable real-time parameter updates via ZMQ
            parameter_broker_url: ZMQ URL for parameter broker
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
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
        print(f"✓ Connected to detection input")

        # Create control shared memory (writer)
        print(f"Creating control shared memory '{control_shm_name}'...")
        self.control_channel = SharedMemoryControlChannel(
            name=control_shm_name,
            create=True,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
        print(f"✓ Created control output")

        self.running = False
        self.frame_count = 0
        self.last_print_time = time.time()

        # Setup parameter updates if enabled
        self.param_client = None
        if enable_parameter_updates:
            from lkas.integration.zmq import ParameterClient

            print(f"\nSetting up real-time parameter updates...")
            self.param_client = ParameterClient(
                category='decision',
                broker_url=parameter_broker_url
            )
            self.param_client.register_callback(self._on_parameter_update)
            print(f"✓ Parameter updates enabled")

        print("\n" + "=" * 60)
        print("Server initialized successfully!")
        print("=" * 60)

    def _on_parameter_update(self, param_name: str, value: float):
        """
        Handle real-time parameter update.

        Args:
            param_name: Parameter name
            value: New value
        """
        success = self.controller.update_parameter(param_name, value)
        if success:
            print(f"[Decision] Parameter updated: {param_name} = {value}")
        else:
            print(f"[Decision] Failed to update parameter: {param_name}")

    def run(self, print_stats: bool = True):
        """Start serving decision requests.

        Args:
            print_stats: Whether to print FPS and latency statistics
        """

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
                try:
                    # Poll for parameter updates (non-blocking)
                    if self.param_client:
                        self.param_client.poll()

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

                    # Stats tracking (only if enabled)
                    if print_stats and time.time() - self.last_print_time > 3.0:
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

                except Exception as e:
                    print(f"\n✗ Error in decision loop: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

        except KeyboardInterrupt:
            print("\n\nStopping decision server...")
        finally:
            self.stop()

    def stop(self):
        """Stop the server and cleanup."""
        self.running = False

        # Close parameter client
        if self.param_client:
            self.param_client.close()

        self.detection_channel.close()
        self.control_channel.close()
        self.control_channel.unlink()
        print("✓ Decision server stopped")


def main():
    """Main entry point for decision server."""
    # Load common config for defaults
    common_config = ConfigManager.load()
    comm = common_config.communication

    parser = argparse.ArgumentParser(description="Standalone Decision Server")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: <project-root>/config.yaml)",
    )

    # Connection retry options
    parser.add_argument(
        "--retry-count",
        type=int,
        default=20,
        help="Number of retry attempts for shared memory connection (default: 20)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.5,
        help="Delay between retry attempts in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable FPS and latency statistics output",
    )

    args = parser.parse_args()

    # Load configuration
    # print("Loading configuration...")
    config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded")

    # Create and run server
    server = DecisionServer(
        config=config,
        detection_shm_name=comm.detection_shm_name,
        control_shm_name=comm.control_shm_name,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay,
    )

    server.run(print_stats=not args.no_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
