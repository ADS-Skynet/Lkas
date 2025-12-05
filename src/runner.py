#!/usr/bin/env python3
"""
LKAS Broker Runner for Real Vehicle

This script runs the LKAS broker which:
1. Reads frames and detections from shared memory
2. Broadcasts them to viewers via ZMQ
3. Receives vehicle state and forwards to viewers
4. Routes parameter updates and actions

Usage:
    python -m lkas.runner
    python -m lkas.runner --verbose
"""

import time
import signal
import argparse
import numpy as np
from pathlib import Path

from lkas.integration.zmq import create_broker_from_config, LKASBroker
from lkas.integration.shared_memory import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel,
)
from skynet_common.config import ConfigManager


class LKASBrokerRunner:
    """
    Runs the LKAS broker and broadcasts frames/detections from shared memory.

    This is the central coordinator for real vehicle operation.
    """

    def __init__(
        self,
        image_shm_name: str = "camera_feed",
        detection_shm_name: str = "detection_results",
        image_shape: tuple = (480, 640, 3),
        jpeg_quality: int = 85,
        verbose: bool = False,
    ):
        """
        Initialize LKAS broker runner.

        Args:
            image_shm_name: Shared memory name for camera images
            detection_shm_name: Shared memory name for detection results
            image_shape: Image shape (height, width, channels)
            jpeg_quality: JPEG encoding quality for broadcasting
            verbose: Enable verbose logging
        """
        print("\n" + "=" * 60)
        print("LKAS Broker Runner")
        print("=" * 60)

        self.image_shm_name = image_shm_name
        self.detection_shm_name = detection_shm_name
        self.image_shape = image_shape
        self.jpeg_quality = jpeg_quality
        self.verbose = verbose

        # Create LKAS broker
        print("Initializing LKAS Broker...")
        self.broker = create_broker_from_config(verbose=verbose)

        # Connect to shared memory (reader mode)
        print(f"Connecting to shared memory...")
        print(f"  Image: {image_shm_name} {image_shape}")
        print(f"  Detection: {detection_shm_name}")

        self.image_channel = SharedMemoryImageChannel(
            name=image_shm_name,
            shape=image_shape,
            create=False,  # Reader mode
            retry_count=30,
            retry_delay=1.0,
        )

        self.detection_channel = SharedMemoryDetectionChannel(
            name=detection_shm_name,
            create=False,  # Reader mode
            retry_count=30,
            retry_delay=1.0,
        )

        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.last_stats_time = time.time()

        self.running = False

        print("=" * 60)
        print("Broker runner ready")
        print("=" * 60 + "\n")

    def run(self):
        """
        Main loop: read from shared memory and broadcast to viewers.
        """
        self.running = True
        last_frame_id = -1
        last_detection_frame_id = -1

        print("[Runner] Starting main loop...")
        print("[Runner] Reading from shared memory and broadcasting to viewers")

        try:
            while self.running:
                # Poll broker for incoming messages (parameters, actions, vehicle status)
                self.broker.poll()

                # Read frame from shared memory and broadcast
                image_msg = self.image_channel.read(copy=False)
                if image_msg and image_msg.frame_id != last_frame_id:
                    print("sending frame", image_msg.frame_id)
                    # Broadcast frame to viewers
                    self.broker.broadcast_frame(
                        image_msg.image,
                        image_msg.frame_id,
                        self.jpeg_quality
                    )
                    last_frame_id = image_msg.frame_id
                    self.frame_count += 1

                # Read detection from shared memory and broadcast
                detection_msg = self.detection_channel.read()
                if detection_msg and detection_msg.frame_id != last_detection_frame_id:
                    # Convert to dict format expected by viewers
                    detection_data = {
                        'left_lane': None,
                        'right_lane': None,
                        'processing_time_ms': detection_msg.processing_time_ms,
                        'frame_id': detection_msg.frame_id
                    }
                    if detection_msg.left_lane:
                        detection_data['left_lane'] = {
                            'x1': detection_msg.left_lane.x1,
                            'y1': detection_msg.left_lane.y1,
                            'x2': detection_msg.left_lane.x2,
                            'y2': detection_msg.left_lane.y2,
                            'confidence': detection_msg.left_lane.confidence
                        }
                    if detection_msg.right_lane:
                        detection_data['right_lane'] = {
                            'x1': detection_msg.right_lane.x1,
                            'y1': detection_msg.right_lane.y1,
                            'x2': detection_msg.right_lane.x2,
                            'y2': detection_msg.right_lane.y2,
                            'confidence': detection_msg.right_lane.confidence
                        }

                    self.broker.broadcast_detection(detection_data, detection_msg.frame_id)
                    last_detection_frame_id = detection_msg.frame_id
                    self.detection_count += 1

                # Print stats periodically
                if time.time() - self.last_stats_time > 3.0:
                    elapsed = time.time() - self.last_stats_time
                    fps = self.frame_count / elapsed
                    det_fps = self.detection_count / elapsed
                    broker_stats = self.broker.get_stats()

                    if self.verbose:
                        print(f"\n[Runner] Stats:")
                        print(f"  Frames broadcast: {fps:.1f} FPS")
                        print(f"  Detections broadcast: {det_fps:.1f} FPS")
                        print(f"  Parameters forwarded: {broker_stats['parameters_forwarded']}")
                        print(f"  Actions received: {broker_stats['actions_received']}")
                        print(f"  Vehicle status received: {broker_stats['vehicle_status_received']}")
                    else:
                        print(f"\r[Runner] {fps:.1f} FPS | Det: {det_fps:.1f} FPS | Frame {last_frame_id}", end="", flush=True)

                    self.frame_count = 0
                    self.detection_count = 0
                    self.last_stats_time = time.time()

                # Small sleep to avoid busy-waiting
                time.sleep(0.001)  # 1ms

        except KeyboardInterrupt:
            print("\n[Runner] Interrupted by user")
        finally:
            self.close()

    def close(self):
        """Cleanup resources."""
        self.running = False
        print("\n[Runner] Shutting down...")

        # Close broker
        if self.broker:
            self.broker.close()

        # Close shared memory channels (readers just close, don't unlink)
        if self.image_channel:
            self.image_channel.close()
        if self.detection_channel:
            self.detection_channel.close()

        print("[Runner] Shutdown complete")


def main():
    """Main entry point."""
    # Load common config
    config = ConfigManager.load()
    comm = config.communication
    camera_cfg = config.camera

    parser = argparse.ArgumentParser(description="LKAS Broker Runner for Real Vehicle")
    parser.add_argument('--image-shm', default=comm.image_shm_name,
                       help=f"Shared memory name for images (default: {comm.image_shm_name})")
    parser.add_argument('--detection-shm', default=comm.detection_shm_name,
                       help=f"Shared memory name for detections (default: {comm.detection_shm_name})")
    parser.add_argument('--jpeg-quality', type=int, default=85,
                       help="JPEG encoding quality (default: 85)")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Image shape from common config
    image_shape = (camera_cfg.height, camera_cfg.width, 3)

    runner = LKASBrokerRunner(
        image_shm_name=args.image_shm,
        detection_shm_name=args.detection_shm,
        image_shape=image_shape,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
    )

    # Handle SIGTERM gracefully
    def signal_handler(signum, frame):
        print("\n[Runner] Received signal, shutting down...")
        runner.running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    runner.run()


if __name__ == "__main__":
    main()
