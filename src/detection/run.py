#!/usr/bin/env python3
"""
Standalone Lane Detection Server

This is a separate process that:
1. Loads a lane detection model (CV or DL)
2. Listens for image requests via Shared Memory
3. Processes images and returns lane detections
4. Ultra-low latency (~0.001ms) using shared memory IPC

Usage:
    # Start server with Computer Vision detector
    lane-detection --method cv

    # Start server with Deep Learning detector on GPU
    lane-detection --method dl --gpu 0

    # Custom shared memory names
    lane-detection --method cv --image-shm-name my_camera --detection-shm-name my_results
"""

import argparse
import sys
import signal
from pathlib import Path

from detection.core.config import ConfigManager
from detection import LaneDetection
from simulation.integration.shared_memory_detection import SharedMemoryDetectionServer
from simulation.integration.messages import ImageMessage, DetectionMessage


class DetectionService:
    """
    Standalone server that wraps detection module.
    Uses shared memory for ultra-low latency communication.
    """

    def __init__(self, config, detection_method: str,
                 image_shm_name: str = "camera_feed",
                 detection_shm_name: str = "detection_results"):
        """
        Initialize detection server.

        Args:
            config: System configuration
            detection_method: Detection method ('cv' or 'dl')
            image_shm_name: Shared memory name for images
            detection_shm_name: Shared memory name for detections
        """
        print("\n" + "=" * 60)
        print("Lane Detection Server")
        print("=" * 60)

        # Initialize detection module
        print(f"\nInitializing {detection_method.upper()} detector...")
        self.detection_module = LaneDetection(config, detection_method)
        print(f"✓ Detector ready: {self.detection_module.get_detector_name()}")
        print(f"  Parameters: {self.detection_module.get_detector_params()}")

        # Create shared memory server
        print()
        print(f"Using SHARED MEMORY mode (ultra-low latency)")
        print(f"  Image input: {image_shm_name}")
        print(f"  Detection output: {detection_shm_name}")
        self.server = SharedMemoryDetectionServer(
            image_shm_name=image_shm_name,
            detection_shm_name=detection_shm_name,
            image_shape=(config.camera.height, config.camera.width, 3)
        )

        print("\n" + "=" * 60)
        print("Server initialized successfully!")
        print("=" * 60)

    def process_image(self, image_msg: ImageMessage) -> DetectionMessage:
        """
        Process image request.

        This is the callback function called by the server for each request.

        Args:
            image_msg: Image message from client

        Returns:
            Detection message with lane results
        """
        # Use detection module to process image
        return self.detection_module.process_image(image_msg)

    def run(self):
        """Start serving detection requests."""

        # Register signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nReceived interrupt signal")
            self.server.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start serving
        self.server.serve(self.process_image)


def main():
    """Main entry point for detection server."""
    parser = argparse.ArgumentParser(description="Standalone Lane Detection Server")

    parser.add_argument(
        "--method",
        type=str,
        default="cv",
        choices=["cv", "dl"],
        help="Lane detection method (cv=Computer Vision, dl=Deep Learning)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: <project-root>/config.yaml)",
    )

    # Shared memory options
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

    # GPU option
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device ID (for DL method)"
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded")

    # Set GPU if specified
    if args.gpu is not None and args.method == "dl":
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"✓ Using GPU {args.gpu}")

    # Create and run server (shared memory only)
    server = DetectionService(
        config=config,
        detection_method=args.method,
        image_shm_name=args.image_shm_name,
        detection_shm_name=args.detection_shm_name
    )

    server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
