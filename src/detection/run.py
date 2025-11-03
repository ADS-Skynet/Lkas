#!/usr/bin/env python3
"""
Standalone Lane Detection Server

This is a separate process that:
1. Loads a lane detection model (CV or DL)
2. Listens for image requests via ZMQ OR Shared Memory
3. Processes images and returns lane detections
4. Can run on a different machine with GPU (ZMQ mode) or same machine (Shared Memory mode)

Communication Modes:
    - ZMQ (default): Network-capable, ~2ms latency, can run on different machines
    - Shared Memory: Ultra-fast ~0.001ms latency, same machine only

Usage:
    # Start server with Computer Vision detector (ZMQ mode)
    lane-detection --method cv --port 5556

    # Start server with Deep Learning detector on GPU (ZMQ mode)
    lane-detection --method dl --port 5555 --gpu 0

    # Start server with Shared Memory mode (ultra-low latency)
    lane-detection --method cv --shared-memory
"""

import argparse
import sys
import signal
from pathlib import Path

from detection.core.config import ConfigManager
from detection import LaneDetection
from simulation.integration.communication import DetectionServer as ZmqDetectionServer
from simulation.integration.shared_memory_detection import SharedMemoryDetectionServer
from simulation.integration.messages import ImageMessage, DetectionMessage


class DetectionService:
    """
    Standalone server that wraps detection module.
    Supports both ZMQ and Shared Memory communication modes.
    """

    def __init__(self, config, detection_method: str,
                 use_shared_memory: bool = False,
                 bind_url: str = None,
                 image_shm_name: str = "camera_feed",
                 detection_shm_name: str = "detection_results"):
        """
        Initialize detection server.

        Args:
            config: System configuration
            detection_method: Detection method ('cv' or 'dl')
            use_shared_memory: Use shared memory instead of ZMQ
            bind_url: ZMQ URL to bind to (if not using shared memory)
            image_shm_name: Shared memory name for images (if using shared memory)
            detection_shm_name: Shared memory name for detections (if using shared memory)
        """
        print("\n" + "=" * 60)
        print("Lane Detection Server")
        print("=" * 60)

        self.use_shared_memory = use_shared_memory

        # Initialize detection module
        print(f"\nInitializing {detection_method.upper()} detector...")
        self.detection_module = LaneDetection(config, detection_method)
        print(f"✓ Detector ready: {self.detection_module.get_detector_name()}")
        print(f"  Parameters: {self.detection_module.get_detector_params()}")

        # Create server based on mode
        print()
        if use_shared_memory:
            print(f"Using SHARED MEMORY mode (ultra-low latency ~0.001ms)")
            print(f"  Image input: {image_shm_name}")
            print(f"  Detection output: {detection_shm_name}")
            self.server = SharedMemoryDetectionServer(
                image_shm_name=image_shm_name,
                detection_shm_name=detection_shm_name,
                image_shape=(config.camera.height, config.camera.width, 3)
            )
        else:
            print(f"Using ZMQ mode (network-capable ~2ms latency)")
            print(f"  Bind URL: {bind_url}")
            self.server = ZmqDetectionServer(bind_url)

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

    # Communication mode
    parser.add_argument(
        "--shared-memory",
        action="store_true",
        help="Use shared memory instead of ZMQ (ultra-low latency, same machine only)",
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

    # ZMQ mode options
    parser.add_argument("--port", type=int, default=5556, help="Port to listen on (ZMQ mode)")
    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="Host to bind to (* for all interfaces, localhost for local only) (ZMQ mode)",
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

    # Create and run server
    if args.shared_memory:
        # Shared memory mode
        server = DetectionService(
            config=config,
            detection_method=args.method,
            use_shared_memory=True,
            image_shm_name=args.image_shm_name,
            detection_shm_name=args.detection_shm_name
        )
    else:
        # ZMQ mode
        bind_url = f"tcp://{args.host}:{args.port}"
        server = DetectionService(
            config=config,
            detection_method=args.method,
            use_shared_memory=False,
            bind_url=bind_url
        )

    server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
