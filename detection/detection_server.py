#!/usr/bin/env python3
"""
Standalone Lane Detection Server

This is a separate process that:
1. Loads a lane detection model (CV or DL)
2. Listens for image requests via ZMQ
3. Processes images and returns lane detections
4. Can run on a different machine with GPU

Usage:
    # Start server with Computer Vision detector
    python detection_server.py --method cv --port 5555

    # Start server with Deep Learning detector on GPU
    python detection_server.py --method dl --port 5555 --gpu 0

Benefits:
    - Runs independently from CARLA client
    - Can be on different machine (e.g., GPU server)
    - Can be restarted without affecting vehicle control
    - Multiple clients can connect to same server
"""

import argparse
import sys
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.core.config import ConfigManager
from detection.detection_module import LaneDetectionModule
from detection.integration.communication import DetectionServer
from detection.integration.messages import ImageMessage, DetectionMessage


class StandaloneDetectionServer:
    """
    Standalone detection server that wraps detection module.
    """

    def __init__(self, config, detection_method: str, bind_url: str):
        """
        Initialize standalone detection server.

        Args:
            config: System configuration
            detection_method: Detection method ('cv' or 'dl')
            bind_url: ZMQ URL to bind to
        """
        print("\n" + "=" * 60)
        print("Lane Detection Server")
        print("=" * 60)

        # Initialize detection module
        print(f"\nInitializing {detection_method.upper()} detector...")
        self.detection_module = LaneDetectionModule(config, detection_method)
        print(f"✓ Detector ready: {self.detection_module.get_detector_name()}")
        print(f"  Parameters: {self.detection_module.get_detector_params()}")

        # Create ZMQ server
        print()
        self.server = DetectionServer(bind_url)

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
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")
    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="Host to bind to (* for all interfaces, localhost for local only)",
    )
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

    # Create bind URL
    bind_url = f"tcp://{args.host}:{args.port}"

    # Create and run server
    server = StandaloneDetectionServer(config, args.method, bind_url)
    server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
