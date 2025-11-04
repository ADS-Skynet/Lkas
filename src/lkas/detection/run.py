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
from pathlib import Path

from lkas.detection.core.config import ConfigManager
from lkas.detection.server import DetectionServer


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

    # Create and run server
    server = DetectionServer(
        config=config,
        detection_method=args.method,
        image_shm_name=args.image_shm_name,
        detection_shm_name=args.detection_shm_name
    )

    server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
