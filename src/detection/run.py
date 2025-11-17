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

from skynet_common.config import ConfigManager
from lkas.detection.server import DetectionServer


def main():
    """Main entry point for detection server."""
    # Load common config for defaults
    common_config = ConfigManager.load()
    comm = common_config.communication

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
        default=comm.image_shm_name,
        help=f"Shared memory name for camera images (default: {comm.image_shm_name})",
    )
    parser.add_argument(
        "--detection-shm-name",
        type=str,
        default=comm.detection_shm_name,
        help=f"Shared memory name for detection results (default: {comm.detection_shm_name})",
    )

    # GPU option
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device ID (for DL method)"
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
        detection_shm_name=args.detection_shm_name,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay,
    )

    server.run(print_stats=not args.no_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
