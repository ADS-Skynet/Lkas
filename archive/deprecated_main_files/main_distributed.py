#!/usr/bin/env python3
"""
Distributed Lane Keeping System - CARLA Client

This is the PRODUCTION-READY architecture with separate processes:
- This process: CARLA client with vehicle control and decision making
- Remote process: Detection server (can be on different machine with GPU)

Benefits:
✅ Detection server can run on GPU machine
✅ Can restart detection without affecting vehicle
✅ Multiple vehicles can share one detection server
✅ Ready for ML model deployment
✅ Cloud-ready architecture

Usage:
    # Terminal 1: Start detection server
    python detection_server.py --method cv --port 5555

    # Terminal 2: Start CARLA client
    python main_distributed.py --detector-url tcp://localhost:5555

    # For remote GPU server:
    python main_distributed.py --detector-url tcp://192.168.1.100:5555
"""

import argparse
import sys
from core.config import ConfigManager
from integration.distributed_orchestrator import DistributedOrchestrator


def main():
    """Main entry point for distributed CARLA client."""
    parser = argparse.ArgumentParser(
        description="Distributed Lane Keeping System - CARLA Client"
    )

    # System options
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    # CARLA connection options (override config if provided)
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="CARLA server host (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="CARLA server port (overrides config)"
    )
    parser.add_argument(
        "--spawn-point",
        type=int,
        default=None,
        help="Vehicle spawn point index"
    )

    # Detection server options
    parser.add_argument(
        "--detector-url",
        type=str,
        default="tcp://localhost:5555",
        help="URL of detection server (e.g., tcp://localhost:5555 or tcp://192.168.1.100:5555)"
    )
    parser.add_argument(
        "--detector-timeout",
        type=int,
        default=1000,
        help="Detection request timeout in milliseconds"
    )

    # Display options
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable visualization display"
    )
    parser.add_argument(
        "--no-autopilot",
        action="store_true",
        help="Disable CARLA autopilot (use lane keeping control only)"
    )

    args = parser.parse_args()

    # Load configuration FIRST
    print("\nLoading configuration...")
    config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded from {args.config}")

    # Use config values as defaults, override with command line args if provided
    carla_host = args.host if args.host else config.carla.host
    carla_port = args.port if args.port else config.carla.port

    # Print banner
    print("\n" + "="*60)
    print("DISTRIBUTED LANE KEEPING SYSTEM - CARLA CLIENT")
    print("="*60)
    print(f"Architecture: Multi-Process (Production Ready)")
    print(f"CARLA Server: {carla_host}:{carla_port}")
    print(f"Detection Server: {args.detector_url}")
    print("="*60)
    print(f"  Camera: {config.camera.width}x{config.camera.height}")
    print(f"  Controller: Kp={config.controller.kp}, Kd={config.controller.kd}")

    # Check if detection server info is provided
    if args.detector_url == "tcp://localhost:5555":
        print("\n⚠  Using default detection server URL: tcp://localhost:5555")
        print("   Make sure detection server is running:")
        print("   python detection_server.py --port 5555")
        print()

    # Create orchestrator
    orchestrator = DistributedOrchestrator(
        config,
        detector_url=args.detector_url
    )

    # Initialize all components
    if not orchestrator.initialize(
        carla_host=carla_host,
        carla_port=carla_port,
        spawn_point=args.spawn_point,
        detector_timeout_ms=args.detector_timeout
    ):
        print("\n✗ Failed to initialize system")
        print("\nTroubleshooting:")
        print("1. Make sure CARLA is running: ./CarlaUE4.sh")
        print("2. Make sure detection server is running:")
        print(f"   python detection_server.py --port {args.detector_url.split(':')[-1]}")
        return 1

    # Run the system
    orchestrator.run(
        enable_autopilot=not args.no_autopilot,
        show_visualization=not args.no_display
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
