"""
Main Entry Point for Modular Lane Keeping System

This is the NEW modular architecture with three separate modules:
1. CARLA Module: Handles simulator connection, vehicle, and sensors
2. Detection Module: Processes images and detects lanes
3. Decision Module: Analyzes lanes and generates control commands

The Orchestrator coordinates data flow between modules.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.core.config import ConfigManager
from detection.integration.orchestrator import SystemOrchestrator


def main():
    """Main entry point for the modular lane keeping system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Modular Lane Keeping Assist System for CARLA"
    )

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
    parser.add_argument(
        "--host", type=str, default=None, help="CARLA server host (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="CARLA server port (overrides config)"
    )
    parser.add_argument(
        "--spawn-point", type=int, default=None, help="Vehicle spawn point index"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable visualization display"
    )
    parser.add_argument(
        "--no-autopilot",
        action="store_true",
        help="Disable CARLA autopilot (use lane keeping control only)",
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded from {args.config}")

    # Use config values as defaults, override with command line args if provided
    carla_host = args.host if args.host else config.carla.host
    carla_port = args.port if args.port else config.carla.port

    print(f"  Detection method: {args.method}")
    print(f"  CARLA: {carla_host}:{carla_port}")
    print(f"  Camera: {config.camera.width}x{config.camera.height}")

    # Create orchestrator
    orchestrator = SystemOrchestrator(config, detection_method=args.method)

    # Initialize all modules
    if not orchestrator.initialize(
        carla_host=carla_host, carla_port=carla_port, spawn_point=args.spawn_point
    ):
        print("\n✗ Failed to initialize system")
        return 1

    # Run the system
    orchestrator.run(
        enable_autopilot=not args.no_autopilot, show_visualization=not args.no_display
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
