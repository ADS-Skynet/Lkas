#!/usr/bin/env python3
"""
Distributed Lane Keeping System - CARLA Client

Features:
- Shared memory IPC with detection server (ultra-low latency)
- ZMQ broadcasting for remote viewers
- Clean architecture with dependency injection

Usage:
    simulation --broadcast

Note: This has been refactored to use SimulationOrchestrator for clean architecture.
"""

import argparse
import sys

from lkas.detection.core.config import ConfigManager
from simulation.orchestrator import SimulationOrchestrator, SimulationConfig
from simulation.constants import CommunicationConstants, SimulationConstants


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Distributed Lane Keeping System - CARLA Client"
    )

    # System options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: <project-root>/config.yaml)",
    )
    parser.add_argument(
        "--host", type=str, default=None, help="CARLA server host (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="CARLA server port (overrides config)"
    )
    parser.add_argument("--spawn-point", type=int, default=None)

    # Shared memory detection (default and only IPC method)
    parser.add_argument(
        "--image-shm-name",
        type=str,
        default=CommunicationConstants.DEFAULT_IMAGE_SHM_NAME,
        help=f"Shared memory name for camera images (default: {CommunicationConstants.DEFAULT_IMAGE_SHM_NAME})",
    )
    parser.add_argument(
        "--detection-shm-name",
        type=str,
        default=CommunicationConstants.DEFAULT_DETECTION_SHM_NAME,
        help=f"Shared memory name for detection results (default: {CommunicationConstants.DEFAULT_DETECTION_SHM_NAME})",
    )
    parser.add_argument(
        "--detector-timeout",
        type=int,
        default=SimulationConstants.DEFAULT_DETECTOR_TIMEOUT_MS,
        help=f"Detection timeout in milliseconds (default: {SimulationConstants.DEFAULT_DETECTOR_TIMEOUT_MS})",
    )

    # Other options
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument(
        "--no-sync", action="store_true", help="Disable synchronous mode"
    )
    parser.add_argument(
        "--base-throttle",
        type=float,
        default=SimulationConstants.DEFAULT_BASE_THROTTLE,
        help=f"Base throttle during initialization/failures (default: {SimulationConstants.DEFAULT_BASE_THROTTLE})",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=SimulationConstants.WARMUP_FRAMES,
        help=f"Frames to use base throttle before full control (default: {SimulationConstants.WARMUP_FRAMES})",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Enable latency tracking and reporting (adds overhead)",
    )

    parser.add_argument(
        "--control-shm-name",
        type=str,
        default="control_commands",
        help="Shared memory name for control commands (default: control_commands)",
    )

    # ZMQ Broadcasting options (legacy - always enabled now)
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="(Legacy flag - broadcasting is now always enabled for viewer support)",
    )
    parser.add_argument(
        "--broadcast-url",
        type=str,
        default=f"tcp://*:{CommunicationConstants.DEFAULT_BROADCAST_PORT}",
        help=f"ZMQ URL for broadcasting vehicle data (default: tcp://*:{CommunicationConstants.DEFAULT_BROADCAST_PORT})",
    )
    parser.add_argument(
        "--action-url",
        type=str,
        default=f"tcp://*:{CommunicationConstants.DEFAULT_ACTION_PORT}",
        help=f"ZMQ URL for receiving actions (default: tcp://*:{CommunicationConstants.DEFAULT_ACTION_PORT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (lane status, steering info)",
    )

    return parser.parse_args()


def print_banner(config: SimulationConfig, system_config: object):
    """
    Print startup banner with configuration.

    Args:
        config: Simulation configuration
        system_config: System-wide configuration
    """
    print("\n" + "=" * 60)
    print("DISTRIBUTED LANE KEEPING SYSTEM")
    print("=" * 60)
    print(f"CARLA Server: {config.carla_host}:{config.carla_port}")

    print(f"SHARED MEMORY TABLE")
    print(f"  Image: {config.image_shm_name}")
    print(f"  Detection: {config.detection_shm_name}")
    print(f"  Control: {config.control_shm_name}")
    print(f"  Timeout: {config.detector_timeout}ms")

    print(f"Camera: {system_config.camera.width}x{system_config.camera.height}")

    # ZMQ Broadcasting
    if config.enable_broadcast:
        print(f"ZMQ Broadcasting: ENABLED")
        print(f"  Broadcast URL: {config.broadcast_url}")
        print(f"  Action URL: {config.action_url}")
    else:
        print(f"ZMQ Broadcasting: DISABLED (use --broadcast to enable)")

    print("=" * 60)


def main():
    """Main entry point for distributed CARLA client with ZMQ broadcasting."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    print("\nLoading configuration...")
    system_config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded")

    # Determine CARLA connection params
    carla_host = args.host if args.host else system_config.carla.host
    carla_port = args.port if args.port else system_config.carla.port

    # Create simulation configuration
    sim_config = SimulationConfig(
        carla_host=carla_host,
        carla_port=carla_port,
        spawn_point=args.spawn_point,
        image_shm_name=args.image_shm_name,
        detection_shm_name=args.detection_shm_name,
        detector_timeout=args.detector_timeout,
        enable_broadcast=args.broadcast,
        broadcast_url=args.broadcast_url,
        action_url=args.action_url,
        enable_autopilot=args.autopilot,
        enable_sync_mode=not args.no_sync,
        base_throttle=args.base_throttle,
        warmup_frames=args.warmup_frames,
        enable_latency_tracking=args.latency,
        control_shm_name=args.control_shm_name,
        verbose=args.verbose,
    )

    # Print banner
    print_banner(sim_config, system_config)

    # Create orchestrator
    orchestrator = SimulationOrchestrator(sim_config, system_config)

    # Setup all subsystems
    if not orchestrator.setup():
        print("\n✗ Setup failed")
        return 1

    # Run main loop
    orchestrator.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
