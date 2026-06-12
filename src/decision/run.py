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

from common.config import ConfigManager
from lkas.integration.shared_memory.messages import DetectionMessage, ControlMessage
from lkas.integration.shared_memory import SharedMemoryDetectionChannel, SharedMemoryControlChannel
from lkas.decision import DecisionController
from lkas.decision.server import DecisionServer


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

    # Shared memory names
    parser.add_argument(
        "--detection-shm-name",
        type=str,
        default=None,
        help="Shared memory name for detection input (default: from config)",
    )
    parser.add_argument(
        "--control-shm-name",
        type=str,
        default=None,
        help="Shared memory name for control output (default: from config)",
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

    # Decision controller method (overrides config.controller.method)
    parser.add_argument(
        "--controller-method",
        type=str,
        default=None,
        help="Override controller method from config (e.g. 'planner', 'pid', 'pd', 'pure_pursuit')",
    )

    # Planner-specific arguments
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help="Path to planner_model.pth (planner method only; defaults to planner-e2e/planner_model.pth)",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
        help="Scenario token for planner (0=LANE_FOLLOW, default: 0)",
    )
    parser.add_argument(
        "--planner-device",
        type=str,
        default="cpu",
        help="Torch device for planner inference (default: cpu)",
    )

    args = parser.parse_args()

    # Load configuration
    config = ConfigManager.load(args.config)
    print(f"✓ Configuration loaded")

    # Use command-line args or fall back to config
    detection_shm_name = args.detection_shm_name or config.communication.detection_shm_name
    control_shm_name = args.control_shm_name or config.communication.control_shm_name

    # Create and run server
    server = DecisionServer(
        config=config,
        detection_shm_name=detection_shm_name,
        control_shm_name=control_shm_name,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay,
        controller_method_override=args.controller_method,
        model_path=args.planner_model,
        scenario=args.scenario,
        planner_device=args.planner_device,
    )

    server.run(print_stats=not args.no_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
