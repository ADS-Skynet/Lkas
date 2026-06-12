#!/usr/bin/env python3
"""
LKAS System Launcher

Starts both Detection and Decision servers as a unified system.
Combines debug logs from both processes into one terminal.

Usage:
    # Start LKAS with Computer Vision detector
    lkas --method cv

    # Start LKAS with Deep Learning detector on GPU
    lkas --method dl --gpu 0

    # Custom configuration
    lkas --method cv --config path/to/config.yaml
"""

import argparse
import sys

from .server import LKASServer


def main():
    """Main entry point for LKAS launcher."""
    parser = argparse.ArgumentParser(
        description="LKAS System Launcher - Starts both Detection and Decision servers"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="dl",
        choices=["cv", "dl"],
        help="Lane detection method (cv=Computer Vision, dl=Deep Learning)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: <project-root>/config.yaml)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (for DL detection method)",
    )
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="Enable ZMQ broadcasting for remote viewers (parameter updates, state, actions)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (FPS stats, latency info)",
    )

    # Decision controller override
    parser.add_argument(
        "--decision-method",
        type=str,
        default=None,
        help="Override decision controller method (e.g. 'planner', 'pid', 'pd', 'pure_pursuit')",
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help="Path to planner_model.pth (planner decision method only)",
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

    # Planner requires DL detection (needs the segmentation mask)
    if args.decision_method == "planner" and args.method != "dl":
        print("⚠ Warning: --decision-method planner works best with --method dl (needs segmentation mask)")

    # Create and run launcher
    launcher = LKASServer(
        method=args.method,
        config=args.config,
        gpu=args.gpu,
        broadcast=args.broadcast,
        verbose=args.verbose,
        decision_method=args.decision_method,
        planner_model=args.planner_model,
        scenario=args.scenario,
        planner_device=args.planner_device,
    )

    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
