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
import os
import subprocess
import signal
import time
import select
import fcntl
import re
from pathlib import Path
from typing import Optional, List

from lkas.utils.terminal import TerminalDisplay, OrderedLogger


class LKASLauncher:
    """Launches and manages both detection and decision servers."""

    def __init__(
        self,
        method: str = "cv",
        config: Optional[str] = None,
        gpu: Optional[int] = None,
        image_shm_name: str = "camera_feed",
        detection_shm_name: str = "detection_results",
        control_shm_name: str = "control_commands",
        verbose: bool = False,
    ):
        """
        Initialize LKAS launcher.

        Args:
            method: Detection method (cv or dl)
            config: Path to configuration file
            gpu: GPU device ID (for DL method)
            image_shm_name: Shared memory name for camera images
            detection_shm_name: Shared memory name for detection results
            control_shm_name: Shared memory name for control commands
            verbose: Enable verbose output (FPS stats, latency info)
        """
        self.method = method
        self.config = config
        self.gpu = gpu
        self.image_shm_name = image_shm_name
        self.detection_shm_name = detection_shm_name
        self.control_shm_name = control_shm_name
        self.verbose = verbose

        self.detection_process: Optional[subprocess.Popen] = None
        self.decision_process: Optional[subprocess.Popen] = None
        self.running = False

        # Terminal display with persistent footer
        self.enable_footer = True
        self.terminal = TerminalDisplay(enable_footer=self.enable_footer)
        self.detection_logger = OrderedLogger("[SubProc]", self.terminal)
        self.decision_logger = OrderedLogger("[SubProc]", self.terminal)

        # Connection tracking
        self.detection_connected = False
        self.decision_connected = False

        # Initialization phase tracking
        self.buffering_mode = True

        # Progress line tracking (for suppressing intermediate updates)
        self.last_progress_line = {"DETECTION": None, "DECISION ": None}
        self.progress_counter = {"DETECTION": 0, "DECISION ": 0}
        self.last_was_progress = {"DETECTION": False, "DECISION ": False}

    def _build_detection_cmd(self) -> List[str]:
        """Build command for detection server."""
        cmd = [
            sys.executable,
            "-m",
            "lkas.detection.run",
            "--method",
            self.method,
            "--image-shm-name",
            self.image_shm_name,
            "--detection-shm-name",
            self.detection_shm_name,
            "--retry-count",
            "60",  # Wait up to 30 seconds for simulation
            "--retry-delay",
            "0.5",
        ]

        if self.config:
            cmd.extend(["--config", self.config])

        if self.gpu is not None and self.method == "dl":
            cmd.extend(["--gpu", str(self.gpu)])

        if not self.verbose:
            cmd.append("--no-stats")

        return cmd

    def _build_decision_cmd(self) -> List[str]:
        """Build command for decision server."""
        cmd = [
            sys.executable,
            "-m",
            "lkas.decision.run",
            "--detection-shm-name",
            self.detection_shm_name,
            "--control-shm-name",
            self.control_shm_name,
            "--retry-count",
            "60",  # Wait up to 30 seconds for detection server
            "--retry-delay",
            "0.5",
        ]

        if self.config:
            cmd.extend(["--config", self.config])

        if not self.verbose:
            cmd.append("--no-stats")

        return cmd

    def _print_header(self):
        """Print startup header."""
        self.terminal.print("\n" + "=" * 70)
        self.terminal.print(" " * 20 + "LKAS System Launcher")
        self.terminal.print("=" * 70)
        self.terminal.print(f"  Detection Method: {self.method.upper()}")
        if self.gpu is not None:
            self.terminal.print(f"  GPU Device: {self.gpu}")
        if self.config:
            self.terminal.print(f"  Config: {self.config}")
        self.terminal.print(f"  Image SHM: {self.image_shm_name}")
        self.terminal.print(f"  Detection SHM: {self.detection_shm_name}")
        self.terminal.print(f"  Control SHM: {self.control_shm_name}")
        self.terminal.print("=" * 70)
        self.terminal.print("")

    def _read_process_output(self, process: subprocess.Popen, prefix: str, logger: OrderedLogger):
        """
        Read and print process output with prefix.
        Non-blocking read using select.

        Args:
            process: Process to read from
            prefix: Prefix for messages
            logger: Logger to use for output
        """
        if process.stdout is None:
            return

        # Use select to check if there's data to read (Unix-like systems)
        if hasattr(select, 'select'):
            ready, _, _ = select.select([process.stdout], [], [], 0)
            if not ready:
                return

            try:
                # Read available data (non-blocking)
                # Stats lines end with \r, regular lines end with \n
                data = process.stdout.read(4096)  # Read up to 4KB
                if data:
                    lines_raw = data.decode('utf-8')
                    lines = lines_raw.splitlines(keepends=True)
                    # for line in lines_raw.splitlines(keepends=True):
                    #     print(repr(line))

                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            continue

                        # Mark as connected when we receive any output
                        if "DETECTION" in prefix and not self.detection_connected:
                            self._update_footer()
                        elif "DECISION" in prefix and not self.decision_connected:
                            self._update_footer()

                        # Print message (buffer or print depending on mode)
                        if self.buffering_mode:
                            logger.print_immediate(stripped, line.endswith('\r'))

                        # Log to file
                        if hasattr(self, 'log_file') and self.log_file:
                            self.log_file.write(f"[{prefix}] {stripped}\n")
            except BlockingIOError:
                # No data available - this is OK for non-blocking I/O
                pass
            except Exception:
                pass
        else:
            # Fallback for Windows
            try:
                # Try to read available data
                data = process.stdout.read(4096)
                if data:
                    lines_raw = data.decode('utf-8')
                    lines = lines_raw.splitlines(keepends=True)

                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            continue

                        # Mark as connected when we receive any output
                        if "DETECTION" in prefix and not self.detection_connected:
                            # self.detection_connected = True
                            self._update_footer()
                        elif "DECISION" in prefix and not self.decision_connected:
                            # self.decision_connected = True
                            self._update_footer()

                        # Print message (buffer or print depending on mode)
                        if self.buffering_mode:
                            logger.print_immediate(stripped, line.endswith('\r'))

                        if hasattr(self, 'log_file') and self.log_file:
                            self.log_file.write(f"[{prefix}] {stripped}\n")
            except Exception:
                pass

    def _update_footer(self):
        """Update the persistent footer with connection status from both processes."""
        if not self.enable_footer:
            return

        self.terminal.update_footer(
            detection_connected=self.detection_connected,
            decision_connected=self.decision_connected
        )

    def run(self):
        """Start both servers and manage their lifecycle."""
        self._print_header()

        # Open log file
        self.log_file = open("lkas_run.log", "w", buffering=1)

        # Initialize footer
        self.terminal.init_footer()

        # Register signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            self.terminal.clear_footer()
            self.terminal.print("\nReceived interrupt signal - shutting down...")
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.close()
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Start decision server FIRST (so it's ready when detection starts)
            self.terminal.print("Starting decision server...")
            decision_cmd = self._build_decision_cmd()
            # Set PYTHONUNBUFFERED to ensure output is not buffered
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            self.decision_process = subprocess.Popen(
                decision_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr to stdout
                bufsize=0,  # Unbuffered
                env=env,
            )
            # Make stdout non-blocking
            if self.decision_process.stdout:
                flags = fcntl.fcntl(self.decision_process.stdout, fcntl.F_GETFL)
                fcntl.fcntl(self.decision_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.terminal.print(f"✓ Decision server started (PID: {self.decision_process.pid})")

            self.detection_connected = True
            self._update_footer()

            # Read decision initialization messages (buffered for ordering)
            # Buffer for 3 seconds to capture the initial setup messages
            init_timeout = time.time() + 3.0
            while time.time() < init_timeout:
                self._read_process_output(self.decision_process, "DECISION ", self.decision_logger)
                time.sleep(0.01)

            # Flush decision initialization
            self.decision_logger.flush()

            # Check if decision server is still running
            if self.decision_process.poll() is not None:
                self.terminal.print("✗ Decision server failed to start!")
                self.stop()
                return 1

            # Small delay before starting detection server
            time.sleep(0.5)

            # Start detection server AFTER decision is ready
            self.terminal.print("\nStarting detection server...")
            detection_cmd = self._build_detection_cmd()
            # Set PYTHONUNBUFFERED to ensure output is not buffered
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            self.detection_process = subprocess.Popen(
                detection_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,  # Unbuffered
                env=env,
            )
            # Make stdout non-blocking
            if self.detection_process.stdout:
                flags = fcntl.fcntl(self.detection_process.stdout, fcntl.F_GETFL)
                fcntl.fcntl(self.detection_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.terminal.print(f"✓ Detection server started (PID: {self.detection_process.pid})")

            self.decision_connected = True
            self._update_footer()

            # Read detection initialization messages (buffered for ordering)
            # Buffer for 4 seconds to capture the initial setup messages
            init_timeout = time.time() + 4.0
            while time.time() < init_timeout:
                self._read_process_output(self.detection_process, "DETECTION", self.detection_logger)
                time.sleep(0.01)

            # Flush detection initialization
            self.detection_logger.flush()

            # Check if detection server is still running
            if self.detection_process.poll() is not None:
                self.terminal.print("✗ Detection server failed to start!")
                return 1

            # Exit buffering mode - from now on, print messages immediately
            self.buffering_mode = False

            self.terminal.print("\n" + "=" * 70)
            self.terminal.print("System running - Press Ctrl+C to stop")
            self.terminal.print("")
            self.terminal.print("Servers will wait up to 30 seconds for connections:")
            self.terminal.print("  - Detection server waiting for camera_feed from simulation")
            self.terminal.print("  - Decision server waiting for detection_results from detection")
            self.terminal.print("")
            self.terminal.print("Ready! Start 'simulation' in another terminal to begin processing")
            self.terminal.print("=" * 70 + "\n")

            # Initialize footer
            self._update_footer()

            self.running = True

            # Main loop - multiplex output from both processes
            while self.running:
                # Read output from both processes (use print_immediate for runtime)
                self._read_process_output(self.detection_process, "DETECTION", self.detection_logger)
                self._read_process_output(self.decision_process, "DECISION ", self.decision_logger)

                # Then check if processes are still alive
                detection_alive = self.detection_process.poll() is None
                decision_alive = self.decision_process.poll() is None

                if not detection_alive:
                    # Try to read remaining output
                    for _ in range(10):
                        self._read_process_output(self.detection_process, "DETECTION", self.detection_logger)
                    self.terminal.clear_footer()
                    self.terminal.print("✗ Detection server died unexpectedly!")
                    self.running = False
                    break

                if not decision_alive:
                    # Try to read remaining output before reporting death
                    self.terminal.clear_footer()
                    self.terminal.print("Decision server process ended, reading final output...")
                    time.sleep(0.1)  # Give output a moment to flush
                    for _ in range(20):
                        self._read_process_output(self.decision_process, "DECISION ", self.decision_logger)
                    self.terminal.print("✗ Decision server died unexpectedly!")
                    self.running = False
                    break

                # Small sleep to prevent busy-waiting
                time.sleep(0.01)

        except Exception as e:
            self.terminal.clear_footer()
            self.terminal.print(f"✗ Error: {e}")
            return 1
        finally:
            self.terminal.clear_footer()
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.close()
            self.stop()

        return 0

    def stop(self):
        """Stop both servers gracefully."""
        self.running = False

        self.terminal.print("\nStopping servers...")

        # Stop decision server first (consumer)
        if self.decision_process and self.decision_process.poll() is None:
            self.terminal.print("Stopping decision server...")
            self.decision_process.terminate()
            try:
                self.decision_process.wait(timeout=5.0)
                self.terminal.print("✓ Decision server stopped")
            except subprocess.TimeoutExpired:
                self.terminal.print("! Decision server not responding, killing...")
                self.decision_process.kill()
                self.decision_process.wait()

        # Then stop detection server (producer)
        if self.detection_process and self.detection_process.poll() is None:
            self.terminal.print("Stopping detection server...")
            self.detection_process.terminate()
            try:
                self.detection_process.wait(timeout=5.0)
                self.terminal.print("✓ Detection server stopped")
            except subprocess.TimeoutExpired:
                self.terminal.print("! Detection server not responding, killing...")
                self.detection_process.kill()
                self.detection_process.wait()

        self.terminal.print("✓ Cleanup complete")


def main():
    """Main entry point for LKAS launcher."""
    parser = argparse.ArgumentParser(
        description="LKAS System Launcher - Starts both Detection and Decision servers"
    )

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
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (for DL method)",
    )

    # Shared memory options (advanced)
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
    parser.add_argument(
        "--control-shm-name",
        type=str,
        default="control_commands",
        help="Shared memory name for control commands (default: control_commands)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (FPS stats, latency info)",
    )

    args = parser.parse_args()

    # Create and run launcher
    launcher = LKASLauncher(
        method=args.method,
        config=args.config,
        gpu=args.gpu,
        image_shm_name=args.image_shm_name,
        detection_shm_name=args.detection_shm_name,
        control_shm_name=args.control_shm_name,
        verbose=args.verbose,
    )

    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
