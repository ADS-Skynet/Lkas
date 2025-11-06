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
        """
        self.method = method
        self.config = config
        self.gpu = gpu
        self.image_shm_name = image_shm_name
        self.detection_shm_name = detection_shm_name
        self.control_shm_name = control_shm_name

        self.detection_process: Optional[subprocess.Popen] = None
        self.decision_process: Optional[subprocess.Popen] = None
        self.running = False

        # Terminal display with persistent footer
        self.enable_footer = True
        self.terminal = TerminalDisplay(enable_footer=self.enable_footer)
        self.detection_logger = OrderedLogger("[Detector]", self.terminal)
        self.decision_logger = OrderedLogger("[Controller]", self.terminal)

        # Stats tracking
        self.last_detection_stats = ""
        self.last_decision_stats = ""

        # Initialization phase tracking
        self.buffering_mode = True

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
                line = process.stdout.readline()
                if line:
                    line = line.decode('utf-8').rstrip()
                    if line:  # Only print non-empty lines
                        # Check if this is a stats line (starts with \r or contains FPS)
                        # Stats lines format: "\r45.2 FPS | Frame 1234 | ..."
                        is_stats = line.startswith('\r') or ('FPS' in line and '|' in line)

                        if is_stats:
                            # Update stats tracking - strip \r and whitespace
                            stats_text = line.lstrip('\r').strip()
                            if prefix == "DETECTION":
                                self.last_detection_stats = stats_text
                            else:
                                self.last_decision_stats = stats_text
                            self._update_footer()
                        else:
                            # Regular message - buffer or print depending on mode
                            if self.buffering_mode:
                                logger.log(line)
                            else:
                                logger.print_immediate(line)

                        # Log to file
                        if hasattr(self, 'log_file') and self.log_file:
                            self.log_file.write(f"[{prefix}] {line}\n")
            except BlockingIOError:
                # No data available - this is OK for non-blocking I/O
                pass
            except Exception:
                pass
        else:
            # Fallback for Windows (less efficient)
            try:
                line = process.stdout.readline()
                if line:
                    line = line.decode('utf-8').rstrip()
                    if line:
                        is_stats = line.startswith('\r') or ('FPS' in line and '|' in line)

                        if is_stats:
                            stats_text = line.lstrip('\r').strip()
                            if prefix == "DETECTION":
                                self.last_detection_stats = stats_text
                            else:
                                self.last_decision_stats = stats_text
                            self._update_footer()
                        else:
                            # Regular message - buffer or print depending on mode
                            if self.buffering_mode:
                                logger.log(line)
                            else:
                                logger.print_immediate(line)

                        if hasattr(self, 'log_file') and self.log_file:
                            self.log_file.write(f"[{prefix}] {line}\n")
            except Exception:
                pass

    def _update_footer(self):
        """Update the persistent footer with latest stats from both processes."""
        if not self.enable_footer:
            return

        parts = []

        if self.last_detection_stats:
            parts.append(f"[Detector] {self.last_detection_stats}")

        if self.last_decision_stats:
            parts.append(f"[Controller] {self.last_decision_stats}")

        if parts:
            footer_text = " | ".join(parts)
            self.terminal.update_footer(footer_text)
        else:
            # If no stats yet, show waiting message
            self.terminal.update_footer("Waiting for stats...")

    def run(self):
        """Start both servers and manage their lifecycle."""
        self._print_header()

        # Open log file
        self.log_file = open("lkas_run.log", "w", buffering=1)
        self.terminal.print("Logging output to lkas_run.log")

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
            # Start detection server
            self.terminal.print("Starting detection server...")
            detection_cmd = self._build_detection_cmd()
            self.detection_process = subprocess.Popen(
                detection_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,  # Unbuffered
            )
            # Make stdout non-blocking
            if self.detection_process.stdout:
                flags = fcntl.fcntl(self.detection_process.stdout, fcntl.F_GETFL)
                fcntl.fcntl(self.detection_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.terminal.print(f"✓ Detection server started (PID: {self.detection_process.pid})")

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

            # Small delay before starting decision server
            time.sleep(0.5)

            # Start decision server
            self.terminal.print("Starting decision server...")
            decision_cmd = self._build_decision_cmd()
            self.decision_process = subprocess.Popen(
                decision_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr to stdout
                bufsize=0,  # Unbuffered
            )
            # Make stdout non-blocking
            if self.decision_process.stdout:
                flags = fcntl.fcntl(self.decision_process.stdout, fcntl.F_GETFL)
                fcntl.fcntl(self.decision_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.terminal.print(f"✓ Decision server started (PID: {self.decision_process.pid})")

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

    args = parser.parse_args()

    # Create and run launcher
    launcher = LKASLauncher(
        method=args.method,
        config=args.config,
        gpu=args.gpu,
        image_shm_name=args.image_shm_name,
        detection_shm_name=args.detection_shm_name,
        control_shm_name=args.control_shm_name,
    )

    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
