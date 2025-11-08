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
from lkas.detection.core.config import ConfigManager
from lkas.constants import LauncherConstants
import yaml


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
        broadcast: bool = False,
        # Process configuration
        retry_count: int = None,
        retry_delay: float = None,
        decision_init_timeout: float = None,
        detection_init_timeout: float = None,
        process_stop_timeout: float = None,
        # Terminal configuration
        terminal_width: int = None,
        log_file: str = None,
        enable_footer: bool = None,
        # Broadcasting configuration
        jpeg_quality: int = None,
        broadcast_log_interval: int = None,
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
            broadcast: Enable ZMQ broadcasting for remote viewers
            retry_count: Number of retries (overrides config)
            retry_delay: Delay between retries (overrides config)
            decision_init_timeout: Decision server init timeout (overrides config)
            detection_init_timeout: Detection server init timeout (overrides config)
            process_stop_timeout: Process stop timeout (overrides config)
            terminal_width: Terminal width (overrides config)
            log_file: Log file path (overrides config)
            enable_footer: Enable footer (overrides config)
            jpeg_quality: JPEG quality (overrides config)
            broadcast_log_interval: Broadcast log interval (overrides config)
        """
        # Core configuration
        self.method = method
        self.config = config
        self.gpu = gpu
        self.image_shm_name = image_shm_name
        self.detection_shm_name = detection_shm_name
        self.control_shm_name = control_shm_name
        self.verbose = verbose
        self.broadcast = broadcast

        # Load config for shared memory setup
        self.system_config = ConfigManager.load(self.config)

        # Load launcher config from yaml (if config path provided)
        launcher_config = {}
        if self.config:
            try:
                config_path = Path(self.config)
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        yaml_data = yaml.safe_load(f)
                        launcher_config = yaml_data.get('launcher', {}) if yaml_data else {}
            except Exception:
                pass  # Use empty dict if loading fails

        # Process configuration: parameters > config.yaml > constants
        self.retry_count = (
            retry_count if retry_count is not None
            else launcher_config.get('retry_count', LauncherConstants.DEFAULT_RETRY_COUNT)
        )
        self.retry_delay = (
            retry_delay if retry_delay is not None
            else launcher_config.get('retry_delay', LauncherConstants.DEFAULT_RETRY_DELAY)
        )
        self.decision_init_timeout = (
            decision_init_timeout if decision_init_timeout is not None
            else launcher_config.get('decision_init_timeout', LauncherConstants.DEFAULT_DECISION_INIT_TIMEOUT)
        )
        self.detection_init_timeout = (
            detection_init_timeout if detection_init_timeout is not None
            else launcher_config.get('detection_init_timeout', LauncherConstants.DEFAULT_DETECTION_INIT_TIMEOUT)
        )
        self.process_stop_timeout = (
            process_stop_timeout if process_stop_timeout is not None
            else launcher_config.get('process_stop_timeout', LauncherConstants.DEFAULT_PROCESS_STOP_TIMEOUT)
        )

        # Terminal configuration
        self.terminal_width = (
            terminal_width if terminal_width is not None
            else launcher_config.get('terminal_width', LauncherConstants.DEFAULT_TERMINAL_WIDTH)
        )
        self.log_file_path = (
            log_file if log_file is not None
            else launcher_config.get('log_file', LauncherConstants.DEFAULT_LOG_FILE)
        )
        self.enable_footer = (
            enable_footer if enable_footer is not None
            else launcher_config.get('enable_footer', True)
        )

        # Broadcasting configuration
        self.jpeg_quality = (
            jpeg_quality if jpeg_quality is not None
            else launcher_config.get('jpeg_quality', LauncherConstants.DEFAULT_JPEG_QUALITY)
        )
        self.broadcast_log_interval = (
            broadcast_log_interval if broadcast_log_interval is not None
            else launcher_config.get('broadcast_log_interval', LauncherConstants.DEFAULT_BROADCAST_LOG_INTERVAL)
        )

        # Process handles
        self.detection_process: Optional[subprocess.Popen] = None
        self.decision_process: Optional[subprocess.Popen] = None
        self.running = False
        self.broker = None

        # Shared memory channels for broadcasting (only if broadcast enabled)
        self.image_channel = None
        self.detection_channel = None

        # Terminal display with persistent footer
        subprocess_prefix = LauncherConstants.DEFAULT_SUBPROCESS_PREFIX
        self.terminal = TerminalDisplay(enable_footer=self.enable_footer)
        self.detection_logger = OrderedLogger(subprocess_prefix, self.terminal)
        self.decision_logger = OrderedLogger(subprocess_prefix, self.terminal)

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
            str(self.retry_count),
            "--retry-delay",
            str(self.retry_delay),
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
            str(self.retry_count),
            "--retry-delay",
            str(self.retry_delay),
        ]

        if self.config:
            cmd.extend(["--config", self.config])

        if not self.verbose:
            cmd.append("--no-stats")

        return cmd

    def _print_header(self):
        """Print startup header."""
        separator = "=" * self.terminal_width
        title_padding = " " * ((self.terminal_width - len("LKAS System Launcher")) // 2)

        self.terminal.print("\n" + separator)
        self.terminal.print(title_padding + "LKAS System Launcher")
        self.terminal.print(separator)
        self.terminal.print(f"  Detection Method: {self.method.upper()}")
        if self.gpu is not None:
            self.terminal.print(f"  GPU Device: {self.gpu}")
        if self.config:
            self.terminal.print(f"  Config: {self.config}")
        self.terminal.print(f"  Image SHM: {self.image_shm_name}")
        self.terminal.print(f"  Detection SHM: {self.detection_shm_name}")
        self.terminal.print(f"  Control SHM: {self.control_shm_name}")
        if self.broadcast:
            self.terminal.print(f"  ZMQ Broadcast: Enabled")
        self.terminal.print(separator)
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
                data = process.stdout.read(LauncherConstants.DEFAULT_BUFFER_READ_SIZE)
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
                data = process.stdout.read(LauncherConstants.DEFAULT_BUFFER_READ_SIZE)
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

    def _setup_broker(self):
        """Setup ZMQ broker for routing parameters and actions, and shared memory readers for broadcasting."""
        if not self.broadcast:
            return

        try:
            from lkas.integration.zmq import LKASBroker

            self.terminal.print("\nInitializing ZMQ broker (routing & broadcasting)...")
            self.broker = LKASBroker()
            self.terminal.print("")
        except Exception as e:
            self.terminal.print(f"✗ Failed to initialize ZMQ broker: {e}")
            self.terminal.print("  Continuing without broker...")
            self.broker = None

    def _setup_shared_memory_readers(self):
        """
        Setup shared memory readers for broadcasting frames and detection.
        Called after servers are initialized, but connection is lazy (on first use).
        """
        if not self.broadcast or not self.broker:
            return

        # Don't connect immediately - wait for simulation to create shared memory
        # Connection will happen on first _broadcast_data() call
        self.terminal.print("\n✓ Broadcasting enabled - will connect to shared memory when available")

    def _broadcast_data(self):
        """
        Read from shared memory and broadcast frames/detection to viewers.
        Called periodically in the main loop.
        Lazy-connects to shared memory on first successful read.
        """
        if not self.broker:
            return

        # Lazy connection to image channel
        if self.image_channel is None:
            try:
                from lkas.detection.integration.shared_memory_detection import SharedMemoryImageChannel
                self.image_channel = SharedMemoryImageChannel(
                    name=self.image_shm_name,
                    shape=(self.system_config.camera.height, self.system_config.camera.width, 3),
                    create=False,  # Reader mode
                    retry_count=1,  # Single attempt
                    retry_delay=0.0,
                )
                self.terminal.print(f"✓ Connected to image channel: {self.image_shm_name}")
            except Exception:
                pass  # Will retry on next call

        # Lazy connection to detection channel
        if self.detection_channel is None:
            try:
                from lkas.detection.integration.shared_memory_detection import SharedMemoryDetectionChannel
                self.detection_channel = SharedMemoryDetectionChannel(
                    name=self.detection_shm_name,
                    create=False,  # Reader mode
                    retry_count=1,  # Single attempt
                    retry_delay=0.0,
                )
                self.terminal.print(f"✓ Connected to detection channel: {self.detection_shm_name}")
            except Exception:
                pass  # Will retry on next call

        # Try to read and broadcast image
        if self.image_channel:
            try:
                image_msg = self.image_channel.read(copy=False)  # Non-blocking read
                if image_msg is not None:
                    self.broker.broadcast_frame(
                        image_msg.image,
                        image_msg.frame_id,
                        jpeg_quality=self.jpeg_quality
                    )
            except Exception:
                pass  # Silently ignore read errors

        # Try to read and broadcast detection
        if self.detection_channel:
            try:
                detection_msg = self.detection_channel.read()  # Non-blocking read
                if detection_msg is not None:
                    # Convert detection message to viewer format (line segments, not arrays)
                    # Viewer expects DetectionData: {left_lane, right_lane, processing_time_ms, frame_id}
                    # Each lane: {x1, y1, x2, y2, confidence} or None
                    detection_data = {
                        'left_lane': {
                            'x1': float(detection_msg.left_lane.x1),
                            'y1': float(detection_msg.left_lane.y1),
                            'x2': float(detection_msg.left_lane.x2),
                            'y2': float(detection_msg.left_lane.y2),
                            'confidence': float(detection_msg.left_lane.confidence)
                        } if detection_msg.left_lane is not None else None,
                        'right_lane': {
                            'x1': float(detection_msg.right_lane.x1),
                            'y1': float(detection_msg.right_lane.y1),
                            'x2': float(detection_msg.right_lane.x2),
                            'y2': float(detection_msg.right_lane.y2),
                            'confidence': float(detection_msg.right_lane.confidence)
                        } if detection_msg.right_lane is not None else None,
                        'processing_time_ms': detection_msg.processing_time_ms,
                        'frame_id': detection_msg.frame_id,
                    }
                    self.broker.broadcast_detection(detection_data, detection_msg.frame_id)
                    # Log successful broadcast at configured interval
                    if detection_msg.frame_id % self.broadcast_log_interval == 0:
                        self.terminal.print(f"[green]✓ Broadcasting detection: frame {detection_msg.frame_id}, L:{detection_msg.left_lane is not None}, R:{detection_msg.right_lane is not None}[/green]")
            except Exception as e:
                # Log errors to help diagnose issues
                self.terminal.print(f"[yellow]Warning: Failed to broadcast detection: {e}[/yellow]")

    def run(self):
        """Start both servers and manage their lifecycle."""
        self._print_header()

        # Open log file
        self.log_file = open(self.log_file_path, "w", buffering=1)

        # Setup ZMQ broker if enabled
        self._setup_broker()

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
            # Buffer to capture the initial setup messages
            init_timeout = time.time() + self.decision_init_timeout
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
            time.sleep(LauncherConstants.DEFAULT_POST_DECISION_DELAY)

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
            # Buffer to capture the initial setup messages
            init_timeout = time.time() + self.detection_init_timeout
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

            # Setup shared memory readers for broadcasting (after servers are ready)
            self._setup_shared_memory_readers()

            separator = "=" * self.terminal_width
            wait_time = int(self.retry_count * self.retry_delay)
            self.terminal.print("\n" + separator)
            self.terminal.print("System running - Press Ctrl+C to stop")
            self.terminal.print("")
            self.terminal.print(f"Servers will wait up to {wait_time} seconds for connections:")
            self.terminal.print("  - Detection server waiting for camera_feed from simulation")
            self.terminal.print("  - Decision server waiting for detection_results from detection")
            self.terminal.print("")
            self.terminal.print("Ready! Start 'simulation' in another terminal to begin processing")
            self.terminal.print(separator + "\n")

            # Initialize footer
            self._update_footer()

            self.running = True

            # Main loop - multiplex output from both processes
            while self.running:
                # Poll broker for parameter updates and action requests
                if self.broker:
                    self.broker.poll()

                # Broadcast frames and detection data to viewers
                if self.broadcast:
                    self._broadcast_data()

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
                time.sleep(LauncherConstants.DEFAULT_MAIN_LOOP_SLEEP)

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

        # Close shared memory readers
        if self.image_channel:
            try:
                self.image_channel.close()
            except:
                pass
        if self.detection_channel:
            try:
                self.detection_channel.close()
            except:
                pass

        # Stop broker
        if self.broker:
            self.broker.close()
            self.broker = None

        # Stop decision server first (consumer)
        if self.decision_process and self.decision_process.poll() is None:
            self.terminal.print("Stopping decision server...")
            self.decision_process.terminate()
            try:
                self.decision_process.wait(timeout=self.process_stop_timeout)
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
                self.detection_process.wait(timeout=self.process_stop_timeout)
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
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="Enable ZMQ broadcasting for remote viewers (parameter updates, state, actions)",
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
        broadcast=args.broadcast,
    )

    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
