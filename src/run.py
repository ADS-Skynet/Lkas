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
from typing import List

from lkas.utils.terminal import TerminalDisplay, OrderedLogger
from skynet_common.config import ConfigManager
from lkas.constants import Launcher
import yaml


class LKASLauncher:
    """Launches and manages both detection and decision servers."""

    def __init__(
        self,
        method: str = "cv",
        config: str | None = None,
        gpu: int | None = None,
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
        self.verbose = verbose
        self.broadcast = broadcast

        # Load config for shared memory setup
        self.system_config = ConfigManager.load(self.config)

        # Use config values if SHM names not specified via command-line
        self.image_shm_name = (
            image_shm_name if image_shm_name is not None
            else self.system_config.communication.image_shm_name
        )
        self.detection_shm_name = (
            detection_shm_name if detection_shm_name is not None
            else self.system_config.communication.detection_shm_name
        )
        self.control_shm_name = (
            control_shm_name if control_shm_name is not None
            else self.system_config.communication.control_shm_name
        )

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

        # Process configuration: parameters > config.yaml > system_config defaults
        # Use extended retry settings for DL method (model loading takes longer)
        use_extended_retry = (self.method == "dl")
        default_retry_count = (
            self.system_config.retry.extended_max_retries if use_extended_retry
            else self.system_config.retry.max_retries
        )
        default_retry_delay = (
            self.system_config.retry.extended_retry_delay_s if use_extended_retry
            else self.system_config.retry.retry_delay_s
        )

        self.retry_count = (
            retry_count if retry_count is not None
            else launcher_config.get('retry_count', default_retry_count)
        )
        self.retry_delay = (
            retry_delay if retry_delay is not None
            else launcher_config.get('retry_delay', default_retry_delay)
        )
        self.decision_init_timeout = (
            decision_init_timeout if decision_init_timeout is not None
            else launcher_config.get('decision_init_timeout', self.system_config.launcher.decision_init_timeout_s)
        )
        self.detection_init_timeout = (
            detection_init_timeout if detection_init_timeout is not None
            else launcher_config.get('detection_init_timeout', self.system_config.launcher.detection_init_timeout_s)
        )
        self.process_stop_timeout = (
            process_stop_timeout if process_stop_timeout is not None
            else launcher_config.get('process_stop_timeout', self.system_config.launcher.process_stop_timeout_s)
        )

        # Terminal configuration
        self.terminal_width = (
            terminal_width if terminal_width is not None
            else launcher_config.get('terminal_width', self.system_config.launcher.terminal_width)
        )
        self.log_file_path = (
            log_file if log_file is not None
            else launcher_config.get('log_file', self.system_config.launcher.log_file)
        )
        self.enable_footer = (
            enable_footer if enable_footer is not None
            else Launcher.ENABLE_FOOTER
        )

        # Broadcasting configuration
        self.jpeg_quality = (
            jpeg_quality if jpeg_quality is not None
            else launcher_config.get('jpeg_quality', self.system_config.streaming.jpeg_quality)
        )
        self.raw_rgb = launcher_config.get('raw_rgb', self.system_config.streaming.raw_rgb)
        self.broadcast_log_interval = (
            broadcast_log_interval if broadcast_log_interval is not None
            else launcher_config.get('broadcast_log_interval', self.system_config.streaming.broadcast_log_interval)
        )

        # Process handles
        self.detection_process: subprocess.Popen | None = None
        self.decision_process: subprocess.Popen | None = None
        self.running = False
        self.broker = None

        # Shared memory channels for broadcasting (only if broadcast enabled)
        self.image_channel = None
        self.detection_channel = None
        self.control_channel = None
        self.last_broadcast_frame_id = -1
        self.last_broadcast_detection_id = -1

        # FPS tracking for footer
        self.broadcast_frame_count = 0
        self.broadcast_detection_count = 0
        self.last_fps_update_time = time.time()
        self.current_frame_fps = 0.0
        self.current_detection_fps = 0.0

        # Terminal display with persistent footer
        subprocess_prefix = self.system_config.launcher.subprocess_prefix
        self.terminal = TerminalDisplay(enable_footer=self.enable_footer)
        # self.detection_logger = OrderedLogger(subprocess_prefix, self.terminal)
        # self.decision_logger = OrderedLogger(subprocess_prefix, self.terminal)

        self.detection_logger = OrderedLogger("[Detection]", self.terminal)
        self.decision_logger = OrderedLogger("[Decision]", self.terminal)


        # Shared memory status tracking for footer
        self.shm_status = {
            self.image_shm_name: False,
            self.detection_shm_name: False,
            self.control_shm_name: False,
        }

        # Initialization phase tracking
        self.buffering_mode = True
        self.decision_initialized = False
        self.detection_initialized = False

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

    def _read_process_output(self, process: subprocess.Popen, logger: OrderedLogger):
        """
        Read and print process output with prefix.
        Non-blocking read using select.

        Args:
            process: Process to read from
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
                data = process.stdout.read(self.system_config.launcher.buffer_read_size)
                if data:
                    lines_raw = data.decode('utf-8')
                    lines = lines_raw.splitlines(keepends=False)
                    # for line in lines_raw.splitlines(keepends=False):
                    #     print(repr(line))

                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            continue

                        # Track shared memory connections from subprocess output
                        self._check_shm_connection(stripped)

                        # Check for initialization completion markers
                        if "Decision Server Started" in stripped:
                            self.decision_initialized = True
                        elif "Detection Server Started" in stripped:
                            self.detection_initialized = True

                        # Print to terminal (buffer during init for ordering, immediate during runtime)
                        if self.buffering_mode:
                            # Buffer messages during initialization for ordered output
                            logger.log(stripped)
                        else:
                            # Print immediately during runtime
                            logger.print_immediate(stripped)

            except BlockingIOError:
                # No data available - this is OK for non-blocking I/O
                pass
            except Exception:
                pass
        else:
            # Fallback for Windows
            try:
                # Try to read available data
                data = process.stdout.read(self.system_config.launcher.buffer_read_size)
                if data:
                    lines_raw = data.decode('utf-8')
                    lines = lines_raw.splitlines(keepends=True)

                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            continue

                        # Track shared memory connections from subprocess output
                        self._check_shm_connection(stripped)

                        # Check for initialization completion markers
                        if "Decision Server Running" in stripped:
                            self.decision_initialized = True
                        elif "Detection Server Running" in stripped:
                            self.detection_initialized = True

                        # Print to terminal (buffer during init for ordering, immediate during runtime)
                        if self.buffering_mode:
                            # Buffer messages during initialization for ordered output
                            logger.log(stripped)
                        else:
                            # Print immediately during runtime
                            logger.print_immediate(stripped, line.endswith('\r'))
            except Exception:
                pass

    def _update_footer(self):
        """Update the persistent footer with shared memory status."""
        if not self.enable_footer:
            return

        self.terminal.update_footer(shm_status=self.shm_status)

    def _check_shm_connection(self, message: str):
        """
        Parse subprocess output to track shared memory connections.

        Args:
            message: Log message from subprocess
        """
        # Look for various shared memory creation/connection messages
        keywords = [
            "Created shared memory:",
            "Connected to shared memory:",
            "Created detection shared memory:",
            "Connected to detection shared memory:",
            "Created control shared memory:",
            "Connected to control shared memory:",
        ]

        for keyword in keywords:
            if keyword in message:
                # Extract the shared memory name
                parts = message.split(keyword)
                if len(parts) > 1:
                    shm_name = parts[1].strip().split()[0]  # Get first word after keyword
                    # Update status if this is one of our tracked shared memories
                    if shm_name in self.shm_status:
                        self.shm_status[shm_name] = True
                        self._update_footer()
                        break

    def _setup_broker(self):
        """Setup ZMQ broker for routing parameters and actions, and shared memory readers for broadcasting."""
        if not self.broadcast:
            return

        try:
            from lkas.integration.zmq import LKASBroker

            self.terminal.print("\nInitializing ZMQ broker (routing & broadcasting)...")
            self.broker = LKASBroker(verbose=self.verbose)
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
                import sys
                import io
                from lkas.integration.shared_memory import SharedMemoryImageChannel

                # Suppress stdout during connection to avoid empty line from flush()
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    self.image_channel = SharedMemoryImageChannel(
                        name=self.image_shm_name,
                        shape=(self.system_config.camera.height, self.system_config.camera.width, 3),
                        create=False,  # Reader mode
                        retry_count=1,  # Single attempt
                        retry_delay=0.0,
                    )
                finally:
                    sys.stdout = old_stdout

                self.terminal.print(f"✓ Connected to image channel: {self.image_shm_name}")
                self.shm_status[self.image_shm_name] = True
                self._update_footer()
            except Exception:
                pass  # Will retry on next call

        # Lazy connection to detection channel
        if self.detection_channel is None:
            try:
                from lkas.integration.shared_memory import SharedMemoryDetectionChannel
                self.detection_channel = SharedMemoryDetectionChannel(
                    name=self.detection_shm_name,
                    create=False,  # Reader mode
                    retry_count=1,  # Single attempt
                    retry_delay=0.0,
                )
                self.terminal.print(f"✓ Connected to detection channel: {self.detection_shm_name}")
                # Note: detection_shm_name status is already set by subprocess output
            except Exception:
                pass  # Will retry on next call

        # Lazy connection to control channel
        if self.control_channel is None:
            try:
                from lkas.integration.shared_memory import SharedMemoryControlChannel
                self.control_channel = SharedMemoryControlChannel(
                    name=self.control_shm_name,
                    create=False,  # Reader mode
                    retry_count=1,  # Single attempt
                    retry_delay=0.0,
                )
                self.terminal.print(f"✓ Connected to control channel: {self.control_shm_name}")
            except Exception:
                pass  # Will retry on next call

        # Try to read and broadcast image
        if self.image_channel:
            try:
                image_msg = self.image_channel.read(copy=False)  # Non-blocking read
                if image_msg is not None and image_msg.frame_id != self.last_broadcast_frame_id:
                    self.broker.broadcast_frame(
                        image_msg.image,
                        image_msg.frame_id,
                        jpeg_quality=self.jpeg_quality,
                        raw_rgb=self.raw_rgb
                    )
                    self.last_broadcast_frame_id = image_msg.frame_id
                    self.broadcast_frame_count += 1
            except Exception:
                pass  # Silently ignore read errors

        # Try to read and broadcast detection with metrics from control
        if self.detection_channel:
            try:
                detection_msg = self.detection_channel.read()  # Non-blocking read
                control_msg = None
                if self.control_channel:
                    control_msg = self.control_channel.read()  # Non-blocking read

                if detection_msg is not None and detection_msg.frame_id != self.last_broadcast_detection_id:
                    # Convert detection message to viewer format (line segments, not arrays)
                    # Viewer expects DetectionData: {left_lane, right_lane, processing_time_ms, frame_id, metrics}
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

                    # Extract metrics from control message (single source of truth!)
                    # Metrics are calculated once in controller, not recalculated here
                    if control_msg is not None:
                        # Use metrics from control message
                        detection_data['lateral_offset_meters'] = control_msg.lateral_offset_meters
                        detection_data['heading_angle_deg'] = control_msg.heading_angle
                        detection_data['lane_width_pixels'] = control_msg.lane_width_pixels
                        detection_data['departure_status'] = control_msg.departure_status
                    else:
                        # No control data available, use None for all metrics
                        detection_data['lateral_offset_meters'] = None
                        detection_data['heading_angle_deg'] = None
                        detection_data['lane_width_pixels'] = None
                        detection_data['departure_status'] = None

                        # Warn about missing control data (only at log interval)
                        if self.verbose and detection_msg.frame_id % self.broadcast_log_interval == 0:
                            self.terminal.print(f"[Broker] Warning: No control data for frame {detection_msg.frame_id}")

                    self.broker.broadcast_detection(detection_data, detection_msg.frame_id)
                    self.last_broadcast_detection_id = detection_msg.frame_id
                    self.broadcast_detection_count += 1

                    # Log successful broadcast at configured interval (only in verbose mode)
                    if self.verbose and detection_msg.frame_id % self.broadcast_log_interval == 0:
                        has_metrics = control_msg is not None
                        self.terminal.print(
                            f"[Broker] Frame {detection_msg.frame_id}: "
                            f"L:{detection_msg.left_lane is not None}, R:{detection_msg.right_lane is not None} | "
                            f"Metrics: offset={detection_data['lateral_offset_meters']:.3f}m, "
                            f"status={detection_data['departure_status']}" if has_metrics
                            else f"[Broker] Frame {detection_msg.frame_id}: "
                            f"L:{detection_msg.left_lane is not None}, R:{detection_msg.right_lane is not None} | "
                            f"Metrics: N/A"
                        )
            except Exception as e:
                # Log errors to help diagnose issues
                self.terminal.print(f"Warning: Failed to broadcast detection: {e}")

        # Update FPS stats in footer (every 1 second)
        now = time.time()
        if now - self.last_fps_update_time >= 1.0:
            elapsed = now - self.last_fps_update_time
            self.current_frame_fps = self.broadcast_frame_count / elapsed
            self.current_detection_fps = self.broadcast_detection_count / elapsed

            # Update footer with FPS stats
            self.terminal.update_footer(fps_stats={
                'Frame': self.current_frame_fps,
                'Detection': self.current_detection_fps
            })

            self.broadcast_frame_count = 0
            self.broadcast_detection_count = 0
            self.last_fps_update_time = now

    def run(self):
        """Start both servers and manage their lifecycle."""
        self._print_header()

        # Open log file
        if self.log_file_path:
            self.log_file = open(self.log_file_path, "w", buffering=1)
            print(f"✓ Logging to: {self.log_file_path}\n")
        else:
            self.log_file = None
            print("⚠ No log file configured")

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
            self.terminal.print("Detaching decision server...")
            decision_cmd = self._build_decision_cmd()
            # Set PYTHONUNBUFFERED to ensure output is not buffered
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            # Redirect stderr to main log file for error capture
            self.decision_process = subprocess.Popen(
                decision_cmd,
                stdout=subprocess.PIPE,
                stderr=self.log_file if self.log_file else subprocess.STDOUT,  # Redirect stderr to main log file
                bufsize=0,  # Unbuffered
                env=env,
            )
            # Make stdout non-blocking
            if self.decision_process.stdout:
                flags = fcntl.fcntl(self.decision_process.stdout, fcntl.F_GETFL)
                fcntl.fcntl(self.decision_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.terminal.print(f"✓ Decision server detached (PID: {self.decision_process.pid})")

            # Small delay to let decision server start
            time.sleep(1)

            # Start detection server in parallel (don't wait for decision to finish)
            self.terminal.print("\nDetaching detection server...")
            detection_cmd = self._build_detection_cmd()
            # Set PYTHONUNBUFFERED to ensure output is not buffered
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            # Redirect stderr to main log file for error capture
            self.detection_process = subprocess.Popen(
                detection_cmd,
                stdout=subprocess.PIPE,
                stderr=self.log_file if self.log_file else subprocess.STDOUT,  # Redirect stderr to main log file
                bufsize=0,  # Unbuffered
                env=env,
            )
            # Make stdout non-blocking
            if self.detection_process.stdout:
                flags = fcntl.fcntl(self.detection_process.stdout, fcntl.F_GETFL)
                fcntl.fcntl(self.detection_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.terminal.print(f"✓ Detection server detached (PID: {self.detection_process.pid})\n")

            # Use micro-buffering during initialization for real-time output with context preservation
            # Buffer messages briefly, then flush periodically to keep contexts together
            max_wait_time = 60.0  # Maximum 60 seconds as failsafe
            start_time = time.time()
            changed_context = "init"

            while not (self.detection_initialized and self.decision_initialized):
                # Buffer messages from both processes
                self._read_process_output(self.detection_process, self.detection_logger)
                self._read_process_output(self.decision_process, self.decision_logger)

                if self.decision_logger.buffer:
                    if changed_context == "det":
                        self.terminal.print("")
                    self.decision_logger.flush()
                    changed_context = "dec"
                elif self.detection_logger.buffer:
                    if changed_context == "dec":
                        self.terminal.print("")
                    self.detection_logger.flush()
                    changed_context = "det"

                # Check if processes are still alive
                if self.detection_process.poll() is not None:
                    # Flush remaining buffers before reporting error
                    self.terminal.print("✗ Detection server died during initialization!")
                    return 1
                if self.decision_process.poll() is not None:
                    # Flush remaining buffers before reporting error
                    self.terminal.print("✗ Decision server died during initialization!")
                    return 1

                # Timeout failsafe
                if time.time() - start_time > max_wait_time:
                    # Flush remaining buffers before reporting error
                    self.terminal.print("✗ Timeout waiting for server initialization!")
                    self.terminal.print(f"  Decision ready: {self.decision_initialized}")
                    self.terminal.print(f"  Detection ready: {self.detection_initialized}")
                    return 1

                time.sleep(0.1)

            # Final flush of any remaining buffered messages
            self.decision_logger.flush()
            self.detection_logger.flush()

            # Now disable buffering for real-time runtime messages
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
                self._read_process_output(self.detection_process, self.detection_logger)
                self._read_process_output(self.decision_process, self.decision_logger)

                # Then check if processes are still alive
                detection_alive = self.detection_process.poll() is None
                decision_alive = self.decision_process.poll() is None

                if not detection_alive:
                    # Try to read remaining output
                    self.terminal.clear_footer()
                    self.terminal.print("✗ Detection server died unexpectedly!")
                    self.running = False
                    break

                if not decision_alive:
                    # Try to read remaining output before reporting death
                    self.terminal.clear_footer()
                    self.terminal.print("✗ Decision server died unexpectedly!")
                    self.running = False
                    break

                # Small sleep to prevent busy-waiting
                time.sleep(self.system_config.timing.main_loop_sleep_s)

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
            except Exception as e:
                # Log but don't fail - shared memory might already be unlinked
                if self.verbose:
                    self.terminal.print(f"  (Image channel close: {e})")
        if self.detection_channel:
            try:
                self.detection_channel.close()
            except Exception as e:
                # Log but don't fail - shared memory might already be unlinked
                if self.verbose:
                    self.terminal.print(f"  (Detection channel close: {e})")

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
        default=None,
        help="Shared memory name for camera images (default: from config)",
    )
    parser.add_argument(
        "--detection-shm-name",
        type=str,
        default=None,
        help="Shared memory name for detection results (default: from config)",
    )
    parser.add_argument(
        "--control-shm-name",
        type=str,
        default=None,
        help="Shared memory name for control commands (default: from config)",
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
