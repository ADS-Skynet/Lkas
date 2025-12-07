"""
LKAS System Launcher (Refactored)

Manages the lifecycle of detection and decision servers.
Provides process management, log multiplexing, and ZMQ broadcasting.

This is a modular refactoring that separates concerns:
- ProcessManager: Handles subprocess lifecycle
- BroadcastManager: Handles ZMQ and shared memory broadcasting
- LKASServer: Orchestrates everything
"""

import sys
import signal
import time
import yaml
from pathlib import Path

from lkas.utils.terminal import TerminalDisplay
from skynet_common.config import ConfigManager
from lkas.constants import Launcher
from lkas.process import DetectionProcessManager, DecisionProcessManager
from lkas.broadcast import BroadcastManager


class LKASServer:
    """Launches and manages both detection and decision servers."""

    def __init__(
        self,
        method: str = "cv",
        config: str | None = None,
        gpu: int | None = None,
        verbose: bool = False,
        broadcast: bool = False,
        # Shared memory names (None = use from config)
        image_shm_name: str | None = None,
        detection_shm_name: str | None = None,
        control_shm_name: str | None = None,
        # Process configuration
        retry_count: int | None = None,
        retry_delay: float | None = None,
        decision_init_timeout: float | None = None,
        detection_init_timeout: float | None = None,
        process_stop_timeout: float | None = None,
        # Terminal configuration
        terminal_width: int | None = None,
        log_file: str | None = None,
        enable_footer: bool | None = None,
        # Broadcasting configuration
        jpeg_quality: int | None = None,
        broadcast_log_interval: int | None = None,
    ):
        """Initialize LKAS launcher."""
        # Core configuration
        self.method = method
        self.config = config
        self.gpu = gpu
        self.verbose = verbose
        self.broadcast = broadcast

        # Load config
        self.system_config = ConfigManager.load(self.config)

        # Resolve shared memory names
        self.image_shm_name = image_shm_name or self.system_config.communication.image_shm_name
        self.detection_shm_name = detection_shm_name or self.system_config.communication.detection_shm_name
        self.control_shm_name = control_shm_name or self.system_config.communication.control_shm_name

        # Load launcher config from yaml
        launcher_config = self._load_launcher_config()

        # Resolve process configuration
        use_extended_retry = (self.method == "dl")
        default_retry_count = (
            self.system_config.retry.extended_max_retries if use_extended_retry
            else self.system_config.retry.max_retries
        )
        default_retry_delay = (
            self.system_config.retry.extended_retry_delay_s if use_extended_retry
            else self.system_config.retry.retry_delay_s
        )

        self.retry_count = retry_count or launcher_config.get('retry_count', default_retry_count)
        self.retry_delay = retry_delay or launcher_config.get('retry_delay', default_retry_delay)
        self.decision_init_timeout = (
            decision_init_timeout or launcher_config.get('decision_init_timeout',
            self.system_config.launcher.decision_init_timeout_s)
        )
        self.detection_init_timeout = (
            detection_init_timeout or launcher_config.get('detection_init_timeout',
            self.system_config.launcher.detection_init_timeout_s)
        )
        self.process_stop_timeout = (
            process_stop_timeout or launcher_config.get('process_stop_timeout',
            self.system_config.launcher.process_stop_timeout_s)
        )

        # Terminal configuration
        self.terminal_width = terminal_width or launcher_config.get('terminal_width', self.system_config.launcher.terminal_width)
        self.log_file_path = log_file or launcher_config.get('log_file', self.system_config.launcher.log_file)
        self.enable_footer = enable_footer if enable_footer is not None else Launcher.ENABLE_FOOTER

        # Broadcasting configuration
        jpeg_quality = jpeg_quality or launcher_config.get('jpeg_quality', self.system_config.streaming.jpeg_quality)
        raw_rgb = launcher_config.get('raw_rgb', self.system_config.streaming.raw_rgb)
        broadcast_log_interval = (
            broadcast_log_interval or launcher_config.get('broadcast_log_interval',
            self.system_config.streaming.broadcast_log_interval)
        )

        # Terminal display
        self.terminal = TerminalDisplay(enable_footer=self.enable_footer)

        # Process managers
        from lkas.utils.terminal import OrderedLogger

        self.detection_manager = DetectionProcessManager(
            logger=OrderedLogger("[Detection]", self.terminal),
            method=self.method,
            config=self.config,
            gpu=self.gpu,
            image_shm_name=self.image_shm_name,
            detection_shm_name=self.detection_shm_name,
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
            verbose=self.verbose,
            buffer_read_size=self.system_config.launcher.buffer_read_size,
        )

        self.decision_manager = DecisionProcessManager(
            logger=OrderedLogger("[Decision]", self.terminal),
            config=self.config,
            detection_shm_name=self.detection_shm_name,
            control_shm_name=self.control_shm_name,
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
            verbose=self.verbose,
            buffer_read_size=self.system_config.launcher.buffer_read_size,
        )

        # Broadcast manager
        self.broadcast_manager = None
        if self.broadcast:
            self.broadcast_manager = BroadcastManager(
                system_config=self.system_config,
                image_shm_name=self.image_shm_name,
                detection_shm_name=self.detection_shm_name,
                control_shm_name=self.control_shm_name,
                jpeg_quality=jpeg_quality,
                raw_rgb=raw_rgb,
                broadcast_log_interval=broadcast_log_interval,
                verbose=self.verbose,
                terminal=self.terminal,
            )

        # Shared memory status tracking
        self.shm_status = {
            self.image_shm_name: False,
            self.detection_shm_name: False,
            self.control_shm_name: False,
        }

        # Server state
        self.running = False
        self.buffering_mode = True
        self.log_file = None

    def _load_launcher_config(self) -> dict:
        """Load launcher configuration from yaml file."""
        if not self.config:
            return {}

        try:
            config_path = Path(self.config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    return yaml_data.get('launcher', {}) if yaml_data else {}
        except Exception:
            pass

        return {}

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

    def _update_footer(self):
        """Update the persistent footer with shared memory status."""
        if not self.enable_footer:
            return
        self.terminal.update_footer(shm_status=self.shm_status)

    def _check_shm_connection(self, message: str):
        """Parse subprocess output to track shared memory connections."""
        keywords = [
            f"Created image shared memory:",
            f"Connected to image shared memory:",
            f"Created detection shared memory:",
            f"Connected to detection shared memory:",
            f"Created control shared memory:",
            f"Connected to control shared memory:",
        ]

        for keyword in keywords:
            if keyword in message:
                parts = message.split(keyword)
                if len(parts) > 1:
                    shm_name = parts[1].strip().split()[0]
                    if shm_name in self.shm_status:
                        self.shm_status[shm_name] = True
                        self._update_footer()
                        break

    def _launch_processes(self):
        """Launch both detection and decision processes."""
        # Start decision server first
        self.terminal.print("Detaching decision server...")
        decision_pid = self.decision_manager.launch(self.log_file)
        self.terminal.print(f" Decision server detached (PID: {decision_pid})")

        # Small delay to let decision server start
        time.sleep(1)

        # Start detection server
        self.terminal.print("\nDetaching detection server...")
        detection_pid = self.detection_manager.launch(self.log_file)
        self.terminal.print(f" Detection server detached (PID: {detection_pid})\n")

    def _wait_for_initialization(self):
        """Wait for both processes to initialize."""
        max_wait_time = 60.0
        start_time = time.time()
        changed_context = "init"

        while not (self.detection_manager.initialized and self.decision_manager.initialized):
            # Read output from both processes
            self.detection_manager.read_output(
                buffering_mode=self.buffering_mode,
                shm_connection_callback=self._check_shm_connection
            )
            self.decision_manager.read_output(
                buffering_mode=self.buffering_mode,
                shm_connection_callback=self._check_shm_connection
            )

            # Flush buffers to show grouped output
            if self.decision_manager.logger.buffer:
                if changed_context == "det":
                    self.terminal.print("")
                self.decision_manager.logger.flush()
                changed_context = "dec"
            elif self.detection_manager.logger.buffer:
                if changed_context == "dec":
                    self.terminal.print("")
                self.detection_manager.logger.flush()
                changed_context = "det"

            # Check if processes are still alive
            if not self.detection_manager.is_alive():
                self.terminal.print(" Detection server died during initialization!")
                return False
            if not self.decision_manager.is_alive():
                self.terminal.print(" Decision server died during initialization!")
                return False

            # Timeout failsafe
            if time.time() - start_time > max_wait_time:
                self.terminal.print(" Timeout waiting for server initialization!")
                self.terminal.print(f"  Decision ready: {self.decision_manager.initialized}")
                self.terminal.print(f"  Detection ready: {self.detection_manager.initialized}")
                return False

            time.sleep(0.1)

        # Final flush
        self.decision_manager.logger.flush()
        self.detection_manager.logger.flush()

        return True

    def _print_ready_message(self):
        """Print system ready message."""
        separator = "=" * self.terminal_width
        wait_time = int(self.retry_count * self.retry_delay)

        self.terminal.print("\n" + separator)
        self.terminal.print("System running - Press Ctrl+C to stop")
        self.terminal.print("")
        self.terminal.print(f"Servers will wait up to '{wait_time}' seconds for connections:")
        self.terminal.print(f"  - Detection server waiting for '{self.image_shm_name}' from vehicle")
        self.terminal.print(f"  - Decision server waiting for '{self.detection_shm_name}' from detection")
        self.terminal.print("")
        self.terminal.print("Ready! Start 'vehicle' or 'simulation' in another terminal to begin processing")
        self.terminal.print(separator + "\n")

    def _main_loop(self):
        """Main processing loop."""
        while self.running:
            # Poll broker and broadcast data
            if self.broadcast_manager:
                self.broadcast_manager.poll_broker()
                self.broadcast_manager.broadcast_data()

            # Read output from processes
            self.detection_manager.read_output(
                buffering_mode=self.buffering_mode,
                shm_connection_callback=self._check_shm_connection
            )
            self.decision_manager.read_output(
                buffering_mode=self.buffering_mode,
                shm_connection_callback=self._check_shm_connection
            )

            # Check if processes are still alive
            if not self.detection_manager.is_alive():
                self.terminal.clear_footer()
                self.terminal.print(" Detection server died unexpectedly!")
                self.running = False
                break

            if not self.decision_manager.is_alive():
                self.terminal.clear_footer()
                self.terminal.print(" Decision server died unexpectedly!")
                self.running = False
                break

            # Small sleep to prevent busy-waiting
            time.sleep(self.system_config.timing.main_loop_sleep_s)

    def run(self):
        """Start both servers and manage their lifecycle."""
        self._print_header()

        # Open log file
        if self.log_file_path:
            self.log_file = open(self.log_file_path, "w", buffering=1)
            print(f" Logging to: {self.log_file_path}\n")
        else:
            print(" No log file configured")

        # Setup broadcast manager
        if self.broadcast_manager:
            self.broadcast_manager.setup()

        # Initialize footer
        self.terminal.init_footer()

        # Register signal handlers
        def signal_handler(sig, frame):
            self.terminal.clear_footer()
            self.terminal.print("\nReceived interrupt signal - shutting down...")
            if self.log_file:
                self.log_file.close()
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Launch processes
            self._launch_processes()

            # Wait for initialization
            if not self._wait_for_initialization():
                return 1

            # Disable buffering for runtime messages
            self.buffering_mode = False

            # Setup broadcasting
            if self.broadcast_manager:
                self.broadcast_manager.announce_ready()

            # Print ready message
            self._print_ready_message()

            # Initialize footer
            self._update_footer()

            # Start main loop
            self.running = True
            self._main_loop()

        except Exception as e:
            self.terminal.clear_footer()
            self.terminal.print(f" Error: {e}")
            return 1
        finally:
            self.terminal.clear_footer()
            if self.log_file:
                self.log_file.close()
            self.stop()

        return 0

    def stop(self):
        """Stop both servers gracefully."""
        self.running = False
        self.terminal.print("\nStopping servers...")

        # Close broadcast manager
        if self.broadcast_manager:
            self.broadcast_manager.close()

        # Stop decision server first (consumer)
        self.decision_manager.stop(timeout=self.process_stop_timeout, terminal=self.terminal)

        # Stop detection server (producer)
        self.detection_manager.stop(timeout=self.process_stop_timeout, terminal=self.terminal)

        self.terminal.print(" Cleanup complete")
