"""
Process Management Module

Provides modular, reusable process handling for LKAS subprocesses.
Extracts common patterns from server initialization and lifecycle management.
"""

import sys
import os
import subprocess
import fcntl
import time
import select
from typing import List, Optional, Callable
from abc import ABC, abstractmethod

from lkas.utils.terminal import OrderedLogger


class ProcessManager(ABC):
    """
    Base class for managing LKAS subprocesses.

    Handles common operations:
    - Command building
    - Process launching
    - Output reading
    - Lifecycle management
    """

    def __init__(
        self,
        logger: OrderedLogger,
        retry_count: int,
        retry_delay: float,
        buffer_read_size: int = 4096,
    ):
        """
        Initialize process manager.

        Args:
            logger: Logger for process output
            retry_count: Number of connection retries
            retry_delay: Delay between retries
            buffer_read_size: Buffer size for reading output
        """
        self.logger = logger
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.buffer_read_size = buffer_read_size

        self.process: Optional[subprocess.Popen] = None
        self.initialized = False

    @abstractmethod
    def build_command(self) -> List[str]:
        """Build the command to launch this process."""
        pass

    @abstractmethod
    def get_process_name(self) -> str:
        """Get the human-readable name of this process."""
        pass

    @abstractmethod
    def check_initialization_marker(self, message: str) -> bool:
        """Check if message indicates process initialization is complete."""
        pass

    def launch(self, log_file=None) -> int:
        """
        Launch the subprocess.

        Args:
            log_file: Optional file handle for stderr logging

        Returns:
            Process PID
        """
        cmd = self.build_command()

        # Set PYTHONUNBUFFERED to ensure output is not buffered
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Launch process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=log_file if log_file else subprocess.STDOUT,
            bufsize=0,  # Unbuffered
            env=env,
        )

        # Make stdout non-blocking
        if self.process.stdout:
            flags = fcntl.fcntl(self.process.stdout, fcntl.F_GETFL)
            fcntl.fcntl(self.process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        return self.process.pid

    def read_output(
        self,
        buffering_mode: bool,
        shm_connection_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Read and process output from the subprocess.

        Args:
            buffering_mode: If True, buffer messages; if False, print immediately
            shm_connection_callback: Optional callback for SHM connection detection
        """
        if not self.process or self.process.stdout is None:
            return

        # Use select to check if there's data to read (Unix-like systems)
        if hasattr(select, 'select'):
            ready, _, _ = select.select([self.process.stdout], [], [], 0)
            if not ready:
                return

        try:
            # Read available data (non-blocking)
            data = self.process.stdout.read(self.buffer_read_size)
            if not data:
                return

            lines = data.decode('utf-8').splitlines(keepends=False)

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Track shared memory connections
                if shm_connection_callback:
                    shm_connection_callback(stripped)

                # Check for initialization completion
                if self.check_initialization_marker(stripped):
                    self.initialized = True

                # Print to terminal
                if buffering_mode:
                    self.logger.log(stripped)
                else:
                    self.logger.print_immediate(stripped)

        except BlockingIOError:
            pass  # No data available
        except Exception:
            pass  # Silently ignore other errors

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 5.0, terminal=None):
        """
        Stop the process gracefully.

        Args:
            timeout: Timeout for graceful shutdown
            terminal: Terminal for status messages
        """
        if not self.process or self.process.poll() is not None:
            return

        process_name = self.get_process_name()

        if terminal:
            terminal.print(f"Stopping {process_name}...")

        self.process.terminate()

        try:
            self.process.wait(timeout=timeout)
            if terminal:
                terminal.print(f" {process_name} stopped")
        except subprocess.TimeoutExpired:
            if terminal:
                terminal.print(f"! {process_name} not responding, killing...")
            self.process.kill()
            self.process.wait()


class DetectionProcessManager(ProcessManager):
    """Manages the detection server subprocess."""

    def __init__(
        self,
        logger: OrderedLogger,
        method: str,
        config: Optional[str],
        gpu: Optional[int],
        image_shm_name: str,
        detection_shm_name: str,
        retry_count: int,
        retry_delay: float,
        verbose: bool,
        buffer_read_size: int = 4096,
    ):
        super().__init__(logger, retry_count, retry_delay, buffer_read_size)

        self.method = method
        self.config = config
        self.gpu = gpu
        self.image_shm_name = image_shm_name
        self.detection_shm_name = detection_shm_name
        self.verbose = verbose

    def build_command(self) -> List[str]:
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

    def get_process_name(self) -> str:
        return "detection server"

    def check_initialization_marker(self, message: str) -> bool:
        return "Detection Server Started" in message


class DecisionProcessManager(ProcessManager):
    """Manages the decision server subprocess."""

    def __init__(
        self,
        logger: OrderedLogger,
        config: Optional[str],
        detection_shm_name: str,
        control_shm_name: str,
        retry_count: int,
        retry_delay: float,
        verbose: bool,
        buffer_read_size: int = 4096,
    ):
        super().__init__(logger, retry_count, retry_delay, buffer_read_size)

        self.config = config
        self.detection_shm_name = detection_shm_name
        self.control_shm_name = control_shm_name
        self.verbose = verbose

    def build_command(self) -> List[str]:
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

    def get_process_name(self) -> str:
        return "decision server"

    def check_initialization_marker(self, message: str) -> bool:
        return "Decision Server Started" in message
