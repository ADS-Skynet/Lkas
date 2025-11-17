"""
Terminal utilities for structured logging with persistent footer.

Provides clean, organized output similar to installation progress displays:
- Main log area (scrolling content)
- Persistent footer line (status/progress)
"""

import sys
import threading
import time
from rich.console import Console
from rich.live import Live
from rich.table import Table


class TerminalDisplay:
    """
    Manages terminal display with a persistent footer line.

    This creates a two-section display:
    1. Main content area (scrolls normally)
    2. Footer line (stays at bottom, updates in-place)

    Similar to how npm/apt show progress during installation.
    """

    def __init__(self, enable_footer: bool = True):
        """
        Initialize terminal display.

        Args:
            enable_footer: Whether to enable the persistent footer
        """
        self.enable_footer = enable_footer
        self.footer_text = ""
        self.lock = threading.Lock()
        self._has_footer = False

        # Rich console and live display
        self.console = Console()
        self.live_display: Live | None = None

        # Shared memory status tracking
        self.shm_status = {}  # {name: connected_bool}

        # FPS tracking
        self.fps_stats = {}  # {name: fps_value}

    def print(self, message: str, prefix: str = "", carriage_return: bool = False):
        """
        Print a message to the main content area.

        Args:
            message: Message to print
            prefix: Optional prefix (e.g., "[DETECTION]")
        """
        # Just use regular print - Rich Live handles the footer
        if prefix:
            # if carriage_return:
            #     print(f"\r{prefix} {message}", end="", flush=True)
            # else:
            print(f"{prefix} {message}")
        else:
            # if carriage_return:
            #     print(f"\r{message}", end="", flush=True)
            # else:
            print(message)

    def init_footer(self):
        """Initialize Rich live footer display."""
        if not self.enable_footer or self.live_display is not None:
            return

        with self.lock:
            self.live_display = Live(
                self._generate_footer_table(),
                console=self.console,
                refresh_per_second=2,
                vertical_overflow="visible"
            )
            self.live_display.start()

    def _generate_footer_table(self) -> Table:
        """Generate footer table showing shared memory status and FPS."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="magenta", no_wrap=True)
        table.add_column(style="yellow", no_wrap=True)

        # Build status displays for each shared memory
        status_items = []
        for shm_name, connected in sorted(self.shm_status.items()):
            status = (
                "[bold green]● CONNECTED[/bold green]" if connected
                else "[bold dim]○ WAITING[/bold dim]"
            )
            status_items.append(f"[bold]{shm_name}:[/bold] {status}")

        # Add shared memory status row (up to 3 columns)
        if len(status_items) == 0:
            table.add_row("[dim]No shared memory configured[/dim]")
        elif len(status_items) == 1:
            table.add_row(status_items[0])
        elif len(status_items) == 2:
            table.add_row(status_items[0], status_items[1])
        else:
            table.add_row(status_items[0], status_items[1], status_items[2] if len(status_items) > 2 else "")

        # Add FPS stats on a separate row if available
        if self.fps_stats:
            fps_items = []
            for fps_name, fps_value in sorted(self.fps_stats.items()):
                fps_items.append(f"[bold]{fps_name}:[/bold] [bold cyan]{fps_value:.1f} FPS[/bold cyan]")

            if len(fps_items) == 1:
                table.add_row(fps_items[0])
            elif len(fps_items) == 2:
                table.add_row(fps_items[0], fps_items[1])
            else:
                table.add_row(fps_items[0], fps_items[1], fps_items[2] if len(fps_items) > 2 else "")

        return table

    def update_footer(self, shm_status: dict | None = None, fps_stats: dict | None = None):
        """
        Update the persistent footer line with shared memory status and FPS.

        Args:
            shm_status: Dictionary mapping shared memory names to connection status
            fps_stats: Dictionary mapping stat names to FPS values
        """

        if not self.enable_footer:
            return

        with self.lock:
            # Update shared memory status
            if shm_status is not None:
                self.shm_status.update(shm_status)

            # Update FPS stats
            if fps_stats is not None:
                self.fps_stats.update(fps_stats)

            if self.live_display is not None:
                self.live_display.update(self._generate_footer_table())

    def clear_footer(self):
        """Clear the footer line."""
        if not self.enable_footer:
            return

        with self.lock:
            if self.live_display is not None:
                try:
                    self.live_display.stop()
                except Exception:
                    pass
                finally:
                    self.live_display = None
                    print()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.clear_footer()


class OrderedLogger:
    """
    Logger that buffers messages until explicitly flushed.

    Useful for ensuring initialization messages from concurrent
    processes appear in order instead of being interleaved.
    """

    def __init__(self, prefix: str, terminal: TerminalDisplay | None = None):
        """
        Initialize ordered logger.

        Args:
            prefix: Prefix for all messages
            terminal: Optional terminal display to use
        """
        self.prefix = prefix
        self.terminal = terminal
        self.buffer = []
        self.lock = threading.Lock()

    def log(self, message: str):
        """
        Add a message to the buffer.

        Args:
            message: Message to buffer
        """
        with self.lock:
            self.buffer.append(message)

    def flush(self):
        """Flush all buffered messages to output."""
        with self.lock:
            for msg in self.buffer:
                if self.terminal:
                    self.terminal.print(msg, self.prefix)
                else:
                    print(f"{self.prefix} {msg}")
            self.buffer.clear()

    def print_immediate(self, message: str, carriage_return: bool = False):
        """
        Print a message immediately without buffering.

        Args:
            message: Message to print
        """
        if self.terminal:
            self.terminal.print(message, self.prefix, carriage_return)
        else:
            print(f"{self.prefix} {message}")


def create_progress_bar(current: int, total: int, width: int = 30) -> str:
    """
    Create a progress bar string.

    Args:
        current: Current progress value
        total: Total/max value
        width: Width of the progress bar in characters

    Returns:
        Progress bar string like: [===========>          ] 50%
    """
    if total == 0:
        percent = 0
    else:
        percent = min(100, int(100 * current / total))

    filled = int(width * current / total) if total > 0 else 0
    filled = min(filled, width)

    bar = "=" * filled + ">" if filled < width else "=" * width
    bar = bar.ljust(width)

    return f"[{bar}] {percent:3d}%"


def format_fps_stats(
    fps: float,
    frame_id: int,
    processing_time_ms: float,
    extra_info: str = ""
) -> str:
    """
    Format FPS and processing stats for footer display.

    Args:
        fps: Frames per second
        frame_id: Current frame ID
        processing_time_ms: Processing time in milliseconds
        extra_info: Additional information to display

    Returns:
        Formatted stats string
    """
    base = f"FPS: {fps:5.1f} | Frame: {frame_id:6d} | Time: {processing_time_ms:6.2f}ms"
    if extra_info:
        base += f" | {extra_info}"
    return base
