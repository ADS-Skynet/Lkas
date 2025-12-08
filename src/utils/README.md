# Utils Module

**Terminal display and utility functions for LKAS.**

## Overview

The Utils module provides terminal display utilities for structured, professional output with live status updates. It features a persistent footer line for real-time stats and ordered logging from concurrent processes, similar to modern CLI tools like npm, cargo, and pip.

## Features

### 1. Two-Section Terminal Display
- **Main Content Area**: Scrolling log messages from concurrent processes
- **Persistent Footer**: Live stats that stay at the bottom (FPS, latency, frame count)
- **ANSI-based**: Uses escape codes for in-place updates

### 2. Ordered Logging
- **Message Buffering**: Concurrent processes buffer startup messages
- **Sequential Flushing**: Display messages in logical order (Detection → Decision)
- **Race-Condition Free**: Prevents interleaved output during concurrent initialization

### 3. Thread-Safe Operations
- **Synchronized Updates**: All terminal operations protected by locks
- **Multi-Process Safe**: Works with concurrent Detection and Decision servers

### 4. Formatting Utilities
- **FPS Statistics**: Formatted performance metrics
- **Progress Bars**: ASCII progress indicators
- **Prefix Support**: Categorized logging with prefixes like `[DETECTION]`, `[DECISION]`

## Usage

### Basic Terminal Display

```python
from lkas.utils.terminal import TerminalDisplay

# Create terminal display with footer enabled
terminal = TerminalDisplay(enable_footer=True)

# Print normal messages (scroll in main area)
terminal.print("Server starting...")
terminal.print("Configuration loaded", prefix="[DETECTION]")

# Update persistent footer (stays at bottom)
terminal.update_footer("FPS: 30.0 | Frame: 1234 | Processing: 15.2ms")

# Clear footer when done
terminal.clear_footer()
```

### Ordered Logging for Concurrent Processes

```python
from lkas.utils.terminal import TerminalDisplay, OrderedLogger

terminal = TerminalDisplay(enable_footer=True)

# Create loggers for each process
detection_logger = OrderedLogger("[DETECTION]", terminal)
decision_logger = OrderedLogger("[DECISION ]", terminal)

# Buffer messages during initialization
detection_logger.log("Loading model...")
detection_logger.log("✓ Model loaded")

decision_logger.log("Connecting to detection...")
decision_logger.log("✓ Connected")

# Flush in order (detection first, then decision)
detection_logger.flush()
decision_logger.flush()

# Or print immediately without buffering
detection_logger.print_immediate("Processing frame 100")
```

### Formatting Utilities

```python
from lkas.utils.terminal import format_fps_stats, create_progress_bar

# Format FPS statistics
stats = format_fps_stats(
    fps=30.5,
    frame_id=1234,
    processing_time_ms=15.2,
    extra_info="Lanes: L=True R=True"
)
# Output: "FPS:  30.5 | Frame:   1234 | Time:  15.20ms | Lanes: L=True R=True"

# Create progress bar
bar = create_progress_bar(current=50, total=100, width=30)
# Output: "[===============>              ] 50%"
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TerminalDisplay                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────┐               │
│  │   Main Content Area (Scrolling)     │               │
│  │                                      │               │
│  │  [DETECTION] Loading model...        │               │
│  │  [DETECTION] ✓ Model loaded          │               │
│  │  [DECISION ] Connecting...           │               │
│  │  [DECISION ] ✓ Connected             │               │
│  │  [LKAS] System running               │               │
│  │                                      │               │
│  └─────────────────────────────────────┘               │
│  ┌─────────────────────────────────────┐               │
│  │  Persistent Footer (Fixed Bottom)   │               │
│  │                                      │               │
│  │  FPS: 30.2 | Frame: 1234 | ...      │  ← Updates    │
│  └─────────────────────────────────────┘     in-place  │
└─────────────────────────────────────────────────────────┘
```

## How It Works

### ANSI Escape Codes

The terminal display uses ANSI escape sequences for cursor control:

| Code | Purpose |
|------|---------|
| `\033[2K` | Clear current line |
| `\033[1A` | Move cursor up one line |
| `\033[1B` | Move cursor down one line |
| `\r` | Return to beginning of line |

### Thread Safety

- **Lock Protection**: All terminal operations use `threading.Lock`
- **Atomic Operations**: Print and footer update are indivisible
- **Multi-Process Safe**: Works with concurrent subprocess output

### Footer Management

1. **Print Message**:
   - Clear footer (if active)
   - Print message on new line
   - Restore footer at bottom

2. **Update Footer**:
   - Move cursor to footer line
   - Clear line
   - Write new footer content
   - Restore cursor position

## API Reference

### TerminalDisplay

Main terminal display manager.

```python
class TerminalDisplay:
    def __init__(self, enable_footer: bool = True):
        """
        Initialize terminal display.

        Args:
            enable_footer: Enable persistent footer (default: True)
        """

    def print(self, message: str, prefix: str = ""):
        """
        Print a message to the main content area.

        Args:
            message: Message to print
            prefix: Optional prefix (e.g., "[DETECTION]")
        """

    def update_footer(self, footer_text: str):
        """
        Update persistent footer content.

        Args:
            footer_text: New footer text (stays at bottom)
        """

    def clear_footer(self):
        """Clear the persistent footer."""

    def __enter__(self) -> 'TerminalDisplay':
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit (auto-clear footer)."""
        self.clear_footer()
```

### OrderedLogger

Buffered logger for ordered output from concurrent processes.

```python
class OrderedLogger:
    def __init__(self, prefix: str, terminal: TerminalDisplay):
        """
        Initialize ordered logger.

        Args:
            prefix: Log prefix (e.g., "[DETECTION]")
            terminal: TerminalDisplay instance
        """

    def log(self, message: str):
        """
        Buffer a message for later ordered display.

        Args:
            message: Message to buffer
        """

    def flush(self):
        """Flush all buffered messages in order."""

    def print_immediate(self, message: str):
        """
        Print message immediately without buffering.

        Args:
            message: Message to print
        """
```

### Formatting Utilities

```python
def format_fps_stats(
    fps: float,
    frame_id: int,
    processing_time_ms: float,
    extra_info: str = ""
) -> str:
    """
    Format FPS statistics for footer display.

    Returns:
        Formatted string: "FPS: 30.5 | Frame: 1234 | Time: 15.20ms | ..."
    """

def create_progress_bar(
    current: int,
    total: int,
    width: int = 30
) -> str:
    """
    Create ASCII progress bar.

    Returns:
        Progress bar: "[===============>              ] 50%"
    """
```

## Example Output

```
======================================================================
                    LKAS System Launcher
======================================================================
[LKAS      ] Starting detection server...
[DETECTION ] Loading configuration...
[DETECTION ] ✓ Configuration loaded
[DETECTION ] Initializing CV detector...
[DETECTION ] ✓ Detector ready: CVLaneDetector
[LKAS      ] Starting decision server...
[DECISION  ] Loading configuration...
[DECISION  ] ✓ Configuration loaded
[DECISION  ] ✓ Controller ready: PDController
======================================================================
[LKAS      ] System running - Press Ctrl+C to stop
======================================================================
[DETECTION ] Processing frame 1234
[DECISION  ] Computing steering for frame 1234

FPS: 30.2 | Frame: 1234 | DET: 8.5ms | DEC: 1.2ms | Steering: +0.125
```

The last line (footer) updates continuously in-place without scrolling.

## Benefits

### For Users
- **Clear Status**: Always see current frame processing stats
- **Organized Logs**: Initialization messages appear in logical order
- **Professional Look**: Similar to npm, pip, cargo, etc.

### For Developers
- **Easy Integration**: Simple API for printing and updating footer
- **Thread-Safe**: Works with concurrent processes
- **Flexible**: Can enable/disable footer as needed

## Integration with LKAS

The LKAS launcher ([src/run.py](../../run.py)) uses TerminalDisplay for professional output:

```python
from lkas.utils.terminal import TerminalDisplay, OrderedLogger

class LKASServer:
    def __init__(self):
        # Initialize terminal display
        self.terminal = TerminalDisplay(enable_footer=True)

        # Create ordered loggers for each subprocess
        self.detection_logger = OrderedLogger("[DETECTION]", self.terminal)
        self.decision_logger = OrderedLogger("[DECISION ]", self.terminal)

    def start(self):
        # Buffer startup messages
        self.detection_logger.log("Loading configuration...")
        self.detection_logger.log("✓ Configuration loaded")
        self.detection_logger.flush()  # Display in order

        self.decision_logger.log("Connecting to detection...")
        self.decision_logger.log("✓ Connected")
        self.decision_logger.flush()

    def run(self):
        while self.running:
            # Update footer with live stats
            footer = format_fps_stats(
                fps=self.fps,
                frame_id=self.frame_id,
                processing_time_ms=self.processing_time,
                extra_info=f"Steering: {self.steering:+.3f}"
            )
            self.terminal.update_footer(footer)

            # Log events to main area
            self.terminal.print("Processing frame", prefix="[DETECTION]")
```

## Advanced Usage

### Context Manager

Automatically clear footer on exit:

```python
from lkas.utils.terminal import TerminalDisplay

with TerminalDisplay(enable_footer=True) as terminal:
    terminal.print("Starting task...")
    terminal.update_footer("Progress: 50%")
    # Do work...
    terminal.update_footer("Progress: 100%")
# Footer automatically cleared on exit
```

### Disable Footer (Non-Interactive Mode)

For logging to files or non-interactive terminals:

```python
import sys

# Detect if running in interactive terminal
is_interactive = sys.stdout.isatty()

terminal = TerminalDisplay(enable_footer=is_interactive)
terminal.print("Log message")  # Always works
terminal.update_footer("...")   # No-op when footer disabled
```

### Custom Formatting

```python
from lkas.utils.terminal import format_fps_stats, create_progress_bar

# Custom footer with progress bar
progress = create_progress_bar(current=50, total=100, width=30)
stats = format_fps_stats(fps=30.5, frame_id=1234, processing_time_ms=15.2)
footer = f"{stats} | {progress}"
terminal.update_footer(footer)
```

## Implementation Details

### Terminal Compatibility

**Supported:**
- Linux terminals (bash, zsh, etc.)
- macOS Terminal.app, iTerm2
- Windows Terminal, PowerShell (Windows 10+)
- VS Code integrated terminal

**Not Supported:**
- Old Windows CMD (pre-Windows 10)
- File output / non-interactive streams
- Terminals without ANSI support

**Auto-Detection:**
```python
import sys

# Automatically detect terminal support
if not sys.stdout.isatty():
    # Disable footer for non-interactive output
    terminal = TerminalDisplay(enable_footer=False)
```

### Performance

- **Footer Update:** <0.1ms (ANSI escape sequence write)
- **Message Print:** <0.5ms (lock acquisition + output)
- **Memory:** Minimal (only buffers messages during flush)
- **Overhead:** Negligible compared to LKAS processing (10-50ms)

### Thread Safety Details

```python
import threading

class TerminalDisplay:
    def __init__(self):
        self._lock = threading.Lock()  # Protect all operations

    def print(self, message: str):
        with self._lock:  # Atomic operation
            # Clear footer
            # Print message
            # Restore footer
```

## Troubleshooting

### Footer Not Showing

**Symptom:** Footer text doesn't appear or disappears immediately.

**Solutions:**
- **Check ANSI support:** Run `echo $TERM` - should show `xterm`, `xterm-256color`, etc.
- **Verify footer enabled:** `TerminalDisplay(enable_footer=True)`
- **Check terminal width:** Footer may be too long for narrow terminals

### Garbled Output / Escape Codes Visible

**Symptom:** Output shows `^[[2K`, `^[[1A`, etc. instead of formatted display.

**Solutions:**
- **Old terminal:** Use `enable_footer=False` for non-ANSI terminals
- **Windows CMD:** Use Windows Terminal or PowerShell instead
- **Redirected output:** Footer disabled automatically when stdout is not a tty

### Messages Out of Order

**Symptom:** Initialization messages from different processes are interleaved.

**Solutions:**
```python
# Bad: Messages print immediately and interleave
detection_logger.print_immediate("Loading...")  # Prints now
decision_logger.print_immediate("Starting...")   # Prints now

# Good: Messages buffered and flushed in order
detection_logger.log("Loading...")   # Buffered
detection_logger.log("✓ Loaded")     # Buffered
detection_logger.flush()             # Print all in order

decision_logger.log("Starting...")
decision_logger.log("✓ Started")
decision_logger.flush()
```

### Footer Updates Too Slow

**Symptom:** Footer lags behind actual stats.

**Solutions:**
- **Reduce update frequency:** Update every N frames instead of every frame
- **Throttle updates:** Only update if >16ms elapsed (60 FPS limit)
```python
import time

last_update = 0
UPDATE_INTERVAL = 0.016  # 60 Hz max

def maybe_update_footer(stats):
    global last_update
    now = time.time()
    if now - last_update >= UPDATE_INTERVAL:
        terminal.update_footer(stats)
        last_update = now
```

## Related Modules

- **lkas/**: Uses TerminalDisplay for LKAS server output
- **detection/**: Server logs via OrderedLogger
- **decision/**: Server logs via OrderedLogger

## Dependencies

- **Python Standard Library:** `threading`, `sys`, `io`
- **No external dependencies**

## Version

Current version: 0.1.0 (see [\_\_init\_\_.py](__init__.py))
