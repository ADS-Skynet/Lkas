# LKAS Terminal Display Utilities

This module provides structured terminal output with a persistent footer line, similar to modern installation progress displays.

## Features

### 1. **Two-Section Display**
- **Main Content Area**: Scrolling log messages from both Detection and Decision servers
- **Persistent Footer**: Live stats that stay at the bottom (FPS, frame count, processing time)

### 2. **Ordered Logging**
- Initialization messages from concurrent processes are buffered and displayed in order
- Prevents interleaved output during startup

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

## How It Works

### ANSI Escape Codes
The terminal display uses ANSI escape sequences to:
- Clear lines: `\033[2K`
- Move cursor: `\033[1A` (up), `\033[1B` (down)
- Update in-place without scrolling

### Thread Safety
All terminal operations are protected by locks to prevent race conditions when multiple processes write simultaneously.

### Footer Management
1. When printing a message, the footer is temporarily cleared
2. The message is printed on a new line
3. The footer is immediately restored at the bottom

## Demo

Run the demo script to see the terminal display in action:

```bash
python test_terminal_display.py
```

This demonstrates:
- Ordered initialization messages
- Scrolling log messages
- Live footer updates
- How messages and footer coexist

## Example Output

```
======================================================================
                    LKAS System Launcher
======================================================================
[LKAS] Starting detection server...
[DETECTION] Loading configuration...
[DETECTION] ✓ Configuration loaded
[DETECTION] Initializing CV detector...
[DETECTION] ✓ Detector ready: CVLaneDetector
[LKAS] Starting decision server...
[DECISION ] Loading configuration...
[DECISION ] ✓ Configuration loaded
[DECISION ] ✓ Decision controller ready
======================================================================
[LKAS] System running - Press Ctrl+C to stop
======================================================================

[DET] 30.2 FPS | Frame 1234 | Processing: 15.20ms | Lanes: L=True R=True | [DEC] 30.5 FPS | Frame 1234 | Decision: 2.10ms | Steering: +0.125 | Throttle: 0.650
```

The last line (footer) updates continuously without scrolling.

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

The LKAS launcher (`src/lkas/run.py`) automatically uses this terminal display:

```python
# Initialize
self.terminal = TerminalDisplay(enable_footer=True)
self.detection_logger = OrderedLogger("[DETECTION]", self.terminal)
self.decision_logger = OrderedLogger("[DECISION ]", self.terminal)

# During startup: buffer messages for ordering
self._read_process_output(self.detection_process, "DETECTION", self.detection_logger)
self.detection_logger.flush()

# During runtime: update footer with stats
self.terminal.update_footer(f"[DET] {det_stats} | [DEC] {dec_stats}")
```

## Advanced Usage

### Custom Context Manager

```python
with TerminalDisplay(enable_footer=True) as terminal:
    terminal.print("Starting...")
    terminal.update_footer("Progress: 50%")
    # Footer automatically cleared on exit
```

### Disable Footer

For logging to files or non-interactive terminals:

```python
terminal = TerminalDisplay(enable_footer=False)
terminal.print("Log message")  # No footer management
terminal.update_footer("...")   # No-op when disabled
```

## Troubleshooting

### Footer Not Showing
- Ensure your terminal supports ANSI escape codes
- Check that `enable_footer=True`
- Verify terminal width is sufficient

### Garbled Output
- May occur in non-ANSI terminals (e.g., old Windows cmd)
- Set `enable_footer=False` for compatibility

### Messages Out of Order
- Use `OrderedLogger` and buffer messages during concurrent initialization
- Call `flush()` in the desired order
