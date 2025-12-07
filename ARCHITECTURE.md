# Terminal Display Architecture

## Visual Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TERMINAL WINDOW                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [LKAS] Starting detection server...                                │
│  [DETECTION] Loading configuration...                               │
│  [DETECTION] ✓ Configuration loaded                                 │
│  [DETECTION] ✓ Detector ready                                       │
│                                                                       │
│  [LKAS] Starting decision server...                                 │
│  [DECISION ] Loading configuration...                               │
│  [DECISION ] ✓ Configuration loaded                                 │
│  [DECISION ] ✓ Controller ready                                     │
│                                                                       │
│  [LKAS] System running - Press Ctrl+C to stop                       │
│  ═════════════════════════════════════════════════════════════      │
│                                                                       │
│  [LKAS] Checkpoint: Processed 150 frames                            │
│  [DETECTION] Model confidence: 0.96                                 │
│                                                                       │
│                    ▲                                                 │
│                    │                                                 │
│              SCROLLING AREA                                          │
│         (Main log messages)                                          │
│                    │                                                 │
│                    ▼                                                 │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  [DET] 30.2 FPS | Frame 1234 | 15.20ms | [DEC] 30.5 FPS | +0.125   │  ◄── PERSISTENT FOOTER
└─────────────────────────────────────────────────────────────────────┘   (Always at bottom)
                                                                           (Updates in-place)
```

## Data Flow

```
┌─────────────────┐         ┌─────────────────┐
│   Detection     │         │    Decision     │
│   Process       │         │    Process      │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ stdout                    │ stdout
         │                           │
         ▼                           ▼
    ┌────────────────────────────────────┐
    │      LKASLauncher.run()            │
    │                                    │
    │  _read_process_output()            │
    │         │                          │
    │         ├── Regular messages ─────►│
    │         │   (scroll)               │
    │         │                          │
    │         └── FPS stats ─────►       │
    │             (footer)               │
    │                                    │
    │  ┌──────────────────────────────┐ │
    │  │    TerminalDisplay           │ │
    │  │                              │ │
    │  │  • print() → scrolling      │ │
    │  │  • update_footer() → bottom │ │
    │  └──────────────────────────────┘ │
    └────────────────────────────────────┘
```

## Initialization Flow (Ordered)

```
Time ───────────────────────────────────────────────►

LKAS Launcher:
  │
  ├─► Start Detection Process
  │
  ├─► Buffer Detection Output (5 sec)
  │   ┌────────────────────────┐
  │   │ OrderedLogger (DET)    │
  │   │  • log("Loading...")   │
  │   │  • log("✓ Loaded")     │
  │   │  • log("✓ Ready")      │
  │   └────────────────────────┘
  │
  ├─► Flush Detection Buffer
  │       ↓
  │   [DETECTION] Loading...
  │   [DETECTION] ✓ Loaded
  │   [DETECTION] ✓ Ready
  │
  ├─► Start Decision Process
  │
  ├─► Buffer Decision Output (3 sec)
  │   ┌────────────────────────┐
  │   │ OrderedLogger (DEC)    │
  │   │  • log("Loading...")   │
  │   │  • log("✓ Loaded")     │
  │   │  • log("✓ Ready")      │
  │   └────────────────────────┘
  │
  ├─► Flush Decision Buffer
  │       ↓
  │   [DECISION ] Loading...
  │   [DECISION ] ✓ Loaded
  │   [DECISION ] ✓ Ready
  │
  └─► Start Main Loop
```

## Runtime Flow (Live Stats)

```
Detection Process                 LKAS Launcher                  Terminal
─────────────────                ─────────────                  ─────────

Frame processing...
   │
   ├─► Print FPS stats ──────►  Read stdout
                                     │
                                     ├─ Detect "FPS" in line
                                     │
                                     ├─ Update last_detection_stats
                                     │
                                     └─ _update_footer()
                                            │
                                            ├─ Combine DET + DEC stats
                                            │
                                            └─► terminal.update_footer() ──►  Clear current footer
                                                                              Print new footer
                                                                              Stay at bottom


Decision Process
─────────────────

Frame processing...
   │
   ├─► Print FPS stats ──────►  Read stdout
                                     │
                                     ├─ Detect "FPS" in line
                                     │
                                     ├─ Update last_decision_stats
                                     │
                                     └─ _update_footer()
                                            │
                                            └─► terminal.update_footer() ──►  Update footer
```

## Component Hierarchy

```
LKASLauncher
│
├── TerminalDisplay (instance)
│   ├── print(message, prefix)       # Scrolling messages
│   ├── update_footer(text)           # Persistent footer
│   ├── clear_footer()                # Remove footer
│   └── lock (threading.Lock)         # Thread safety
│
├── OrderedLogger (detection_logger)
│   ├── buffer: List[str]             # Message buffer
│   ├── log(message)                  # Add to buffer
│   ├── flush()                       # Print all buffered
│   └── print_immediate(message)      # Bypass buffer
│
├── OrderedLogger (decision_logger)
│   └── (same as above)
│
├── last_detection_stats: str         # Latest detection stats
├── last_decision_stats: str          # Latest decision stats
│
├── detection_process: Popen          # Detection subprocess
└── decision_process: Popen           # Decision subprocess
```

## Message Classification

```
Process Output Line
       │
       ├─── Contains "\r" or "FPS"?
       │
       ├─ YES ──► Stats Line
       │           │
       │           ├─ Extract stats
       │           ├─ Update last_*_stats
       │           └─ Call _update_footer()
       │                 │
       │                 └─ Combine DET + DEC
       │                    │
       │                    └─ terminal.update_footer()
       │
       └─ NO ──► Regular Message
                   │
                   └─ logger.print_immediate()
                        │
                        └─ terminal.print()
```

## ANSI Escape Sequence Usage

```
Operation                    ANSI Code          Effect
─────────────────────────   ─────────────      ─────────────────────
Clear current line          \033[2K\r          Erase line, return to start
Move cursor up              \033[1A            Move up 1 line
Move cursor down            \033[1B            Move down 1 line
Save cursor position        \033[s             Remember position
Restore cursor position     \033[u             Go back to saved position
Hide cursor                 \033[?25l          Invisible cursor
Show cursor                 \033[?25h          Visible cursor
```

## Threading Model

```
Main Thread (LKASLauncher.run())
     │
     ├─ Spawn Detection Process
     │     │
     │     └─ stdout pipe
     │
     ├─ Spawn Decision Process
     │     │
     │     └─ stdout pipe
     │
     └─ Main Loop:
         │
         ├─► _read_process_output(detection, ...) ──┐
         │                                           │
         ├─► _read_process_output(decision, ...)  ──┤
         │                                           │
         │                    ┌────────────────────►│
         │                    │                      │
         │                    │  Terminal Lock       │
         │                    │  ───────────────     │
         │                    │  Prevents race       │
         │                    │  conditions when     │
         │                    │  both processes      │
         │                    │  write at same time  │
         │                    │                      │
         └────────────────────┘◄─────────────────────┘
```

## Example State Transitions

### Startup Phase
```
State 1: INIT
  ├─ terminal = TerminalDisplay()
  ├─ detection_logger = OrderedLogger()
  └─ decision_logger = OrderedLogger()

State 2: DETECTION_STARTING
  ├─ Start detection process
  ├─ Buffer detection messages
  └─ Flush detection buffer

State 3: DECISION_STARTING
  ├─ Start decision process
  ├─ Buffer decision messages
  └─ Flush decision buffer

State 4: RUNNING
  ├─ Read both process outputs
  ├─ Classify: stats vs regular
  └─ Update footer or print message
```

### Footer Update Sequence
```
1. New stats line from Detection
   └─► last_detection_stats = "30.2 FPS | Frame 1234..."

2. Call _update_footer()
   ├─ parts = []
   ├─ parts.append(f"[DET] {last_detection_stats}")
   ├─ parts.append(f"[DEC] {last_decision_stats}")
   └─ footer_text = " | ".join(parts)

3. terminal.update_footer(footer_text)
   ├─ Acquire lock
   ├─ Write: \033[2K\r (clear current footer)
   ├─ Write: footer_text
   ├─ Flush stdout
   └─ Release lock

4. Result:
   [DET] 30.2 FPS | Frame 1234 | ... | [DEC] 30.5 FPS | ...
```

## Performance Characteristics

```
Operation                  Complexity    Notes
────────────────────────  ───────────   ─────────────────────
print() with footer       O(1)          Clear, print, restore
print() without footer    O(1)          Direct stdout.write()
update_footer()           O(1)          Clear + write
Buffered logging          O(n)          n = number of messages
Lock acquisition          O(1)          Fast uncontended
```

## Design Patterns Used

1. **Facade Pattern**
   - `TerminalDisplay` hides ANSI complexity
   - Simple API: print(), update_footer()

2. **Buffering Pattern**
   - `OrderedLogger` buffers messages
   - Flush when ready for ordering

3. **Observer Pattern**
   - LKASLauncher observes subprocess output
   - Routes to appropriate handler

4. **Thread Safety**
   - All terminal operations protected by locks
   - Prevents interleaved output

5. **Separation of Concerns**
   - Display logic in terminal.py
   - Business logic in run.py
   - Clean boundaries
