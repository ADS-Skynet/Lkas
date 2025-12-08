# LKAS Server Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        LKASServer                           │
│                     (Orchestrator)                          │
│                                                             │
│  Responsibilities:                                          │
│  • Configuration management                                 │
│  • Lifecycle orchestration                                  │
│  • Signal handling                                          │
│  • Terminal display coordination                            │
└────┬──────────────────────┬──────────────────────┬──────────┘
     │                      │                      │
     │                      │                      │
     ▼                      ▼                      ▼
┌──────────┐      ┌──────────────┐      ┌─────────────────┐
│Detection │      │  Decision    │      │   Broadcast     │
│Process   │      │  Process     │      │   Manager       │
│Manager   │      │  Manager     │      │   (Optional)    │
└────┬─────┘      └──────┬───────┘      └───────┬─────────┘
     │                   │                      │
     │extends            │extends               │uses
     │                   │                      │
     ▼                   ▼                      │
┌────────────────────────────────────┐          │
│      ProcessManager (ABC)          │          │
│                                    │          │
│  Methods:                          │          │
│  • launch()                        │          │
│  • read_output()                   │          │
│  • is_alive()                      │          │
│  • stop()                          │          │
│                                    │          │
│  Abstract:                         │          │
│  • build_command()                 │          │
│  • get_process_name()              │          │
│  • check_initialization_marker()   │          │
└────────────────────────────────────┘          │
                                                │
                                                ▼
                                    ┌────────────────────────┐
                                    │  ZMQ Broker            │
                                    │  Shared Memory         │
                                    │  - Image Channel       │
                                    │  - Detection Channel   │
                                    │  - Control Channel     │
                                    └────────────────────────┘
```

## Data Flow

```
┌──────────────┐
│ Simulation   │
│  (External)  │
└──────┬───────┘
       │ camera_feed
       │ (SharedMemory)
       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Detection Server                        │
│                  (Subprocess via Manager)                    │
│                                                              │
│  DetectionProcessManager launches:                           │
│  python -m lkas.detection.run --method cv ...                │
└──────────────────────────────┬──────────────────────────────┘
                               │ detection_results
                               │ (SharedMemory)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Decision Server                         │
│                  (Subprocess via Manager)                    │
│                                                              │
│  DecisionProcessManager launches:                            │
│  python -m lkas.decision.run ...                             │
└──────────────────────────────┬──────────────────────────────┘
                               │ control_commands
                               │ (SharedMemory)
                               ▼
                        ┌──────────────┐
                        │  Simulation  │
                        │  (External)  │
                        └──────────────┘

Optional Broadcasting:
─────────────────────
BroadcastManager reads from SharedMemory:
  • camera_feed → broadcasts frames via ZMQ
  • detection_results + control_commands → broadcasts detection + metrics

  Viewers (remote) connect to ZMQ and receive:
  • Live camera frames (JPEG or raw RGB)
  • Lane detection results
  • Control metrics (offset, heading, etc.)
```

## Sequence Diagram: Server Startup

```
User           LKASServer       ProcessManagers     Subprocesses    BroadcastMgr
 │                  │                  │                 │                │
 │  run()           │                  │                 │                │
 ├─────────────────>│                  │                 │                │
 │                  │                  │                 │                │
 │                  │ setup()          │                 │                │
 │                  ├─────────────────────────────────────────────────────>│
 │                  │                  │                 │                │
 │                  │ launch()         │                 │                │
 │                  ├─────────────────>│                 │                │
 │                  │                  │ subprocess.Popen│                │
 │                  │                  ├────────────────>│                │
 │                  │                  │                 │                │
 │                  │ wait_for_init()  │                 │                │
 │                  ├─────────────────>│                 │                │
 │                  │                  │ read_output()   │                │
 │                  │                  ├────────────────>│                │
 │                  │                  │<────────────────┤                │
 │                  │                  │ "Server Started"│                │
 │                  │                  │                 │                │
 │                  │ announce_ready() │                 │                │
 │                  ├─────────────────────────────────────────────────────>│
 │                  │                  │                 │                │
 │                  │ main_loop()      │                 │                │
 │                  │ ┌────────────────┴─────────────────┴───────────────┐│
 │                  │ │  Loop:                                            ││
 │                  │ │    • poll_broker()                                ││
 │                  │ │    • broadcast_data()                             ││
 │                  │ │    • read_output() from processes                 ││
 │                  │ │    • check is_alive()                             ││
 │                  │ └────────────────┬─────────────────┬───────────────┘│
 │                  │                  │                 │                │
 │  Ctrl+C          │                  │                 │                │
 ├─────────────────>│                  │                 │                │
 │                  │                  │                 │                │
 │                  │ stop()           │                 │                │
 │                  ├─────────────────>│                 │                │
 │                  │                  │ terminate()     │                │
 │                  │                  ├────────────────>│                │
 │                  │                  │ wait()          │                │
 │                  │                  ├────────────────>│                │
 │                  │                  │                 │                │
 │                  │ close()          │                 │                │
 │                  ├─────────────────────────────────────────────────────>│
 │                  │                  │                 │                │
 │<─────────────────┤                  │                 │                │
 │  exit(0)         │                  │                 │                │
```

## Class Responsibilities

### ProcessManager (Abstract Base)
```python
class ProcessManager:
    """Base class for all subprocess management."""

    # Concrete methods (implemented):
    def launch(log_file) -> int
    def read_output(buffering_mode, shm_callback)
    def is_alive() -> bool
    def stop(timeout, terminal)

    # Abstract methods (must implement):
    @abstractmethod
    def build_command() -> List[str]
    @abstractmethod
    def get_process_name() -> str
    @abstractmethod
    def check_initialization_marker(msg) -> bool
```

### DetectionProcessManager
```python
class DetectionProcessManager(ProcessManager):
    """Manages detection server subprocess."""

    # Implementation specifics:
    • Builds: python -m lkas.detection.run --method {cv|dl} ...
    • Monitors: "Detection Server Started"
    • Manages: GPU, image/detection SHM, retry config
```

### DecisionProcessManager
```python
class DecisionProcessManager(ProcessManager):
    """Manages decision server subprocess."""

    # Implementation specifics:
    • Builds: python -m lkas.decision.run ...
    • Monitors: "Decision Server Started"
    • Manages: Detection/control SHM, retry config
```

### BroadcastManager
```python
class BroadcastManager:
    """Manages ZMQ broadcasting and SHM reading."""

    # Key responsibilities:
    • setup() - Initialize ZMQ broker
    • broadcast_data() - Read SHM, broadcast via ZMQ
    • poll_broker() - Handle parameter updates
    • close() - Cleanup resources

    # Lazy connections:
    • Image channel (camera feed)
    • Detection channel (lane results)
    • Control channel (steering, metrics)
```

### LKASServer
```python
class LKASServer:
    """Main orchestrator for LKAS system."""

    # Key responsibilities:
    • Configuration loading and resolution
    • Component initialization
    • Process lifecycle coordination
    • Signal handling (Ctrl+C)
    • Terminal display management

    # Delegation:
    • ProcessManagers handle subprocess details
    • BroadcastManager handles ZMQ/SHM
    • Terminal handles UI
```

## Benefits of This Architecture

1. **Separation of Concerns**
   - Each class has ONE clear purpose
   - No mixing of subprocess, broadcasting, and orchestration logic

2. **Reusability**
   - ProcessManager can be extended for new servers
   - BroadcastManager can be reused in other projects

3. **Testability**
   - Mock ProcessManagers to test LKASServer
   - Mock subprocess to test ProcessManagers
   - Mock SHM to test BroadcastManager

4. **Maintainability**
   - Bug in broadcasting? Check BroadcastManager
   - Process crash? Check ProcessManager
   - Orchestration issue? Check LKASServer

5. **Extensibility**
   - Add new process type: Extend ProcessManager
   - Change broadcast protocol: Modify BroadcastManager
   - Add new feature: Clear where it belongs
