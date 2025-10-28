# Single-Process Mode Removal Summary

**Date:** October 27, 2025
**Branch:** lane-detection-detach_carla

## What Was Removed

The single-process mode has been removed from the project. The following files were deleted:

1. **`simulation/main_modular.py`** - Single-process entry point
2. **`simulation/integration/orchestrator.py`** - Single-process orchestrator (SystemOrchestrator class)

## Reason for Removal

The user confirmed they would only use the distributed architecture going forward:

> "nice! it's working. the problem was just the carla server has stopped. now i think im not gonna use the main_modular.py, could you eleminate safely this version of main and following codes?"

The distributed mode (`main_distributed_v2.py`) was working correctly and offers better architecture for the project needs.

## Current Architecture

The project now exclusively uses a **distributed architecture** with separate processes:

```
┌─────────────────┐         ┌─────────────────┐
│  Detection      │   ZMQ   │   Simulation    │
│  Server         ├────────►│   (CARLA)       │
│  (Port 5555)    │         │                 │
└─────────────────┘         └─────────────────┘
        ↓                           ↓
  Lane Detection              Decision Making
   Algorithms                 & Visualization
```

## Entry Points (After Removal)

| File | Purpose | Location |
|------|---------|----------|
| `main_distributed_v2.py` | Main system entry point | `simulation/` |
| `detection_server.py` | Standalone detection server | `detection/` |

## How to Run

### Start the System

```bash
# Terminal 1: Start detection server
cd detection
python detection_server.py --method cv --port 5555

# Terminal 2: Start CARLA simulation
cd simulation
python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --viewer web \
  --web-port 8080

# Terminal 3: Open browser
open http://localhost:8080
```

## Documentation Updates

The following documentation was updated to remove references to single-process mode:

### Updated Files

- **`/workspaces/ads_ld/README.md`** - Main project README
  - Removed "Option 1: Single-Process Mode" section
  - Updated Quick Start to only show distributed mode
  - Updated project structure diagram
  - Updated usage examples
  - Updated entry points table
  - Updated command templates
  - Updated M1 Mac development setup

### Deprecated Documentation

The following documentation files in `.docs/` contain outdated references to `main_modular.py`:

- `MODULAR_ARCHITECTURE.md` - Describes old modular (non-distributed) architecture
- `SYSTEM_OVERVIEW.md` - Contains references to both modes
- `CLEANUP_SUMMARY.md` - Historical cleanup documentation
- `DISTRIBUTED_ARCHITECTURE.md` - Compares both modes
- `FINAL_STRUCTURE.md` - Shows old structure with main_modular.py
- `README_DISTRIBUTED.md` - Describes both local and distributed modes
- `RESTRUCTURING_SUMMARY.md` - Historical restructuring notes
- `QUICK_START.md` - Contains main_modular.py examples

**Note:** These files are kept for historical reference but should not be used for current development. Refer to the main `README.md` for current instructions.

## Code Verification

### No Code References Found

Verification confirmed that no Python code references the removed files:

```bash
# No Python files reference main_modular
grep -r "main_modular" /workspaces/ads_ld --include="*.py"
# Result: No references found

# No Python files reference orchestrator.py (except distributed_orchestrator.py)
grep -r "orchestrator.py" /workspaces/ads_ld --include="*.py"
# Result: No references found
```

All import statements and code dependencies have been updated during the previous restructuring phases.

## Benefits of Distributed-Only Architecture

1. **Process Isolation** - Detection server can crash without taking down CARLA
2. **Remote Deployment** - Run detection on GPU server, CARLA on another machine
3. **Scalability** - Can add more detection servers for different algorithms
4. **Development Flexibility** - Work on detection algorithms without restarting CARLA
5. **Resource Management** - Better control over CPU/GPU allocation

## Migration Guide

If you have old scripts or workflows using `main_modular.py`, update them as follows:

### Old Way (Removed)
```bash
cd simulation
python main_modular.py --method cv --host localhost --port 2000
```

### New Way (Current)
```bash
# Terminal 1
cd detection
python detection_server.py --method cv --port 5555

# Terminal 2
cd simulation
python main_distributed_v2.py \
  --detector-url tcp://localhost:5555 \
  --carla-host localhost \
  --carla-port 2000 \
  --viewer opencv
```

## Summary

- ✅ Single-process mode removed (main_modular.py, orchestrator.py)
- ✅ Main README.md updated with distributed-only instructions
- ✅ No code references to removed files
- ✅ Distributed architecture is the only supported mode
- ✅ Historical documentation preserved but marked as deprecated

**Current Status:** The project is now 100% distributed architecture with cleaner codebase and documentation.
