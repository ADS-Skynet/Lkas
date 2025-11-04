# Documentation Directory

## Current Documentation

For current project documentation, please refer to:

**👉 [Main README.md](../README.md)** - Start here for current usage instructions

## Important Notes

### Deprecation Notice

Many files in this directory contain **outdated information** from previous architecture iterations. The project has undergone several major restructuring phases:

1. **Phase 1:** Monolithic architecture
2. **Phase 2:** Modular architecture (single-process)
3. **Phase 3:** Distributed architecture (current)

**Current Status (October 2025):** The project exclusively uses **distributed architecture** with separate processes for detection and simulation.

### Deprecated Files

The following documentation files reference removed code (`main_modular.py`, single-process mode) and should be considered **historical references only**:

- ❌ `MODULAR_ARCHITECTURE.md` - Describes removed single-process architecture
- ❌ `SYSTEM_OVERVIEW.md` - References both old and new architectures
- ❌ `CLEANUP_SUMMARY.md` - Historical cleanup notes
- ❌ `DISTRIBUTED_ARCHITECTURE.md` - Compares removed modes
- ❌ `FINAL_STRUCTURE.md` - Shows old structure
- ❌ `README_DISTRIBUTED.md` - References removed code
- ❌ `RESTRUCTURING_SUMMARY.md` - Historical notes
- ❌ `QUICK_START.md` - Contains removed code examples

### Still Relevant Documentation

The following files are still relevant:

- ✅ `START_HERE.md` - M1 Mac dev container setup
- ✅ `DEVCONTAINER_SETUP.md` - Detailed dev container guide
- ✅ `ARCHITECTURE_DECISION.md` - Design decisions (mostly still valid)
- ✅ `VISUALIZATION_GUIDE.md` - Visualization options
- ✅ `SINGLE_PROCESS_REMOVAL.md` - Summary of recent changes
- ✅ `M1_QUICK_REFERENCE.md` - M1 Mac quick reference
- ✅ `MACOS_M1_SETUP.md` - macOS setup guide

## Quick Start (Current)

For the current way to run the system, see the main README:

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
```

## Project Structure (Current)

```
ads_ld/
├── simulation/           # CARLA orchestration
│   └── main_distributed_v2.py  # Only entry point
├── detection/            # Lane detection
│   └── detection_server.py     # Detection server
└── decision/             # Control logic
```

## For New Contributors

1. **Ignore** most files in this `.docs/` directory
2. **Read** the main [README.md](../README.md) instead
3. **Use** the distributed architecture only
4. **Refer** to `SINGLE_PROCESS_REMOVAL.md` for recent changes

---

**Last Updated:** October 27, 2025
**Current Architecture:** Distributed (ZMQ-based multi-process)
