# Archive Directory

This directory contains deprecated files and documentation from the project's evolution. These files are kept for historical reference but should **NOT** be used in active development.

## Directory Structure

### `deprecated_main_files/`
Contains old entry points and implementations that have been replaced:
- **main.py** - Original OOP monolithic version
- **main_distributed.py** - First distributed architecture attempt (v1)
- **carla_interface.py** - Old monolithic CARLA interface
- **test_connection.py** - Duplicate test file
- **model.py** - Simple compatibility wrapper

**Current Alternatives:**
- Use `lane_detection/main_modular.py` for single-process architecture
- Use `lane_detection/main_distributed_v2.py` for distributed architecture
- Use `lane_detection/modules/carla_module/` for CARLA functionality

### `old_temp_files/`
Contains temporary/demo files from development and refactoring:
- **benchmark_performance.py** - Old benchmarking script
- **cv_lane_detector.py** - Duplicate of `method/computer_vision/cv_lane_detector.py`
- **demo_refactored_architecture.py** - Refactoring demo
- **example_refactored_usage.py** - Usage examples
- **lane_net.py** - Duplicate of `method/deep_learning/lane_net.py`

**Current Alternatives:**
- Use implementations in `lane_detection/method/` directory
- See active documentation for usage examples

### `deprecated_docs/`
Contains documentation from the refactoring process:
- **PHASE2_COMPLETE.md** - Phase 2 completion notes
- **PHASE3_COMPLETE.md** - Phase 3 completion notes
- **REFACTORING_GUIDE.md** - Historical refactoring guide
- **COMPLETE_REFACTORING_SUMMARY.md** - Detailed refactoring summary
- **claude_suggestion.md** - Early architecture discussions

**Current Alternatives:**
- See `docs/ARCHITECTURE_DECISION.md` for current architecture rationale
- See `docs/MODULAR_ARCHITECTURE.md` for architecture explanation
- See `lane_detection/SYSTEM_OVERVIEW.md` for system components

## Important Notes

1. **Do NOT import from archived files** - All functionality has been preserved in the active codebase
2. **For reference only** - These files show the evolution of the project but are not maintained
3. **Can be deleted** - If disk space is needed, this entire directory can be safely removed
4. **Not in git** - Consider adding `archive/` to `.gitignore` if you don't want to track historical files

## Migration Path

If you have old code using these files:

```python
# OLD (don't use)
from carla_interface import CARLAInterface

# NEW (use this)
from modules.carla_module import CARLAConnection, VehicleManager, CameraSensor
```

```bash
# OLD (don't use)
python lane_detection/main.py --method cv

# NEW (use this)
python lane_detection/main_modular.py --method cv
# or
python lane_detection/main_distributed_v2.py --detector-url tcp://localhost:5555
```

## Questions?

See `CLEANUP_SUMMARY.md` in the project root for details on what was cleaned up and why.
