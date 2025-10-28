# Siwoo Codebase Merge Summary

**Date:** October 28, 2025
**Branch:** `lane-detection-merge_decision`
**Original Location:** `/workspaces/ads_ld/siwoo/`
**Archived Location:** `/workspaces/ads_ld/archive/siwoo_original/`

---

## Overview

Successfully merged valuable features from Siwoo's codebase into the main lane detection system while maintaining the robust architecture and ZMQ-based IPC of the main codebase.

---

## Merge Strategy

### What We Kept (Main Codebase - Priority)
1. **Lane Detection Module** - Sophisticated CV + DL detection with temporal smoothing
2. **Architecture** - Distributed ZMQ-based system (production-ready)
3. **Viewer System** - Multi-backend viewer (OpenCV/Pygame/Web)
4. **Decision Structure** - Robust PD controller + LaneAnalyzer
5. **IPC Method** - ZMQ messaging (replaced Siwoo's file-based IPC)

### What We Integrated from Siwoo's Code
1. **World Management Utilities**
   - `cleanup_world()` - Removes pedestrians and vehicles
   - `set_all_traffic_lights_green()` - Freezes traffic lights
   - `setup_synchronous_mode()` - Configures deterministic simulation

2. **Adaptive Throttle Control**
   - Throttle decreases during sharp turns to prevent overshooting
   - Linear interpolation between base (0.45) and min (0.18) throttle
   - Improves vehicle stability during maneuvers

---

## Changes Made

### 1. Enhanced CARLAConnection ([connection.py](../simulation/connection.py))

**Added Methods:**
```python
def setup_synchronous_mode(enabled: bool = True, fixed_delta_seconds: float = 0.05)
def cleanup_world()
def set_all_traffic_lights_green()
```

**Location:** `/workspaces/ads_ld/simulation/connection.py:120-193`

**Benefits:**
- Deterministic simulation with synchronous mode
- Clean environment for testing
- No traffic interference

---

### 2. Adaptive Throttle in DecisionController ([controller.py](../decision/controller.py))

**Added:**
- `ThrottlePolicyConfig` dataclass with parameters:
  - `base`: 0.45 (base throttle)
  - `min`: 0.18 (minimum throttle)
  - `steer_threshold`: 0.15 (when to start reducing)
  - `steer_max`: 0.70 (maximum steering)

**New Method:**
```python
def compute_adaptive_throttle(steering: float) -> float
```

**Location:** `/workspaces/ads_ld/decision/controller.py:79-108`

**Algorithm:**
- If `|steering| <= steer_threshold`: use base throttle (0.45)
- Otherwise: linearly interpolate between base and min based on steering magnitude
- Prevents overshooting in sharp turns

---

### 3. Configuration Updates

**File:** [config.yaml](../simulation/config.yaml:56-62)

**Added Section:**
```yaml
throttle_policy:
  base: 0.45              # Base throttle when steering is minimal
  min: 0.18               # Minimum throttle during sharp turns
  steer_threshold: 0.15   # Steering magnitude to start reducing throttle
  steer_max: 0.70         # Maximum expected steering magnitude
```

**File:** [config.py](../detection/core/config.py)

**Added Classes:**
- `ThrottlePolicyConfig` (line 77-83)
- Updated `Config` class to include `throttle_policy` field (line 120)
- Updated `ConfigManager.load()` to parse throttle policy (line 225-234)

---

### 4. Main Entry Point Updates ([main_distributed_v2.py](../simulation/main_distributed_v2.py))

**Changes:**
1. **World Setup** (lines 129-133):
   - Added world management utilities after CARLA connection
   - Synchronous mode setup (20 FPS, Δt=0.05s)
   - World cleanup and traffic light freezing

2. **Controller Initialization** (lines 158-173):
   - Passes `throttle_policy` from config
   - Prints adaptive throttle status

3. **Visualization** (line 259):
   - Updated HUD to display both steering and throttle

**Steps Changed:**
- `[1/4]` → `[1/5]` Connecting to CARLA
- `[2/4]` → `[2/5]` Setting up world environment (NEW)
- `[3/4]` → `[3/5]` Spawning vehicle
- `[4/4]` → `[4/5]` Setting up camera
- Added `[5/5]` Connecting to detection server

---

## Technical Details

### Adaptive Throttle Formula

```python
if |steering| <= steer_threshold:
    throttle = base
else:
    t = (|steering| - steer_threshold) / (steer_max - steer_threshold)
    t = clamp(t, 0, 1)
    throttle = base - (base - min) * t
    throttle = clamp(throttle, min, base)
```

**Example Values:**
- Steering = 0.0 → Throttle = 0.45 (full speed)
- Steering = 0.15 → Throttle = 0.45 (threshold)
- Steering = 0.40 → Throttle = ~0.29 (reduced)
- Steering = 0.70+ → Throttle = 0.18 (minimum)

---

### World Management Flow

```python
# On startup:
1. Connect to CARLA
2. Setup synchronous mode (fixed_delta_seconds=0.05)
3. Cleanup world (remove all vehicles/pedestrians)
4. Freeze traffic lights (set to green)
5. Spawn our vehicle
6. Attach camera sensor
7. Connect to detection server
8. Start main loop
```

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `simulation/connection.py` | Added 3 world management methods | +75 |
| `decision/controller.py` | Added adaptive throttle logic | +38 |
| `simulation/config.yaml` | Added throttle_policy section | +7 |
| `detection/core/config.py` | Added ThrottlePolicyConfig class | +18 |
| `simulation/main_distributed_v2.py` | Integrated new features | +20 |

**Total Changes:** ~158 lines added

---

## Not Merged (Intentionally)

### Siwoo's Lane Detection
- **Reason:** Main codebase has more sophisticated CV detector with temporal smoothing
- **Location:** `archive/siwoo_original/lane_detection/`
- **Status:** Archived, not integrated

### Siwoo's Decision Logic
- **Reason:** Main codebase has better structure (PD controller + LaneAnalyzer)
- **Location:** `archive/siwoo_original/decision/`
- **Status:** Archived, adaptive throttle extracted and integrated

### File-based IPC
- **Reason:** ZMQ is production-ready, faster, and more robust
- **Location:** `archive/siwoo_original/carla/vehicle.py` (SharedFrameStore)
- **Status:** Archived, replaced with ZMQ

---

## Testing Recommendations

### 1. Basic Functionality Test
```bash
# Terminal 1: Start CARLA
./CarlaUE4.sh

# Terminal 2: Start detection server
python detection/detection_server.py

# Terminal 3: Start CARLA client
python simulation/main_distributed_v2.py --detector-url tcp://localhost:5555
```

**Expected Output:**
```
[1/5] Connecting to CARLA...
✓ Connected to CARLA server

[2/5] Setting up world environment...
✓ World mode set to: sync (Δt=0.05s)
✓ World cleaned: removed X pedestrians and Y vehicles
✓ Set Z traffic lights to GREEN and frozen

[3/5] Spawning vehicle...
✓ Vehicle spawned

[4/5] Setting up camera...
✓ Camera sensor attached

[5/5] Connecting to detection server...
✓ Connected to detection server

✓ Adaptive throttle enabled: base=0.45, min=0.18
```

### 2. Adaptive Throttle Test
- Drive on curved roads
- Observe throttle decreasing during turns
- Monitor HUD display: `Steering: +0.350 | Throttle: 0.280`
- Verify smooth control without overshooting

### 3. World Management Test
- Check that no other vehicles/pedestrians spawn
- Verify all traffic lights are green
- Confirm synchronous mode (stable FPS)

---

## Rollback Instructions

If issues arise, revert these commits:

```bash
# View recent commits
git log --oneline

# Revert merge (if committed)
git revert <commit-hash>

# Or restore Siwoo's code from archive
cp -r archive/siwoo_original siwoo
```

---

## Integration Benefits

1. **Improved Stability**
   - Adaptive throttle prevents overshooting
   - Better control during sharp turns

2. **Cleaner Environment**
   - No interference from other vehicles
   - Consistent traffic light states

3. **Deterministic Simulation**
   - Synchronous mode ensures reproducibility
   - Fixed time step (0.05s = 20 FPS)

4. **Maintained Architecture**
   - ZMQ-based IPC still intact
   - Production-ready distributed system
   - Multi-viewer support preserved

---

## Future Enhancements

Potential improvements based on Siwoo's patterns:

1. **OOP Vehicle Class** (optional)
   - Consider refactoring `VehicleManager` to be more self-contained
   - Follow Siwoo's `CarlaVehicle` pattern

2. **Dynamic Throttle Tuning**
   - Add runtime throttle policy adjustment
   - Expose via CLI arguments

3. **Speed-based Throttle**
   - Incorporate vehicle speed into throttle calculation
   - Adaptive based on both steering and current velocity

---

## References

- **Siwoo's Original Code:** `/workspaces/ads_ld/archive/siwoo_original/`
- **Main Codebase Docs:** `/workspaces/ads_ld/README.md`
- **Configuration:** `/workspaces/ads_ld/simulation/config.yaml`

---

## Contributors

- **Siwoo:** Original world management and adaptive throttle implementation
- **Main Team:** Distributed architecture, ZMQ integration, lane detection
- **Merge:** Integration of best features from both codebases

---

**Status:** ✅ Merge Complete
**Testing:** ⏳ Pending
**Documentation:** ✅ Complete
