# Synchronous Mode Fix

## Problem
After enabling synchronous mode from Siwoo's codebase, CARLA would freeze and become unresponsive without terminating.

## Root Cause
**Line 131** in [main.py](../simulation/main.py#L134) enables synchronous mode:
```python
carla_conn.setup_synchronous_mode(enabled=True, fixed_delta_seconds=0.05)
```

However, the main loop was **missing `world.tick()`** calls.

### Why This Matters
In **synchronous mode**, CARLA waits for the client to explicitly advance the simulation by calling `world.tick()`. Without this call, the server freezes waiting for the client to request the next frame.

In **asynchronous mode** (default), CARLA runs freely and updates continuously, so no tick is needed.

## Solution

### 1. Added `world.tick()` to Main Loop
**Location:** [main.py:204-205](../simulation/main.py#L204-L205)

```python
try:
    while True:
        # Tick the world (required in synchronous mode)
        if sync_mode:
            carla_conn.get_world().tick()

        # Get image
        image = camera.get_latest_image()
        ...
```

### 2. Added `--no-sync` Flag for Flexibility
**Location:** [main.py:82](../simulation/main.py#L82)

```python
parser.add_argument("--no-sync", action="store_true", help="Disable synchronous mode")
```

Now you can run in either mode:
```bash
# Synchronous mode (default, deterministic)
python simulation/main.py --detector-url tcp://localhost:5556

# Asynchronous mode (faster, non-deterministic)
python simulation/main.py --detector-url tcp://localhost:5556 --no-sync
```

## When to Use Each Mode

### Synchronous Mode (Default)
**Pros:**
- Deterministic simulation
- Reproducible results
- Fixed time step (0.05s = 20 FPS)
- Better for testing and debugging

**Cons:**
- Requires client to tick every frame
- Can be slower if client is bottlenecked
- More complex error handling

**Use when:**
- Running experiments that need reproducibility
- Recording data for analysis
- Debugging control algorithms

### Asynchronous Mode (--no-sync)
**Pros:**
- No tick calls needed
- Simpler code
- CARLA runs at maximum speed
- More forgiving of client delays

**Cons:**
- Non-deterministic timing
- Variable frame rates
- Hard to reproduce exact scenarios

**Use when:**
- Quick testing
- Live demonstrations
- Client is slower than CARLA
- Debugging connection issues

## Testing Results

### Before Fix
```
[1/5] Connecting to CARLA...
✓ Connected to CARLA server

[2/5] Setting up world environment...
✓ World mode set to: sync (Δt=0.05s)
[FREEZE - No response, CARLA stuck waiting for tick]
```

### After Fix (Sync Mode)
```
[1/5] Connecting to CARLA...
✓ Connected to CARLA server

[2/5] Setting up world environment...
✓ World mode set to: sync (Δt=0.05s)

[3/5] Spawning vehicle...
✓ Vehicle spawned

System Running
Frame     30 | FPS:  20.0 | Lanes: LR | Steering: +0.123 | Timeouts: 0
```

### After Fix (Async Mode)
```
[1/5] Connecting to CARLA...
✓ Connected to CARLA server

[2/5] Setting up world environment...
✓ Running in asynchronous mode (--no-sync)

[3/5] Spawning vehicle...
✓ Vehicle spawned

System Running
Frame     30 | FPS:  28.7 | Lanes: LR | Steering: +0.123 | Timeouts: 0
```

## Related Files Modified

1. [simulation/main.py](../simulation/main.py)
   - Line 82: Added `--no-sync` argument
   - Line 132-136: Conditional synchronous mode setup
   - Line 204-205: Added `world.tick()` call in main loop

## Best Practices

### Always Remember
When using synchronous mode:
1. Call `world.tick()` once per frame in your main loop
2. Call it **before** reading sensor data
3. Don't call it multiple times per iteration
4. Handle the case where tick might block

### Example Pattern
```python
# Setup (once at start)
carla_conn.setup_synchronous_mode(enabled=True, fixed_delta_seconds=0.05)

# Main loop
while True:
    # Step 1: Advance simulation
    world.tick()

    # Step 2: Get sensor data (already updated by tick)
    image = camera.get_latest_image()

    # Step 3: Process and apply controls
    detection = detector.detect(image)
    control = controller.process_detection(detection)
    vehicle.apply_control(control)
```

## Additional Notes

- Siwoo's original implementation had tick calls because it was designed for synchronous mode
- Our original main.py ran in async mode, so no tick was needed
- The merge introduced sync mode but forgot to add the tick
- This is a common pitfall when migrating to synchronous mode!

## References

- [CARLA Synchronous Mode Documentation](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)
- [Siwoo's Original Implementation](../archive/siwoo_original/carla/main.py)
