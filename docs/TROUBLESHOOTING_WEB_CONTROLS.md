# Troubleshooting Web Viewer Controls

## Issue: Buttons visible but CARLA doesn't respond

I've added extensive debug logging to help identify the issue. Follow these steps:

### Step 1: Check Terminal Output

When you start the simulation, look for these debug messages:

```
[DEBUG] viz object type: <class 'simulation.integration.visualization.VisualizationManager'>
[DEBUG] viz.viewer type: <class 'simulation.ui.web_viewer.WebViewer'>
[DEBUG] Has register_action? True

✓ Web viewer controls registered successfully!
  • Press 'R' or click 'Respawn' button to respawn vehicle
  • Press 'Space' or click 'Pause' button to pause/resume simulation
```

**If you see this**, actions are registered correctly. Continue to Step 2.

**If you see "⚠ Warning: Could not register"**, the issue is with registration. Check:
- Are you using `--viewer web`?
- Is the web viewer starting correctly?

### Step 2: Test an Action

Click the "Respawn" button or press `R` in the browser.

**Expected output in terminal:**
```
[WebViewer] POST request received: /action
[WebViewer] Action requested: 'respawn'
[WebViewer] Registered actions: ['respawn', 'pause', 'resume']
[WebViewer] Executing action: 'respawn'

🔄 Respawn requested from web viewer
✓ Vehicle respawned successfully
[WebViewer] Action 'respawn' completed
```

### Step 3: Check Browser Console

Open browser Developer Tools (F12) → Console tab

**Look for:**
- Any JavaScript errors?
- Network tab shows POST to `/action`?
- Response is 200 OK?

### Step 4: Test Pause

Click "Pause" or press `Space`.

**Expected output:**
```
[WebViewer] POST request received: /action
[WebViewer] Action requested: 'pause'
[WebViewer] Registered actions: ['respawn', 'pause', 'resume']
[WebViewer] Executing action: 'pause'

⏸ Paused from web viewer - simulation loop will freeze
[WebViewer] Action 'pause' completed
```

**In CARLA:** Vehicle should stop moving (simulation loop paused)

## Common Issues

### Issue 1: No POST requests in terminal

**Symptom:** Clicking buttons shows nothing in terminal

**Cause:** JavaScript not sending requests

**Fix:**
1. Open browser console (F12)
2. Look for JavaScript errors
3. Try manual fetch:
   ```javascript
   fetch('/action', {
       method: 'POST',
       headers: {'Content-Type': 'application/json'},
       body: JSON.stringify({action: 'respawn'})
   }).then(r => r.json()).then(console.log)
   ```

### Issue 2: "Unknown action" error

**Symptom:**
```
[WebViewer] ERROR: Unknown action 'respawn'
[WebViewer] Registered actions: []
```

**Cause:** Actions not registered

**Fix:** Check that terminal shows registration message at startup. Ensure using `--viewer web`.

### Issue 3: Action executes but vehicle doesn't respawn

**Symptom:**
```
🔄 Respawn requested from web viewer
✗ Failed to respawn vehicle
```

**Cause:** Vehicle manager issue

**Fix:**
- Check CARLA connection is active
- Try manual respawn to verify vehicle manager works
- Check for CARLA server errors

### Issue 4: Pause doesn't stop vehicle

**Symptom:** Pause message appears but vehicle keeps moving

**Cause:** Main loop not checking pause state

**Fix:** This should be fixed in the code. Check that you see:
```
⏸ Paused from web viewer - simulation loop will freeze
```

Then watch frame counter in terminal - it should stop incrementing.

## Debug Commands

### Test action registration
```python
# In Python console
from simulation.ui.web_viewer import WebViewer
viewer = WebViewer(port=8080)
viewer.register_action('test', lambda: print("Test action!"))
print(viewer.action_callbacks)  # Should show {'test': <function>}
viewer.start()
```

### Test HTTP endpoint manually
```bash
# In another terminal while simulation running
curl -X POST http://localhost:8080/action \
  -H "Content-Type: application/json" \
  -d '{"action":"respawn"}'
```

Should return: `{"status":"ok","action":"respawn"}`

### Check browser network
1. Open DevTools (F12) → Network tab
2. Click Respawn button
3. Look for POST to `/action`
4. Check Request Payload: `{"action":"respawn"}`
5. Check Response: `{"status":"ok","action":"respawn"}`

## Quick Test Sequence

Run this to verify everything:

1. **Start simulation:**
   ```bash
   python simulation/simulation.py \
     --detector-url tcp://localhost:5556 \
     --viewer web \
     --web-port 8080
   ```

2. **Check startup messages** - Should see "✓ Web viewer controls registered successfully!"

3. **Open browser** → http://localhost:8080

4. **Open console** (F12)

5. **Click Respawn** → Check terminal for messages

6. **Click Pause** → Frame counter should stop

7. **Click Resume** → Frame counter should continue

## Getting More Debug Info

If still not working, add this to see more:

```python
# In simulation.py, after registration
print(f"[DEBUG] Registered callbacks: {viz.viewer.action_callbacks}")
```

## Expected Complete Flow

### Successful Respawn:
```
Browser: User clicks "🔄 Respawn Vehicle"
  ↓
JavaScript: sendAction('respawn')
  ↓
HTTP POST: /action with {"action":"respawn"}
  ↓
Web Viewer: Receives POST, finds 'respawn' in callbacks
  ↓
Callback: handle_respawn() executes
  ↓
Vehicle Manager: vehicle_mgr.respawn_vehicle()
  ↓
CARLA: Vehicle respawns at new location
  ↓
Terminal: "✓ Vehicle respawned successfully"
  ↓
Browser: Shows success notification
```

## If Still Not Working

Please share:
1. Terminal output from startup (especially DEBUG lines)
2. Terminal output when clicking button
3. Browser console errors
4. Network tab request/response

This will help identify where the flow breaks!
