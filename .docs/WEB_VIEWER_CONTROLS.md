# Web Viewer Controls

The web viewer now includes interactive controls for managing the simulation directly from your browser!

## Features Added

### 1. Respawn Vehicle
- **Button**: 🔄 Respawn Vehicle
- **Keyboard Shortcut**: `R`
- **Function**: Respawns the vehicle at a new location when it gets stuck or crashes

### 2. Pause/Resume
- **Button**: ⏸ Pause / ▶️ Resume
- **Keyboard Shortcut**: `Space`
- **Function**: Pauses the simulation loop (freezes vehicle and detection)

### 3. Visual Feedback
- Status indicator changes color when paused (green → orange)
- Toast notifications appear when actions are triggered
- Button text changes dynamically (Pause ↔ Resume)

## Usage

### Starting the Web Viewer

```bash
# Terminal 1: Start detection server
cd detection
python detection.py --method cv --port 5556

# Terminal 2: Start simulation with web viewer
cd simulation
python simulation.py \
  --detector-url tcp://localhost:5556 \
  --viewer web \
  --web-port 8080

# Open browser: http://localhost:8080
```

### Keyboard Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| `R` | Respawn | Respawn vehicle at new location |
| `Space` | Pause/Resume | Toggle simulation pause |
| `Q` | Quit | Quit application (in terminal, not browser) |

### Button Controls

Click the buttons in the browser interface:
- **🔄 Respawn Vehicle** - Respawns the vehicle
- **⏸ Pause** / **▶️ Resume** - Toggles pause state

## How It Works

### Architecture

```
Browser (JavaScript)
    ↓ HTTP POST /action
Web Viewer (Python HTTP Server)
    ↓ Callback function
Vehicle Manager / Simulation Loop
    ↓ CARLA API
CARLA Simulator
```

### Technical Details

1. **Frontend (JavaScript)**
   - Listens for keyboard events (`keydown`)
   - Sends POST requests to `/action` endpoint
   - Updates UI based on state

2. **Backend (Python)**
   - `register_action()` - Register callback functions
   - HTTP POST handler receives action requests
   - Executes corresponding callback
   - Returns JSON response

3. **Integration**
   - Callbacks are registered in `simulation.py`
   - Actions directly call vehicle manager methods
   - Pause state is checked in main loop

## Implementation Example

### Registering Custom Actions

```python
from simulation.ui.web_viewer import WebViewer

viewer = WebViewer(port=8080)

# Register custom action
def my_custom_action():
    print("Custom action triggered!")

viewer.register_action('custom', my_custom_action)

viewer.start()
```

### In Browser

```javascript
// Send custom action from browser
fetch('/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'custom'})
});
```

## Extending

To add new controls:

1. **Register action in simulation.py**:
   ```python
   def handle_my_action():
       print("My action!")

   viewer.register_action('my_action', handle_my_action)
   ```

2. **Add button in HTML** (edit `web_viewer.py`):
   ```html
   <button class="btn" onclick="sendAction('my_action')">
       My Action
   </button>
   ```

3. **Add keyboard shortcut**:
   ```javascript
   if (event.key === 'M') {
       sendAction('my_action');
   }
   ```

## Benefits

- **No X11 Required** - Control simulation remotely via browser
- **User-Friendly** - Visual interface instead of terminal commands
- **Extensible** - Easy to add new actions
- **Real-time Feedback** - Toast notifications and status updates

## Troubleshooting

### Actions Not Working

**Check console logs**:
- Browser: Open Developer Tools (F12) → Console tab
- Terminal: Watch for callback messages

**Verify registration**:
```
✓ Web viewer controls registered (R = Respawn, Space = Pause/Resume)
```

### Keyboard Shortcuts Not Responding

- Click somewhere on the page to ensure focus
- Check browser console for JavaScript errors
- Try using button clicks instead

### Vehicle Doesn't Respawn

- Check terminal for error messages
- Ensure CARLA connection is active
- Try spawning manually first to test connection

## Related Files

- [simulation/ui/web_viewer.py](web_viewer.py) - Web viewer implementation
- [simulation/simulation.py](../simulation.py) - Main simulation loop
- [simulation/vehicle.py](../vehicle.py) - Vehicle manager with respawn

## Demo

After starting the simulation:

1. Open http://localhost:8080
2. Press `Space` → Simulation pauses, status turns orange
3. Press `Space` again → Simulation resumes
4. Press `R` → Vehicle respawns at new location
5. See notifications appear in top-right corner

Enjoy controlling your autonomous vehicle from the comfort of your browser! 🚗💨
