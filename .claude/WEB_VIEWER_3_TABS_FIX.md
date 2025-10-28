# Web Viewer 3-Tab Auto-Open Issue - FIXED

## Problem
When starting the web viewer, **3 browser tabs automatically opened** showing the same URL.

## Root Cause
The URL `http://localhost:8080` was being printed **3 times** in the console:

1. **visualization.py:73** - `"View at: http://localhost:8080"`
2. **web_viewer.py:220** - `"View at: http://localhost:8080"`
3. **main.py:213** - `"View at: http://localhost:8080"`

**Why this causes 3 tabs:**
- VSCode (and many modern terminals) detect URLs in terminal output
- They make URLs clickable or auto-open them
- Each URL print triggered a separate browser tab/window

## Solution Applied

### Fixed Files:

**1. visualization.py (Line 72-73)**
```python
# BEFORE:
print(f"✓ Using Web viewer")
print(f"  View at: http://localhost:{self.web_port}")

# AFTER:
print(f"✓ Using Web viewer on port {self.web_port}")
# URL is printed by web_viewer.py, don't print it here too
```

**2. main.py (Line 212-213)**
```python
# BEFORE:
if viewer_type == "web":
    print(f"View at: http://localhost:{args.web_port}")

# AFTER:
# URL already printed by web_viewer.py, don't print again
```

**3. web_viewer.py (Line 217-224)** - KEPT (single source of truth)
```python
if not self.quiet:
    print(f"\n✓ Web viewer started")
    print(f"  View at: http://localhost:{self.port}")
    print(f"  (Copy and paste the URL manually into your browser)")
    print(f"  Press Ctrl+C to stop\n")
```

### Result:
Now the URL is printed **only once** → **only 1 tab opens** (or none if you disable auto-open)

---

## Additional Improvements

### 1. Added `quiet` Mode
You can now suppress URL output completely:

```python
viewer = WebViewer(port=8080, quiet=True)
```

### 2. Request Logging
Added debug logging to see what's being requested:

```python
[WebViewer] Request #1: /
[WebViewer] Request #2: /stream
[WebViewer] Request #3: /favicon.ico
```

### 3. Favicon Handler
Prevents browsers from retrying `/favicon.ico` requests:

```python
elif self.path == '/favicon.ico':
    self.send_response(204)  # No Content
    self.end_headers()
```

### 4. Cleared Python Cache
Removed stale `.pyc` files to ensure fresh code runs:
```bash
find /workspaces/ads_ld/simulation -type d -name "__pycache__" -exec rm -rf {} +
```

---

## Testing

### Before Fix:
```bash
python simulation/main.py --viewer web

# Output:
✓ Using Web viewer
  View at: http://localhost:8080     # Tab 1 opens
  View at: http://localhost:8080     # Tab 2 opens
View at: http://localhost:8080       # Tab 3 opens
```

### After Fix:
```bash
python simulation/main.py --viewer web

# Output:
✓ Using Web viewer on port 8080

✓ Web viewer started
  View at: http://localhost:8080     # Only 1 tab opens (or manually copy)
  (Copy and paste the URL manually into your browser)
```

---

## How to Disable Auto-Open Completely

If VSCode/terminal still auto-opens the URL, you can:

### Option 1: Disable VSCode URL Detection
**Settings.json:**
```json
{
  "terminal.integrated.enableLinks": false
}
```

### Option 2: Use Different Port
```bash
python simulation/main.py --viewer web --web-port 8081
```

### Option 3: Format URL Differently
Edit `web_viewer.py` to break the URL format:
```python
print(f"  View at: localhost port {self.port} (http)")
```

---

## Files Modified

| File | Line | Change |
|------|------|--------|
| `simulation/integration/visualization.py` | 72-73 | Removed duplicate URL print |
| `simulation/main.py` | 212-213 | Removed duplicate URL print |
| `simulation/ui/web_viewer.py` | 28-37 | Added `quiet` parameter |
| `simulation/ui/web_viewer.py` | 57-63 | Added request logging |
| `simulation/ui/web_viewer.py` | 107-110 | Added favicon handler |
| `simulation/ui/web_viewer.py` | 217-224 | Improved URL output message |

---

## Summary

**Problem:** 3 browser tabs auto-opening
**Cause:** URL printed 3 times
**Solution:** Print URL only once
**Result:** ✅ Fixed - only 1 tab opens now

---

**Status:** ✅ Fixed and tested
**Date:** 2025-10-28
