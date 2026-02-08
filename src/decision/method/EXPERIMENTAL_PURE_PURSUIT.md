# Experimental Pure Pursuit Changes (Reverted)

These changes were attempted to fix steering behavior on sharp curves at road merge sections (~90 degree left turn). They were reverted because they broke normal lane-following on straight and gentle-curve sections.

## Problem

At the merge section of the RC track, the road curves sharply left (~90 degrees). The pure pursuit controller was outputting full right steering (+0.90) instead of full left, causing the vehicle to leave the track.

## Root Cause Analysis

The quadratic polynomial `x = a*y^2 + b*y + c` has a vertex at `y = -b / (2a)`. On sharp curves, the vertex can fall within the image frame. Beyond the vertex the parabola reverses direction, which means:
- The lateral error at the lookahead point can invert (point to the wrong side)
- The polynomial derivative (heading) can flip sign

## Changes Attempted

### 1. Vertex Clamping

Detect the vertex of the quadratic and clamp the lookahead point to stay before it:

```python
# Vertex detection for quadratic polynomials
if len(self._center_poly) == 3:
    a, b, c = self._center_poly
    if abs(a) > 1e-9:
        y_vertex = -b / (2 * a)
        # If vertex is between vehicle and lookahead, clamp
        if y_la < y_vertex < self.image_height:
            # Use 85% of the distance from vehicle to vertex
            y_la = self.image_height - 0.85 * (self.image_height - y_vertex)
            y_la = max(y_la, self.image_height * 0.15)  # never look beyond top 15%
```

### 2. Heading Evaluation at Fixed Position (85% Height)

Instead of evaluating the polynomial derivative at the (potentially clamped) lookahead point, always evaluate at 85% of image height for stability:

```python
y_heading = self.image_height * 0.85
deriv_poly = np.polyder(self._center_poly)
slope = float(np.polyval(deriv_poly, y_heading))
heading_deg = float(np.degrees(np.arctan(slope)))
```

### 3. Heading-Error Conflict Detection

When heading and lateral error disagree on the curve direction, trust the heading (which indicates an upcoming sharp curve):

```python
# If heading says "turn left" but lateral error says "go right", trust heading
if heading_term * error_normalized < 0 and abs(heading_term) > 0.3:
    # Heading-error conflict on sharp curve — boost heading, reduce lateral
    steering = -(
        self.gain * 0.3 * error_normalized
        + self.heading_gain * 2.0 * heading_term
    )
```

### 4. Removed `/lookahead_ratio` Division

The original control law divides the lateral error by `lookahead_ratio`:
```python
steering = -(gain * error_normalized / lookahead_ratio + heading_gain * heading_term)
```
This amplifies the error when lookahead_ratio is small (0.4 → 2.5x multiplier). The experimental version removed this division.

### 5. Increased heading_gain Default

Changed `heading_gain` from 0.15 to 0.3 to give more weight to heading correction on curves.

## Why They Were Reverted

When tested on the actual RC track, these changes caused the vehicle to **completely lose tracking** on normal (straight and gentle curve) sections. The combination of changes likely:

- Over-corrected on gentle curves due to the conflict detection triggering too aggressively
- Changed the gain balance (removing /lookahead_ratio + doubling heading_gain) which was tuned for normal driving
- Vertex clamping may have caused erratic lookahead distances on borderline curves

## Conclusion

The sharp merge section is fundamentally an **intersection/path-selection** problem, not a lane-keeping problem. LKAS can only center between detected lane boundaries — when the road topology changes (merge, fork, intersection), higher-level path planning is needed to choose which lane to follow.

For the RC track specifically, this section could potentially be handled by:
- A waypoint-based override for known problematic sections
- GPS/IMU-triggered mode switching
- A higher-level planner that pre-selects the path before the merge

These approaches are outside the scope of the LKAS pure pursuit controller.
