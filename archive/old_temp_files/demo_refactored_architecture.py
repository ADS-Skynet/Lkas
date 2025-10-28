"""
Demonstration of the refactored OOP architecture.

This script shows how to use the new components without touching the old code.
Run this to see the new architecture in action!
"""

import numpy as np
from core.models import Lane, LaneMetrics, LaneDepartureStatus, DetectionResult
from core.config import ConfigManager, Config


def demo_data_models():
    """Demonstrate type-safe data models."""
    print("=" * 60)
    print("DEMO 1: Type-Safe Data Models")
    print("=" * 60)

    # OLD WAY (Tuples - confusing!)
    print("\n❌ OLD WAY: Tuples")
    left_lane_old = (100, 600, 120, 360)
    print(f"  left_lane = {left_lane_old}")
    print("  What does each number mean? 🤷")

    # NEW WAY (Lane objects - self-documenting!)
    print("\n✅ NEW WAY: Lane Objects")
    left_lane = Lane(x1=100, y1=600, x2=120, y2=360, confidence=0.95)
    print(f"  left_lane = Lane(x1={left_lane.x1}, y1={left_lane.y1}, "
          f"x2={left_lane.x2}, y2={left_lane.y2})")
    print(f"  slope = {left_lane.slope:.2f} (calculated property!)")
    print(f"  length = {left_lane.length:.1f}px (calculated property!)")
    print(f"  confidence = {left_lane.confidence}")

    # Lane metrics
    print("\n✅ Lane Metrics Object")
    metrics = LaneMetrics(
        vehicle_center_x=400.0,
        lane_center_x=420.0,
        lateral_offset_meters=0.15,
        departure_status=LaneDepartureStatus.CENTERED,
        has_both_lanes=True
    )
    print(f"  Offset: {metrics.lateral_offset_meters}m")
    print(f"  Status: {metrics.departure_status.value}")
    print(f"  Has both lanes: {metrics.has_both_lanes}")

    # Detection result
    print("\n✅ Detection Result")
    result = DetectionResult(
        left_lane=left_lane,
        right_lane=Lane(x1=700, y1=600, x2=680, y2=360),
        debug_image=None,
        processing_time_ms=12.5
    )
    print(f"  Has left lane: {result.has_left_lane}")
    print(f"  Has right lane: {result.has_right_lane}")
    print(f"  Processing time: {result.processing_time_ms}ms")


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n" + "=" * 60)
    print("DEMO 2: Configuration Management")
    print("=" * 60)

    # Create default config
    config = Config()

    print("\n✅ CARLA Configuration")
    print(f"  Host: {config.carla.host}")
    print(f"  Port: {config.carla.port}")
    print(f"  Vehicle: {config.carla.vehicle_type}")

    print("\n✅ Camera Configuration")
    print(f"  Resolution: {config.camera.width}x{config.camera.height}")
    print(f"  FOV: {config.camera.fov}°")
    print(f"  Position: {config.camera.position}")
    print(f"  Rotation: {config.camera.rotation}")

    print("\n✅ CV Detector Configuration")
    print(f"  Canny low: {config.cv_detector.canny_low}")
    print(f"  Canny high: {config.cv_detector.canny_high}")
    print(f"  Hough threshold: {config.cv_detector.hough_threshold}")
    print(f"  Smoothing factor: {config.cv_detector.smoothing_factor}")

    print("\n✅ DL Detector Configuration")
    print(f"  Model type: {config.dl_detector.model_type}")
    print(f"  Input size: {config.dl_detector.input_size}")
    print(f"  Threshold: {config.dl_detector.threshold}")
    print(f"  Device: {config.dl_detector.device}")

    print("\n✅ Analyzer Configuration")
    print(f"  Drift threshold: {config.analyzer.drift_threshold}")
    print(f"  Departure threshold: {config.analyzer.departure_threshold}")
    print(f"  Lane width: {config.analyzer.lane_width_meters}m")

    print("\n✅ Controller Configuration")
    print(f"  Kp (proportional): {config.controller.kp}")
    print(f"  Kd (derivative): {config.controller.kd}")

    print("\n✅ Visualization Configuration")
    print(f"  Left lane color (BGR): {config.visualization.color_left_lane}")
    print(f"  Right lane color (BGR): {config.visualization.color_right_lane}")
    print(f"  Alert blink frequency: {config.visualization.alert_blink_frequency}")


def demo_configuration_loading():
    """Demonstrate loading config from YAML."""
    print("\n" + "=" * 60)
    print("DEMO 3: Loading Configuration from YAML")
    print("=" * 60)

    # Try to load config.yaml
    print("\n📄 Attempting to load config.yaml...")
    config = ConfigManager.load('config.yaml')

    if config:
        print("✅ Config loaded successfully!")
        print(f"  CARLA host from config: {config.carla.host}")
        print(f"  Camera width from config: {config.camera.width}")
    else:
        print("⚠️  Config file not found, using defaults")


def demo_before_after_comparison():
    """Show before/after code comparison."""
    print("\n" + "=" * 60)
    print("DEMO 4: Before vs After Comparison")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("❌ BEFORE: Hardcoded, unclear, no type safety")
    print("-" * 60)
    print("""
# Creating a detector (OLD WAY)
detector = CVLaneDetector(
    canny_low=50,           # What's a good value?
    canny_high=150,         # Magic number!
    hough_threshold=50,     # Another magic number!
    # ... 8 more parameters to remember
)

# Processing result (OLD WAY)
left_lane, right_lane, debug_img = detector.detect(image)
# left_lane is (x1, y1, x2, y2) - which is which?

# Creating metrics (OLD WAY)
metrics = {
    'lateral_offset_meters': 0.15,
    'departure_status': 'Centered',  # String (typo-prone!)
    # ... 8 more dictionary keys to remember
}
""")

    print("\n" + "-" * 60)
    print("✅ AFTER: Configuration-driven, type-safe, clean")
    print("-" * 60)
    print("""
# Loading config (NEW WAY)
config = ConfigManager.load('config.yaml')

# Creating detector (NEW WAY)
factory = DetectorFactory(config)
detector = factory.create('cv')  # All params from config!

# Processing result (NEW WAY)
result = detector.detect(image)
# result.left_lane is Lane object with properties!
# result.right_lane is Lane object
# result.debug_image is the visualization

# Creating metrics (NEW WAY)
metrics = LaneMetrics(
    lateral_offset_meters=0.15,
    departure_status=LaneDepartureStatus.CENTERED,  # Enum (type-safe!)
)
# IDE autocomplete knows all fields!
""")


def demo_extensibility():
    """Show how easy it is to extend the system."""
    print("\n" + "=" * 60)
    print("DEMO 5: Extensibility")
    print("=" * 60)

    print("\n✅ Adding a new detector is easy!")
    print("""
# 1. Create your detector class
class MyCustomDetector(LaneDetector):  # Inherit from ABC
    def detect(self, image):
        # Your custom detection logic
        left = Lane(...)
        right = Lane(...)
        return DetectionResult(left_lane=left, right_lane=right)

    def get_name(self):
        return "My Custom Detector"

    def get_parameters(self):
        return {'param1': value1, ...}

# 2. Register in factory (add to factory.py)
# That's it! Now you can use it:
detector = factory.create('custom')
""")

    print("\n✅ Swapping detectors is trivial!")
    print("""
# OLD WAY: Change code, recompile logic
if method == "cv":
    detector = CVLaneDetector(param1, param2, ...)
elif method == "dl":
    detector = DLLaneDetector(param3, param4, ...)
elif method == "custom":  # Forgot to handle this!
    # Error!

# NEW WAY: Just change config or argument
detector = factory.create('cv')   # Uses CV
detector = factory.create('dl')   # Uses DL
detector = factory.create('custom')  # Uses custom
# No code changes needed!
""")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "REFACTORED OOP ARCHITECTURE DEMO" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")

    demo_data_models()
    demo_configuration()
    demo_configuration_loading()
    demo_before_after_comparison()
    demo_extensibility()

    print("\n" + "=" * 60)
    print("✨ BENEFITS SUMMARY")
    print("=" * 60)
    print("""
✅ Type Safety        - Lane objects instead of tuples
✅ Configuration      - YAML file instead of hardcoded values
✅ Extensibility      - Easy to add new detectors
✅ Testability        - Each component testable in isolation
✅ Maintainability    - Clear structure, single responsibility
✅ IDE Support        - Autocomplete, type hints everywhere
✅ Documentation      - Self-documenting with dataclasses
""")

    print("\n" + "=" * 60)
    print("📚 NEXT STEPS")
    print("=" * 60)
    print("""
1. Read REFACTORING_GUIDE.md for full details
2. Review core/ directory code
3. Test config loading with your config.yaml
4. Decide which phase to tackle first
5. Ask for help refactoring specific components!
""")

    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
