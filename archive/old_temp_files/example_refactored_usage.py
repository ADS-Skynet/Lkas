"""
Complete Example: Using the Refactored OOP Architecture

This script demonstrates how to use the new refactored components.
Perfect for learning Python OOP concepts!

TUTORIAL FOR C++ DEVELOPERS:
- Shows factory pattern
- Shows type-safe data models
- Shows configuration management
- Explains Python class concepts as we go
"""

import numpy as np
import cv2

# Import refactored components
from core.config import Config, ConfigManager
from core.factory import DetectorFactory
from core.models import Lane, DetectionResult


def demo_basic_usage():
    """
    DEMO 1: Basic usage with factory pattern.

    CONCEPT: Factory Pattern
    - Don't create detectors directly
    - Use factory to create them
    - Factory handles all configuration
    """
    print("=" * 70)
    print("DEMO 1: Basic Factory Pattern Usage")
    print("=" * 70)

    # Step 1: Load configuration
    print("\n1. Loading configuration...")
    config = ConfigManager.load('config.yaml')
    print(f"   ✓ Config loaded: CARLA at {config.carla.host}:{config.carla.port}")

    # Step 2: Create factory
    print("\n2. Creating detector factory...")
    factory = DetectorFactory(config)
    print(f"   ✓ Factory created")
    print(f"   Available detectors: {factory.list_available_detectors()}")

    # Step 3: Create CV detector
    print("\n3. Creating CV detector using factory...")
    cv_detector = factory.create('cv')
    print(f"   ✓ Detector created: {cv_detector.get_name()}")
    print(f"   Parameters: {cv_detector.get_parameters()}")

    # Step 4: Create DL detector
    print("\n4. Creating DL detector using factory...")
    dl_detector = factory.create('dl')
    print(f"   ✓ Detector created: {dl_detector.get_name()}")
    print(f"   Parameters: {dl_detector.get_parameters()}")

    print("\n" + "=" * 70)
    print("KEY POINT: Factory pattern makes it easy to switch detectors!")
    print("=" * 70)


def demo_detection_with_cv():
    """
    DEMO 2: Running detection with CV method.

    CONCEPT: Using abstract interface
    - All detectors implement same interface
    - Can swap detectors without changing code
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Detection with CV Method")
    print("=" * 70)

    # Create config and factory
    config = Config()  # Use defaults
    factory = DetectorFactory(config)

    # Create CV detector
    detector = factory.create('cv')
    print(f"Using: {detector.get_name()}")

    # Create test image (random noise for demo)
    print("\nCreating test image...")
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    # Run detection
    print("Running detection...")
    result = detector.detect(test_image)

    # Access results using DetectionResult object
    print("\nResults:")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Has left lane: {result.has_left_lane}")
    print(f"  Has right lane: {result.has_right_lane}")
    print(f"  Has both lanes: {result.has_both_lanes}")

    if result.left_lane:
        print(f"\n  Left lane details:")
        print(f"    Coordinates: ({result.left_lane.x1}, {result.left_lane.y1}) → ({result.left_lane.x2}, {result.left_lane.y2})")
        print(f"    Slope: {result.left_lane.slope:.2f}")
        print(f"    Length: {result.left_lane.length:.1f}px")

    print("\n" + "=" * 70)
    print("KEY POINT: DetectionResult object is type-safe and self-documenting!")
    print("=" * 70)


def demo_detection_with_dl():
    """
    DEMO 3: Running detection with DL method.

    CONCEPT: Polymorphism
    - Same code works with different detector types
    - This is the power of abstract interfaces!
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Detection with DL Method")
    print("=" * 70)

    # Create config and factory
    config = Config()
    factory = DetectorFactory(config)

    # Create DL detector (notice: same code as CV!)
    detector = factory.create('dl')
    print(f"Using: {detector.get_name()}")

    # Create test image
    print("\nCreating test image...")
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    # Run detection (SAME CODE as CV!)
    print("Running detection...")
    result = detector.detect(test_image)

    # Access results (SAME CODE as CV!)
    print("\nResults:")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Has left lane: {result.has_left_lane}")
    print(f"  Has right lane: {result.has_right_lane}")

    if result.left_lane:
        print(f"\n  Left lane:")
        print(f"    Slope: {result.left_lane.slope:.2f}")
        print(f"    Length: {result.left_lane.length:.1f}px")

    print("\n" + "=" * 70)
    print("KEY POINT: CV and DL detectors are interchangeable!")
    print("Same interface → same code → easy to swap")
    print("=" * 70)


def demo_configuration_customization():
    """
    DEMO 4: Customizing configuration.

    CONCEPT: Configuration-driven design
    - Change behavior without changing code
    - Just modify config values
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Configuration Customization")
    print("=" * 70)

    # Create custom config
    config = Config()

    print("\n1. Default CV detector parameters:")
    print(f"   Canny low: {config.cv_detector.canny_low}")
    print(f"   Canny high: {config.cv_detector.canny_high}")
    print(f"   Hough threshold: {config.cv_detector.hough_threshold}")

    # Modify config
    print("\n2. Customizing parameters...")
    config.cv_detector.canny_low = 30
    config.cv_detector.canny_high = 200
    config.cv_detector.hough_threshold = 70

    print(f"   Canny low: {config.cv_detector.canny_low} (changed!)")
    print(f"   Canny high: {config.cv_detector.canny_high} (changed!)")
    print(f"   Hough threshold: {config.cv_detector.hough_threshold} (changed!)")

    # Create detector with custom config
    factory = DetectorFactory(config)
    detector = factory.create('cv')

    print("\n3. Detector created with custom parameters:")
    params = detector.get_parameters()
    print(f"   Canny low: {params['canny_low']}")
    print(f"   Canny high: {params['canny_high']}")
    print(f"   Hough threshold: {params['hough_threshold']}")

    print("\n" + "=" * 70)
    print("KEY POINT: Configuration makes tuning easy!")
    print("No code changes needed, just config values")
    print("=" * 70)


def demo_lane_object_features():
    """
    DEMO 5: Lane object features.

    CONCEPT: Data models with computed properties
    - Lane is not just data storage
    - Has useful methods and properties
    - Better than plain tuples!
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Lane Object Features")
    print("=" * 70)

    # Create Lane objects
    print("\n1. Creating Lane objects...")

    # Old way (tuple - confusing!)
    old_lane = (100, 600, 120, 360)
    print(f"   OLD: {old_lane}")
    print("        What does each number mean? 🤷")

    # New way (Lane object - clear!)
    new_lane = Lane(x1=100, y1=600, x2=120, y2=360, confidence=0.95)
    print(f"   NEW: Lane(x1={new_lane.x1}, y1={new_lane.y1}, x2={new_lane.x2}, y2={new_lane.y2})")
    print("        Self-documenting! ✨")

    # Use computed properties
    print("\n2. Computed properties (no manual calculation needed):")
    print(f"   Slope: {new_lane.slope:.2f}")
    print(f"   Length: {new_lane.length:.1f}px")
    print(f"   Confidence: {new_lane.confidence}")

    # Convert between formats
    print("\n3. Convert to/from tuple:")
    as_tuple = new_lane.as_tuple()
    print(f"   as_tuple(): {as_tuple}")

    from_tuple = Lane.from_tuple((200, 600, 220, 360), confidence=0.88)
    print(f"   from_tuple(): {from_tuple}")

    print("\n" + "=" * 70)
    print("KEY POINT: Lane objects are smarter than tuples!")
    print("Properties, methods, type safety - much better!")
    print("=" * 70)


def demo_python_vs_cpp():
    """
    DEMO 6: Python OOP concepts for C++ developers.

    Explains the keywords and concepts.
    """
    print("\n" + "=" * 70)
    print("DEMO 6: Python OOP Concepts for C++ Developers")
    print("=" * 70)

    print("""
PYTHON → C++ TRANSLATION:

1. CLASS DEFINITION:
   Python: class MyClass(BaseClass):
   C++:    class MyClass : public BaseClass { ... };

2. CONSTRUCTOR:
   Python: def __init__(self, arg):
   C++:    MyClass(arg) { ... }

3. INSTANCE REFERENCE:
   Python: self (explicit first parameter)
   C++:    this (implicit pointer)

4. INSTANCE VARIABLE:
   Python: self.variable = value
   C++:    this->variable = value;

5. METHOD:
   Python: def method(self, arg):
   C++:    void method(arg) { ... }

6. TYPE HINTS:
   Python: def func(arg: int) -> str:
   C++:    string func(int arg) { ... }

7. NULLABLE TYPES:
   Python: Optional[Lane] (can be None)
   C++:    Lane* (can be nullptr)

8. INHERITANCE:
   Python: All methods are virtual by default
   C++:    Need 'virtual' keyword

9. ABSTRACT METHODS:
   Python: @abstractmethod
   C++:    virtual void method() = 0;

10. PROPERTIES:
    Python: @property (getter without parens)
    C++:    Must call getter(): obj.getValue()

KEY DIFFERENCES:
- Python: Duck typing (runtime), C++: Static typing (compile-time)
- Python: No private keyword, C++: private/public/protected
- Python: self explicit, C++: this implicit
- Python: Everything is reference, C++: Value vs reference
    """)

    print("=" * 70)


def main():
    """
    Main function - run all demonstrations.

    Shows complete workflow from configuration to detection.
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "REFACTORED OOP SYSTEM - EXAMPLES" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run all demos
    demo_basic_usage()
    demo_detection_with_cv()
    demo_detection_with_dl()
    demo_configuration_customization()
    demo_lane_object_features()
    demo_python_vs_cpp()

    # Final summary
    print("\n" + "=" * 70)
    print("✨ SUMMARY OF BENEFITS")
    print("=" * 70)
    print("""
✅ TYPE SAFETY       - Lane objects instead of tuples
✅ CONFIGURATION     - YAML file, no hardcoded values
✅ FACTORY PATTERN   - Easy detector creation and swapping
✅ POLYMORPHISM      - Same code works with CV and DL
✅ EXTENSIBILITY     - Add new detectors easily
✅ MAINTAINABILITY   - Clean, organized, documented code
✅ TESTABILITY       - Each component testable independently
✅ IDE SUPPORT       - Autocomplete, type hints everywhere
    """)

    print("=" * 70)
    print("📚 NEXT STEPS")
    print("=" * 70)
    print("""
1. Read the code in this file - lots of explanations!
2. Try modifying config values and see what changes
3. Experiment with creating your own detectors
4. Read REFACTORING_GUIDE.md for architecture details
5. Ask questions about any Python concepts you're unsure about!
    """)

    print("=" * 70)
    print("🎉 Examples Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
