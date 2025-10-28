"""
Frame Processor - Orchestrates the detection → analysis → visualization pipeline.

SINGLE RESPONSIBILITY: Process one frame through the complete pipeline.

For C++ developers:
    This is like a pipeline class that chains operations together.
    Similar to: FrameProcessor::process() calling multiple methods in sequence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Tuple, Optional

from detection.core.interfaces import LaneDetector
from detection.core.models import DetectionResult, LaneMetrics
from simulation.utils.lane_analyzer import LaneAnalyzer
from simulation.utils.visualizer import LKASVisualizer


class FrameProcessor:
    """
    Processes a single frame through the complete lane detection pipeline.

    Pipeline:
        Input Image → Detector → Analyzer → Visualizer → Output

    KEYWORD EXPLANATION:
        class FrameProcessor: - Defines a new class
        def __init__(self, ...): - Constructor (like FrameProcessor(...) in C++)
        self.detector - Instance variable (like this->detector in C++)
    """

    def __init__(self,
                 detector: LaneDetector,
                 analyzer: LaneAnalyzer,
                 visualizer: LKASVisualizer):
        """
        Initialize frame processor.

        COMPOSITION PATTERN: FrameProcessor "has a" detector, analyzer, visualizer
        Not inheritance - we're composing functionality!

        Args:
            detector: Lane detection component (CV or DL)
            analyzer: Lane analysis component
            visualizer: Visualization component
        """
        # Store references to components
        # self.variable: Instance variable (member variable in C++)
        self.detector = detector
        self.analyzer = analyzer
        self.visualizer = visualizer

        # Performance tracking
        self.frame_count = 0
        self.total_detection_time = 0.0

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, LaneMetrics, Optional[float]]:
        """
        Process a single frame through the pipeline.

        PIPELINE PATTERN: Chain of operations
            image → detect → analyze → visualize → result

        Args:
            image: RGB input image from camera/CARLA

        Returns:
            Tuple of (visualization_image, metrics, steering_correction)

        PYTHON KEYWORDS:
            -> Tuple[...]: Return type annotation (like Tuple<...> in C++)
            Optional[float]: Can be float or None (like float* in C++)
        """
        # Step 1: Detect lanes
        # detector.detect() returns DetectionResult object
        detection_result = self.detector.detect(image)

        # Track performance
        self.frame_count += 1
        self.total_detection_time += detection_result.processing_time_ms

        # Step 2: Analyze lanes (calculate metrics)
        # analyzer.get_metrics() expects Lane objects (not tuples!)
        metrics = self.analyzer.get_metrics(
            detection_result.left_lane,
            detection_result.right_lane
        )

        # Step 3: Visualize results
        # Draw lanes, HUD, vehicle position on image
        vis_image = self._create_visualization(
            image,
            detection_result,
            metrics
        )

        # Step 4: Return results
        # Returns tuple - easy to unpack in caller
        return vis_image, metrics, None  # Steering computed separately

    def _create_visualization(self,
                              image: np.ndarray,
                              detection: DetectionResult,
                              metrics: LaneMetrics) -> np.ndarray:
        """
        Create complete visualization with lanes, HUD, and alerts.

        PRIVATE METHOD: Starts with _ (convention for internal use)
        Like: private void createVisualization() in C++
        But not enforced - just a convention!

        Args:
            image: Original image
            detection: Detection result with lanes
            metrics: Analysis metrics

        Returns:
            Annotated image
        """
        # Start with original image
        vis_image = image.copy()

        # Draw lanes (if detected)
        if detection.left_lane or detection.right_lane:
            vis_image = self.visualizer.draw_lanes(
                vis_image,
                detection.left_lane,
                detection.right_lane
            )

        # Draw vehicle position indicator
        vis_image = self.visualizer.draw_vehicle_position(
            vis_image,
            metrics.vehicle_center_x,
            metrics.lane_center_x,
            metrics.departure_status
        )

        # Draw HUD with metrics
        vis_image = self.visualizer.draw_hud(
            vis_image,
            metrics.to_dict(),  # Convert to dict for compatibility
            show_steering=False  # Steering drawn separately
        )

        return vis_image

    def get_average_detection_time(self) -> float:
        """
        Get average detection time across all processed frames.

        COMPUTED PROPERTY: Could also use @property decorator
        Returns derived data, not stored data.

        Returns:
            Average detection time in milliseconds
        """
        if self.frame_count == 0:
            return 0.0
        return self.total_detection_time / self.frame_count

    def reset_stats(self):
        """Reset performance statistics."""
        self.frame_count = 0
        self.total_detection_time = 0.0


# =============================================================================
# DESIGN PATTERN: PIPELINE PATTERN
# =============================================================================
"""
The Pipeline Pattern chains operations together.

C++ ANALOGY:
    // Pipeline in C++
    Image result = image;
    result = detector.detect(result);
    result = analyzer.analyze(result);
    result = visualizer.draw(result);

PYTHON VERSION (this class):
    result = detector.detect(image)
    metrics = analyzer.get_metrics(result.lanes)
    vis = visualizer.draw(image, result, metrics)

BENEFITS:
    - Clear flow: input → step1 → step2 → step3 → output
    - Easy to test each step independently
    - Easy to add/remove steps
    - Single responsibility: each component does one thing

COMPOSITION vs INHERITANCE:
    FrameProcessor "HAS A" detector (composition) ✓
    NOT "IS A" detector (inheritance) ✗

    Why? More flexible!
    - Can swap detector at runtime
    - Can change analyzer without changing processor
    - Follows "favor composition over inheritance" principle
"""


if __name__ == "__main__":
    # Example usage
    print("Testing FrameProcessor...")

    from core.config import Config
    from core.factory import DetectorFactory

    # Create components
    config = Config()
    factory = DetectorFactory(config)
    detector = factory.create('cv')
    analyzer = LaneAnalyzer(
        image_width=800,
        image_height=600,
        drift_threshold=config.analyzer.drift_threshold,
        departure_threshold=config.analyzer.departure_threshold
    )
    visualizer = LKASVisualizer(image_width=800, image_height=600)

    # Create processor
    processor = FrameProcessor(detector, analyzer, visualizer)

    # Test with dummy image
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    vis_image, metrics, steering = processor.process(test_image)

    print(f"✓ Processed frame successfully")
    print(f"  Metrics: {metrics.departure_status.value}")
    print(f"  Average detection time: {processor.get_average_detection_time():.2f}ms")
