"""
Computer Vision Lane Detection (OOP Version)

This implementation uses the OOP architecture.
Implements the LaneDetector interface and uses type-safe data models.

KEY PYTHON CLASS KEYWORDS EXPLAINED:
- class ClassName(BaseClass): Inheritance - this class inherits from BaseClass
- def __init__(self, ...): Constructor - runs when creating new instance
- self: Reference to current instance (like 'this' in C++)
- @property: Makes method accessible like attribute (getter)
- super(): Calls parent class methods
- Optional[Type]: Type can be None
- -> Type: Return type annotation
"""

import cv2
import numpy as np
import time
from typing import Tuple, List

from detection.core.interfaces import LaneDetector  # ABC = Abstract Base Class
from detection.core.models import Lane, DetectionResult  # Type-safe data models
from detection.core.config import CVDetectorConfig  # Configuration


class CVLaneDetector(LaneDetector):
    """
    Computer Vision lane detector.

    INHERITANCE: This class inherits from LaneDetector (ABC)
    - Must implement all @abstractmethod methods from parent
    - In C++: class CVLaneDetector : public LaneDetector
    """

    def __init__(self,
                 roi_vertices: List[Tuple[int, int]] | None = None,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 hough_rho: int = 2,
                 hough_theta: float = np.pi / 180,
                 hough_threshold: int = 50,
                 hough_min_line_len: int = 40,
                 hough_max_line_gap: int = 100,
                 smoothing_factor: float = 0.7,
                 config: CVDetectorConfig | None = None):
        """
        Constructor (initializer).

        __init__: Special method called when creating new object
        self: Reference to this instance (like 'this' in C++)

        Usage:
            detector = CVLaneDetector()  # Calls __init__

        Args:
            roi_vertices: Region of interest vertices
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            hough_rho: Distance resolution for Hough transform
            hough_theta: Angle resolution for Hough transform
            hough_threshold: Minimum votes for Hough line detection
            hough_min_line_len: Minimum line length
            hough_max_line_gap: Maximum gap between line segments
            smoothing_factor: Temporal smoothing factor [0, 1]
            config: Optional configuration object (overrides individual params)
        """
        # If config provided, use it (config takes priority)
        if config:
            canny_low = config.canny_low
            canny_high = config.canny_high
            hough_threshold = config.hough_threshold
            hough_min_line_len = config.hough_min_line_len
            hough_max_line_gap = config.hough_max_line_gap
            smoothing_factor = config.smoothing_factor

        # Store parameters as instance variables
        # self.variable: Instance variable (like member variable in C++)
        self.roi_vertices = roi_vertices
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.smoothing_factor = smoothing_factor

        # Lane tracking state
        self.prev_left_lane: Lane | None = None  # Type hint: Lane | None
        self.prev_right_lane: Lane | None = None
        self.frame_count = 0  # Track number of detections for adaptive smoothing

    # =========================================================================
    # IMPLEMENTING ABSTRACT INTERFACE METHODS
    # These methods are required by LaneDetector ABC
    # =========================================================================

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect lanes in the image.

        IMPLEMENTS: Abstract method from LaneDetector interface
        -> DetectionResult: Return type annotation (for IDE autocomplete)

        Args:
            image: RGB input image

        Returns:
            DetectionResult object containing lanes and debug image
        """
        start_time = time.time()

        height, width = image.shape[:2]

        # Preprocess
        preprocessed = self._preprocess_image(image)

        # Edge detection
        edges = self._detect_edges(preprocessed)

        # Apply ROI
        roi_vertices = self.roi_vertices if self.roi_vertices else self._get_default_roi((height, width))
        roi_edges = self._region_of_interest(edges, roi_vertices)

        # Detect lines
        lines = self._detect_lines(roi_edges)

        # Separate left and right lanes
        left_lines, right_lines = self._separate_lanes(lines, width)

        # Average and smooth lanes (broader recognition area)
        y_min = int(height * 0.5)  # Look at bottom 50% (was 0.6 = 40%)
        y_max = height

        left_lane_tuple = self._average_lane_lines(left_lines, y_min, y_max)
        right_lane_tuple = self._average_lane_lines(right_lines, y_min, y_max)

        # Convert tuples to Lane objects (NEW: type-safe!)
        left_lane = Lane.from_tuple(left_lane_tuple) if left_lane_tuple else None
        right_lane = Lane.from_tuple(right_lane_tuple) if right_lane_tuple else None

        # Increment frame count for adaptive smoothing
        self.frame_count += 1

        # Apply temporal smoothing with adaptive factor
        left_lane = self._smooth_lane_adaptive(left_lane, self.prev_left_lane)
        right_lane = self._smooth_lane_adaptive(right_lane, self.prev_right_lane)

        # Update previous lanes
        self.prev_left_lane = left_lane
        self.prev_right_lane = right_lane

        # Create debug image
        debug_image = self._create_debug_image(image, left_lane, right_lane, roi_vertices)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Return DetectionResult object (NEW: type-safe structure!)
        return DetectionResult(
            left_lane=left_lane,
            right_lane=right_lane,
            debug_image=debug_image,
            processing_time_ms=processing_time_ms
        )

    def get_name(self) -> str:
        """
        Get detector name.

        IMPLEMENTS: Abstract method from LaneDetector interface
        """
        return "Computer Vision (Canny + Hough Transform)"

    def get_parameters(self) -> dict:
        """
        Get current detector parameters.

        IMPLEMENTS: Abstract method from LaneDetector interface
        """
        return {
            'canny_low': self.canny_low,
            'canny_high': self.canny_high,
            'hough_threshold': self.hough_threshold,
            'hough_min_line_len': self.hough_min_line_len,
            'hough_max_line_gap': self.hough_max_line_gap,
            'smoothing_factor': self.smoothing_factor,
        }

    def reset_smoothing(self):
        """
        Reset temporal smoothing state.

        Clears previous lane history, allowing detector to respond
        immediately to new detections without contamination from old data.

        Use cases:
        - After warmup period (clear bad initial detections)
        - When changing lanes
        - When detection confidence is low for extended period
        """
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.frame_count = 0

    # =========================================================================
    # PRIVATE HELPER METHODS
    # These start with _ (Python convention for "private")
    # Not enforced like C++ private, but signals internal use only
    # =========================================================================

    def _get_default_roi(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Get default ROI based on image shape (broader detection area)."""
        height, width = image_shape
        vertices = np.array([[
            (int(width * 0.05), height),           # Bottom-left (wider)
            (int(width * 0.35), int(height * 0.5)),  # Top-left (wider, looks further)
            (int(width * 0.65), int(height * 0.5)),  # Top-right (wider, looks further)
            (int(width * 0.95), height)            # Bottom-right (wider)
        ]], dtype=np.int32)
        return vertices

    def _region_of_interest(self, image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Apply ROI mask to image."""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(image, mask)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image (grayscale + blur)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny."""
        return cv2.Canny(image, self.canny_low, self.canny_high)

    def _detect_lines(self, edge_image: np.ndarray) -> np.ndarray | None:
        """Detect lines using Hough transform."""
        lines = cv2.HoughLinesP(
            edge_image,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap
        )
        return lines

    def _separate_lanes(self, lines: np.ndarray, image_width: int) -> Tuple[List, List]:
        """Separate detected lines into left and right lanes."""
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:  # Vertical line
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope and position
            if abs(slope) < 0.5:  # Too horizontal
                continue

            # Separate left and right based on slope and position
            if slope < 0 and x1 < image_width / 2:  # Left lane
                left_lines.append(line[0])
            elif slope > 0 and x1 > image_width / 2:  # Right lane
                right_lines.append(line[0])

        return left_lines, right_lines

    def _average_lane_lines(self, lines: List, y_min: int, y_max: int) -> Tuple[int, int, int, int] | None:
        """Average multiple line segments into a single lane line."""
        if not lines:
            return None

        # Fit a linear polynomial to all points
        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return None

        # Fit line
        poly = np.polyfit(y_coords, x_coords, 1)

        # Calculate x coordinates for y_min and y_max
        x1 = int(poly[0] * y_max + poly[1])
        x2 = int(poly[0] * y_min + poly[1])

        return (x1, y_max, x2, y_min)

    def _smooth_lane(self, current_lane: Lane | None,
                     previous_lane: Lane | None) -> Lane | None:
        """
        Smooth lane detection using exponential moving average.

        UPDATED: Now works with Lane objects instead of tuples!
        """
        if current_lane is None:
            return previous_lane

        if previous_lane is None:
            return current_lane

        # Apply smoothing to each coordinate
        x1 = int(self.smoothing_factor * current_lane.x1 + (1 - self.smoothing_factor) * previous_lane.x1)
        y1 = int(self.smoothing_factor * current_lane.y1 + (1 - self.smoothing_factor) * previous_lane.y1)
        x2 = int(self.smoothing_factor * current_lane.x2 + (1 - self.smoothing_factor) * previous_lane.x2)
        y2 = int(self.smoothing_factor * current_lane.y2 + (1 - self.smoothing_factor) * previous_lane.y2)

        # Create new Lane object with smoothed coordinates
        return Lane(x1=x1, y1=y1, x2=x2, y2=y2, confidence=current_lane.confidence)

    def _smooth_lane_adaptive(self, current_lane: Lane | None,
                              previous_lane: Lane | None) -> Lane | None:
        """
        Adaptive temporal smoothing that reduces smoothing during startup.

        During the first ~50 frames, uses lower smoothing factor to allow
        detector to respond quickly and not be contaminated by bad initial
        detections from spawn point.

        Smoothing schedule:
        - Frames 0-20:   factor = 0.95 (95% new, 5% old) - Very responsive
        - Frames 21-50:  factor = 0.80 (80% new, 20% old) - Medium smoothing
        - Frames 51+:    factor = 0.70 (70% new, 30% old) - Full smoothing (configured value)

        This matches the warmup period in main.py!
        """
        if current_lane is None:
            return previous_lane

        if previous_lane is None:
            return current_lane

        # Determine adaptive smoothing factor based on frame count
        if self.frame_count <= 20:
            # Early frames: Minimal smoothing - respond quickly
            adaptive_factor = 0.95
        elif self.frame_count <= 50:
            # Warmup period: Medium smoothing
            adaptive_factor = 0.80
        else:
            # After warmup: Full smoothing (use configured value)
            adaptive_factor = self.smoothing_factor

        # Apply smoothing to each coordinate
        x1 = int(adaptive_factor * current_lane.x1 + (1 - adaptive_factor) * previous_lane.x1)
        y1 = int(adaptive_factor * current_lane.y1 + (1 - adaptive_factor) * previous_lane.y1)
        x2 = int(adaptive_factor * current_lane.x2 + (1 - adaptive_factor) * previous_lane.x2)
        y2 = int(adaptive_factor * current_lane.y2 + (1 - adaptive_factor) * previous_lane.y2)

        # Create new Lane object with smoothed coordinates
        return Lane(x1=x1, y1=y1, x2=x2, y2=y2, confidence=current_lane.confidence)

    def _create_debug_image(self, image: np.ndarray,
                           left_lane: Lane | None,
                           right_lane: Lane | None,
                           roi_vertices: np.ndarray) -> np.ndarray:
        """
        Create debug visualization image.

        UPDATED: Now works with Lane objects!
        """
        debug_img = image.copy()

        # Draw ROI
        cv2.polylines(debug_img, roi_vertices, True, (0, 255, 0), 2)

        # Draw lane lines (using Lane object properties)
        if left_lane:
            cv2.line(debug_img, (left_lane.x1, left_lane.y1),
                    (left_lane.x2, left_lane.y2), (255, 0, 0), 5)

        if right_lane:
            cv2.line(debug_img, (right_lane.x1, right_lane.y1),
                    (right_lane.x2, right_lane.y2), (0, 0, 255), 5)

        # Fill lane area
        if left_lane and right_lane:
            lane_poly = np.array([[
                [left_lane.x1, left_lane.y1],
                [left_lane.x2, left_lane.y2],
                [right_lane.x2, right_lane.y2],
                [right_lane.x1, right_lane.y1]
            ]], dtype=np.int32)

            overlay = debug_img.copy()
            cv2.fillPoly(overlay, lane_poly, (0, 255, 0))
            debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)

        return debug_img


# =============================================================================
# PYTHON CLASS KEYWORDS SUMMARY (for C++ developers)
# =============================================================================
"""
class ClassName(BaseClass):
    - Defines a new class inheriting from BaseClass
    - Like: class ClassName : public BaseClass in C++

def __init__(self, arg1, arg2):
    - Constructor, called when creating instance
    - Like: ClassName(arg1, arg2) constructor in C++
    - self is like 'this' pointer

self:
    - Reference to current instance
    - Like 'this' in C++
    - MUST be first parameter of instance methods

self.variable:
    - Instance variable (member variable)
    - Like: this->variable in C++

def method(self, arg):
    - Instance method
    - Like: void method(arg) in C++ class
    - Needs 'self' to access instance variables

@property:
    - Decorator to make method look like attribute
    - Like: getter without parentheses
    - Example: lane.slope instead of lane.slope()

@staticmethod:
    - Static method (no self needed)
    - Like: static method in C++

@classmethod:
    - Class method (receives class, not instance)
    - Like: static method with access to class

Optional[Type]:
    - Type can be None
    - Like: Type* in C++ (nullable pointer)

-> ReturnType:
    - Return type annotation
    - Like: ReturnType method() in C++
    - Not enforced at runtime, but IDE uses it

_method_name:
    - Convention for "private" method
    - NOT enforced (Python has no true private)
    - Like: private: in C++ (but just convention)

__method_name:
    - Name mangling (harder to access from outside)
    - Closest to C++ private
    - Still accessible but discouraged

super():
    - Calls parent class method
    - Like: BaseClass::method() in C++
"""


if __name__ == "__main__":
    # Example usage
    print("Testing CV Detector...")

    detector = CVLaneDetector()
    print(f"Detector name: {detector.get_name()}")
    print(f"Parameters: {detector.get_parameters()}")

    # Test with dummy image
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)
    result = detector.detect(test_image)

    print(f"\nDetection result:")
    print(f"  Left lane: {result.left_lane}")
    print(f"  Right lane: {result.right_lane}")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Has both lanes: {result.has_both_lanes}")
