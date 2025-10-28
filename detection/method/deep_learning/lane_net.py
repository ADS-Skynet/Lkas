"""
Deep Learning Lane Detection (OOP Version)

This implementation uses the OOP architecture.
Wraps the existing base DL detector to implement the new interface.

For beginners: This shows how to adapt existing code to new interfaces!
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np
from typing import Optional, Tuple

from core.interfaces import LaneDetector
from core.models import Lane, DetectionResult
from core.config import DLDetectorConfig

# Import the base DL detector implementation
from method.deep_learning.lane_net_base import DLLaneDetector as _BaseDLLaneDetector


class DLLaneDetector(LaneDetector):
    """
    Deep Learning lane detector.

    ADAPTER PATTERN: Wraps existing DLLaneDetector to implement new interface
    - Uses composition instead of modifying original code
    - Original code returns (left, right, debug) tuple
    - We adapt it to return DetectionResult object
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = 'pretrained',
                 device: str = 'auto',
                 input_size: Tuple[int, int] = (256, 256),
                 threshold: float = 0.5,
                 config: Optional[DLDetectorConfig] = None):
        """
        Initialize DL detector.

        Args:
            model_path: Path to pretrained model weights
            model_type: 'pretrained', 'simple', or 'full'
            device: 'cpu', 'cuda', or 'auto'
            input_size: Model input size
            threshold: Segmentation threshold
            config: Optional config object (overrides individual params)
        """
        # If config provided, use it
        if config:
            model_type = config.model_type
            input_size = config.input_size
            threshold = config.threshold
            device = config.device

        # Store parameters
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.input_size = input_size
        self.threshold = threshold

        # Create the base detector internally
        # COMPOSITION: We "have a" detector, not "are a" detector
        self._detector = _BaseDLLaneDetector(
            model_path=model_path,
            model_type=model_type,
            device=device,
            input_size=input_size,
            threshold=threshold
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect lanes using deep learning.

        IMPLEMENTS: LaneDetector.detect() interface

        Args:
            image: RGB input image

        Returns:
            DetectionResult with lanes and debug image
        """
        start_time = time.time()

        # Call original detector (returns tuple format)
        left_lane_tuple, right_lane_tuple, debug_image = self._detector.detect(image)

        # Convert tuples to Lane objects (if they exist)
        left_lane = Lane.from_tuple(left_lane_tuple) if left_lane_tuple else None
        right_lane = Lane.from_tuple(right_lane_tuple) if right_lane_tuple else None

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Return DetectionResult (new format!)
        return DetectionResult(
            left_lane=left_lane,
            right_lane=right_lane,
            debug_image=debug_image,
            processing_time_ms=processing_time_ms
        )

    def get_name(self) -> str:
        """Get detector name."""
        return f"Deep Learning ({self.model_type} U-Net)"

    def get_parameters(self) -> dict:
        """Get current parameters."""
        return {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'threshold': self.threshold,
            'device': self.device,
            'model_path': self.model_path,
        }


# =============================================================================
# DESIGN PATTERN: ADAPTER
# =============================================================================
"""
The Adapter Pattern allows incompatible interfaces to work together.

ANALOGY (C++):
    // Old interface
    class OldDetector {
        tuple<int,int,int,int> detect(Image img);  // Returns tuple
    };

    // New interface
    class LaneDetector {
        virtual DetectionResult detect(Image img) = 0;  // Returns object
    };

    // Adapter: Makes old detector work with new interface
    class DLLaneDetector : public LaneDetector {
        OldDetector* detector;  // Composition

        DetectionResult detect(Image img) override {
            auto tuple = detector->detect(img);  // Call old
            return convert_to_result(tuple);      // Convert format
        }
    };

PYTHON VERSION:
    We do the same thing, but with Python syntax!
    - Old detector returns tuple
    - New interface requires DetectionResult
    - Adapter converts between them

BENEFIT:
    - Don't modify original code (keeps it working)
    - Gradually migrate to new system
    - Both old and new code work together
"""


if __name__ == "__main__":
    # Example usage
    print("Testing DL Detector...")

    detector = DLLaneDetector(model_type='pretrained')
    print(f"Detector name: {detector.get_name()}")
    print(f"Parameters: {detector.get_parameters()}")

    # Test with dummy image
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    result = detector.detect(test_image)

    print(f"\nDetection result:")
    print(f"  Left lane: {result.left_lane}")
    print(f"  Right lane: {result.right_lane}")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Has both lanes: {result.has_both_lanes}")

    if result.left_lane:
        print(f"  Left lane slope: {result.left_lane.slope:.2f}")
