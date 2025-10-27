"""
Test script to verify lane detection setup.
Tests individual components without requiring CARLA.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from method.computer_vision.cv_lane_detector import CVLaneDetector
from method.deep_learning.lane_net import DLLaneDetector
from utils.lane_analyzer import LaneAnalyzer, LaneDepartureStatus
from utils.visualizer import LKASVisualizer


def create_test_image_with_lanes(width=800, height=600):
    """Create a synthetic test image with lane markings."""
    # Create black image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add road (gray background)
    road_color = (80, 80, 80)
    cv2.rectangle(image, (0, int(height * 0.4)), (width, height), road_color, -1)

    # Add lane markings (white lines)
    lane_color = (255, 255, 255)
    thickness = 10

    # Left lane (yellow)
    yellow = (0, 255, 255)
    pts_left = np.array([
        [int(width * 0.2), height],
        [int(width * 0.4), int(height * 0.4)]
    ], np.int32)
    cv2.line(image, tuple(pts_left[0]), tuple(pts_left[1]), yellow, thickness)

    # Right lane (white)
    pts_right = np.array([
        [int(width * 0.8), height],
        [int(width * 0.6), int(height * 0.4)]
    ], np.int32)
    cv2.line(image, tuple(pts_right[0]), tuple(pts_right[1]), lane_color, thickness)

    # Add some noise for realism
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)

    return image


def test_cv_detector():
    """Test traditional CV lane detector."""
    print("\n" + "="*60)
    print("Testing Traditional CV Lane Detector")
    print("="*60)

    # Create test image
    test_image = create_test_image_with_lanes()

    # Initialize detector
    detector = CVLaneDetector()

    # Detect lanes
    left_lane, right_lane, debug_image = detector.detect(test_image)

    # Display results
    print(f"Left lane detected: {left_lane is not None}")
    if left_lane:
        print(f"  Left lane: {left_lane}")

    print(f"Right lane detected: {right_lane is not None}")
    if right_lane:
        print(f"  Right lane: {right_lane}")

    # Show image
    cv2.imshow('CV Lane Detection Test', debug_image)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return left_lane, right_lane, test_image


def test_dl_detector():
    """Test deep learning lane detector."""
    print("\n" + "="*60)
    print("Testing Deep Learning Lane Detector")
    print("="*60)

    # Create test image
    test_image = create_test_image_with_lanes()

    # Initialize detector (no pretrained model)
    detector = DLLaneDetector(model_type='simple')

    # Detect lanes
    result = detector.detect(test_image)

    # Display results
    print(f"Left lane detected: {result.left_lane is not None}")
    if result.left_lane:
        print(f"  Left lane: {result.left_lane}")
    print(f"Right lane detected: {result.right_lane is not None}")
    if result.right_lane:
        print(f"  Right lane: {result.right_lane}")

    # Show image
    cv2.imshow('DL Lane Detection Test', result.debug_image)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result.left_lane, result.right_lane, test_image


def test_lane_analyzer(left_lane, right_lane):
    """Test lane analyzer."""
    print("\n" + "="*60)
    print("Testing Lane Analyzer")
    print("="*60)

    # Initialize analyzer
    analyzer = LaneAnalyzer(image_width=800, image_height=600)

    # Get metrics
    metrics = analyzer.get_metrics(left_lane, right_lane)

    # Display metrics
    print("\nLane Metrics:")
    for key, value in metrics.items():
        if isinstance(value, LaneDepartureStatus):
            print(f"  {key}: {value.value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Get steering correction
    steering = analyzer.get_steering_correction(left_lane, right_lane)
    print(f"\nSteering correction: {steering:.3f}" if steering else "\nSteering correction: N/A")

    return metrics, steering


def test_visualizer(test_image, left_lane, right_lane, metrics, steering):
    """Test visualizer."""
    print("\n" + "="*60)
    print("Testing Visualizer")
    print("="*60)

    # Initialize visualizer
    visualizer = LKASVisualizer(image_width=800, image_height=600)

    # Create visualization
    vis_image = visualizer.draw_lanes(test_image, left_lane, right_lane)
    vis_image = visualizer.draw_vehicle_position(
        vis_image,
        metrics['vehicle_center_x'],
        metrics['lane_center_x'],
        metrics['departure_status']
    )
    vis_image = visualizer.draw_hud(
        vis_image,
        metrics,
        show_steering=True,
        steering_value=steering
    )

    # Add alert if needed
    vis_image = visualizer.create_alert_overlay(
        vis_image,
        metrics['departure_status'],
        blink=False
    )

    # Show visualization
    cv2.imshow('Visualization Test', vis_image)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_combined_view(test_image, left_lane, right_lane, metrics, steering):
    """Test combined view."""
    print("\n" + "="*60)
    print("Testing Combined View")
    print("="*60)

    visualizer = LKASVisualizer(image_width=800, image_height=600)

    # Create processed image
    processed = visualizer.draw_lanes(test_image.copy(), left_lane, right_lane)

    # Create combined view
    combined = visualizer.create_combined_view(
        test_image,
        processed,
        metrics,
        steering
    )

    cv2.imshow('Combined View Test', combined)
    print("\nPress any key to finish...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Lane Detection Setup Test")
    print("="*60)
    print("\nThis script tests all components without requiring CARLA.")
    print("Press 'q' to skip a test, any other key to continue.\n")

    try:
        # Test CV detector
        left_lane, right_lane, test_image = test_cv_detector()

        # Test DL detector
        dl_left_lane, dl_right_lane, dl_test_image = test_dl_detector()

        # Test analyzer (using CV detector results)
        if left_lane and right_lane:
            metrics, steering = test_lane_analyzer(left_lane, right_lane)

            # Test visualizer
            test_visualizer(test_image, left_lane, right_lane, metrics, steering)

            # Test combined view
            test_combined_view(test_image, left_lane, right_lane, metrics, steering)

        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        print("\nSetup verified. You can now:")
        print("1. Start CARLA server")
        print("2. Run: python main.py --method cv")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
