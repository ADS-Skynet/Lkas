"""
TensorFlow Lite Lane Detection Implementation

This module provides lane detection using TFLite models.
Optimized for embedded devices like Jetson with quantized models.
"""

import sys
import time
import numpy as np
import cv2
from typing import Tuple

from lkas.detection.core.interfaces import LaneDetector
from skynet_common.types.models import Lane, DetectionResult
from skynet_common.config import DLDetectorConfig

# Install global handler to suppress TFLite Delegate.__del__ errors
# This must be done before importing TensorFlow
def _suppress_tflite_delegate_errors(unraisable):
    """Suppress unraisable exceptions from TFLite Delegate cleanup."""
    # Check if this is a Delegate.__del__ error
    exc_str = str(unraisable.exc_value)
    obj_str = str(type(unraisable.object).__name__) if unraisable.object is not None else ""

    if ('Delegate' in obj_str and '_library' in exc_str) or '_library' in exc_str:
        return  # Silently ignore TFLite delegate errors

    # For other exceptions, use default behavior
    if hasattr(sys, '__unraisablehook__'):
        sys.__unraisablehook__(unraisable)

# Install the hook globally
sys.unraisablehook = _suppress_tflite_delegate_errors


class TFLiteLaneDetector(LaneDetector):
    """
    TensorFlow Lite lane detector.

    Loads .tflite models for fast inference on embedded devices.
    Supports quantized models for optimal performance on Jetson.
    """

    def __init__(self,
                 model_path: str | None = None,
                 input_size: Tuple[int, int] = (640, 384),
                 threshold: float = 0.5,
                 config: DLDetectorConfig | None = None):
        """
        Initialize TFLite lane detector.

        Args:
            model_path: Path to .tflite model (local path only)
            input_size: Model input size (width, height)
            threshold: Segmentation threshold
            config: Optional config object (overrides individual params)
        """
        # If config provided, use it
        if config:
            input_size = tuple(config.input_size)
            threshold = config.threshold
            if config.model_path:
                model_path = config.model_path

        # Store parameters
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.interpreter = None

        # Initialization logs
        print(f"✓ Using framework: TensorFlow Lite", flush=True)
        print(f"✓ Model path: {model_path if model_path else 'None'}", flush=True)
        print(f"✓ Input size: {input_size}", flush=True)
        print(f"✓ Threshold: {threshold}", flush=True)

        # Load model
        if model_path:
            self._load_model(model_path)
        else:
            raise ValueError("model_path is required for TFLite detector")

    def _load_model(self, model_path: str):
        """
        Load TFLite model from file.

        Args:
            model_path: Path to .tflite model file
        """
        import sys
        import os
        import tensorflow as tf

        # Suppress TensorFlow/TFLite warnings and errors globally
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        print("=" * 60, flush=True)
        print("🔄 Loading TFLite model...", flush=True)

        try:
            # Try to load with GPU delegate first (for Jetson)
            print(f"   Loading from: {model_path}", flush=True)
            print(f"   Attempting GPU acceleration...", flush=True)

            # Suppress TFLite delegate errors completely
            # This includes both loading errors AND cleanup errors
            import os
            import sys
            import warnings

            # Suppress warnings during delegate operations
            warnings.filterwarnings('ignore')

            # Save original stderr
            original_stderr_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)

            try:
                # Redirect stderr to suppress all delegate-related errors
                os.dup2(devnull_fd, 2)

                delegate_loaded = False

                # Try Edge TPU delegate (Coral devices)
                try:
                    delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
                    self.interpreter = tf.lite.Interpreter(
                        model_path=model_path,
                        experimental_delegates=[delegate]
                    )
                    delegate_loaded = True
                    # Restore stderr to print success message
                    os.dup2(original_stderr_fd, 2)
                    print(f"✓ Using Edge TPU delegate", flush=True)
                except Exception:
                    pass

                if not delegate_loaded:
                    # Try standard GPU delegate (Jetson)
                    try:
                        delegate = tf.lite.experimental.load_delegate(
                            'libtensorflowlite_gpu_delegate.so'
                        )
                        self.interpreter = tf.lite.Interpreter(
                            model_path=model_path,
                            experimental_delegates=[delegate]
                        )
                        delegate_loaded = True
                        # Restore stderr to print success message
                        os.dup2(original_stderr_fd, 2)
                        print(f"✓ Using GPU delegate", flush=True)
                    except Exception:
                        pass

                if not delegate_loaded:
                    # Restore stderr for CPU fallback message
                    os.dup2(original_stderr_fd, 2)
                    print(f"⚠️  GPU delegate unavailable, using CPU", flush=True)
                    # Suppress stderr again for CPU interpreter creation
                    os.dup2(devnull_fd, 2)
                    self.interpreter = tf.lite.Interpreter(model_path=model_path)
                    # Restore stderr
                    os.dup2(original_stderr_fd, 2)

            finally:
                # Force garbage collection while stderr is STILL suppressed
                # This prevents delegate __del__ errors from appearing
                import gc
                os.dup2(devnull_fd, 2)  # Ensure stderr is suppressed
                gc.collect()  # Clean up failed delegates

                # Now restore stderr and cleanup file descriptors
                try:
                    os.dup2(original_stderr_fd, 2)
                except:
                    pass
                try:
                    os.close(devnull_fd)
                    os.close(original_stderr_fd)
                except:
                    pass

            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"✓ Loaded TFLite model successfully!", flush=True)
            print(f"   Input shape: {self.input_details[0]['shape']}", flush=True)
            print(f"   Input dtype: {self.input_details[0]['dtype']}", flush=True)
            print(f"   Output shape: {self.output_details[0]['shape']}", flush=True)
            print(f"   Output dtype: {self.output_details[0]['dtype']}", flush=True)
            print("=" * 60, flush=True)

        except Exception as e:
            print(f"✗ ERROR: Failed to load TFLite model: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: RGB input image

        Returns:
            Preprocessed image ready for model
        """
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        return batched

    def _postprocess(self, prediction: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output to binary mask.

        Args:
            prediction: Model output
            original_size: Original image size (height, width)

        Returns:
            Binary lane mask
        """
        # Remove batch dimension
        mask = prediction.squeeze()

        # Handle multi-channel output (for multi-class lane detection)
        if len(mask.shape) == 3:
            # Take max across channels
            mask = np.max(mask, axis=-1)

        # Apply threshold
        binary_mask = (mask > self.threshold).astype(np.uint8)

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Convert to 0-255 range
        binary_mask = binary_mask * 255

        # Resize to original size
        resized_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))

        return resized_mask

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect lanes using TFLite model.

        Args:
            image: RGB input image

        Returns:
            DetectionResult with lanes and debug image
        """
        start_time = time.time()
        original_size = image.shape[:2]

        # Preprocess
        input_data = self._preprocess(image)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Postprocess
        lane_mask = self._postprocess(prediction, original_size)

        # Extract lane lines from mask
        left_lane, right_lane = self._extract_lane_lines(lane_mask, original_size)

        # Create debug visualization
        debug_image = self._create_debug_image(image, lane_mask, left_lane, right_lane)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Convert tuples to Lane objects
        left_lane_obj = Lane.from_tuple(left_lane) if left_lane else None
        right_lane_obj = Lane.from_tuple(right_lane) if right_lane else None

        return DetectionResult(
            left_lane=left_lane_obj,
            right_lane=right_lane_obj,
            debug_image=debug_image,
            processing_time_ms=processing_time_ms
        )

    def _extract_lane_lines(self, lane_mask: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[Tuple | None, Tuple | None]:
        """Extract left and right lane lines from segmentation mask."""
        height, width = image_shape

        # Define ROI for lane detection (bottom 50% of image)
        y_min = int(height * 0.5)
        y_max = height

        # Split mask into left and right halves
        center_x = width // 2
        left_mask = lane_mask[:, :center_x]
        right_mask = lane_mask[:, center_x:]

        # Extract left lane
        left_lane = self._fit_lane_line(left_mask, y_min, y_max, offset_x=0)

        # Extract right lane
        right_lane = self._fit_lane_line(right_mask, y_min, y_max, offset_x=center_x)

        return left_lane, right_lane

    def _fit_lane_line(self, mask: np.ndarray, y_min: int, y_max: int, offset_x: int = 0) -> Tuple[int, int, int, int] | None:
        """Fit a lane line from a binary mask."""
        # Find lane pixels
        lane_pixels = np.where(mask > 127)

        if len(lane_pixels[0]) < 10:
            return None

        y_coords = lane_pixels[0]
        x_coords = lane_pixels[1]

        # Filter to ROI
        roi_mask = (y_coords >= y_min) & (y_coords <= y_max)
        y_coords = y_coords[roi_mask]
        x_coords = x_coords[roi_mask]

        if len(y_coords) < 10:
            return None

        try:
            # Fit polynomial (degree 1 = line)
            poly = np.polyfit(y_coords, x_coords, 1)

            # Calculate x coordinates for y_min and y_max
            x1 = int(poly[0] * y_max + poly[1]) + offset_x
            x2 = int(poly[0] * y_min + poly[1]) + offset_x

            # Sanity check
            if x1 < 0 or x2 < 0:
                return None

            return (x1, y_max, x2, y_min)
        except:
            return None

    def _create_debug_image(self, image: np.ndarray, lane_mask: np.ndarray,
                           left_lane: Tuple | None = None,
                           right_lane: Tuple | None = None) -> np.ndarray:
        """Create debug visualization."""
        # Create colored overlay for segmentation mask
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = lane_mask  # Green channel

        # Blend mask
        debug_image = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

        # Draw fitted lane lines
        if left_lane:
            cv2.line(debug_image, (left_lane[0], left_lane[1]),
                    (left_lane[2], left_lane[3]), (255, 0, 0), 5)  # Blue

        if right_lane:
            cv2.line(debug_image, (right_lane[0], right_lane[1]),
                    (right_lane[2], right_lane[3]), (0, 0, 255), 5)  # Red

        # Fill lane area if both lanes detected
        if left_lane and right_lane:
            lane_poly = np.array([[
                [left_lane[0], left_lane[1]],
                [left_lane[2], left_lane[3]],
                [right_lane[2], right_lane[3]],
                [right_lane[0], right_lane[1]]
            ]], dtype=np.int32)

            lane_overlay = debug_image.copy()
            cv2.fillPoly(lane_overlay, lane_poly, (0, 255, 255))  # Yellow
            debug_image = cv2.addWeighted(debug_image, 0.8, lane_overlay, 0.2, 0)

        return debug_image

    def get_name(self) -> str:
        """Get detector name."""
        return "Deep Learning (TensorFlow Lite U-Net)"

    def get_parameters(self) -> dict:
        """Get current parameters."""
        return {
            'framework': 'tflite',
            'model_path': self.model_path,
            'input_size': self.input_size,
            'threshold': self.threshold,
        }


if __name__ == "__main__":
    # Example usage
    print("Testing TFLite Lane Detector...")
    print("Use this detector by setting framework='tflite' in config.yaml")
