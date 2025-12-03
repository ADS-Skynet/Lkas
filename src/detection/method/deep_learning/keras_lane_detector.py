"""
Keras/TensorFlow Lane Detection Implementation

This module provides lane detection using Keras/TensorFlow models.
Supports loading .keras models from local paths or Hugging Face Hub.
"""

import time
import numpy as np
import cv2
from typing import Tuple

from lkas.detection.core.interfaces import LaneDetector
from lkas.detection.core.models import Lane, DetectionResult
from skynet_common.config import DLDetectorConfig


class KerasLaneDetector(LaneDetector):
    """
    Keras/TensorFlow-based lane detector.

    Loads .keras models and performs lane detection using TensorFlow.
    Compatible with models trained using Keras/TensorFlow framework.
    """

    def __init__(self,
                 model_path: str | None = None,
                 input_size: Tuple[int, int] = (224, 224),
                 threshold: float = 0.5,
                 config: DLDetectorConfig | None = None):
        """
        Initialize Keras lane detector.

        Args:
            model_path: Path to .keras model (local or HuggingFace repo)
            input_size: Model input size (height, width)
            threshold: Segmentation threshold
            config: Optional config object (overrides individual params)
        """
        # If config provided, use it
        if config:
            input_size = config.input_size
            threshold = config.threshold
            if config.model_path:
                model_path = config.model_path

        # Store parameters
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.model = None

        # Initialization logs
        print(f"✓ Using framework: Keras/TensorFlow", flush=True)
        print(f"✓ Model path: {model_path if model_path else 'None'}", flush=True)
        print(f"✓ Input size: {input_size}", flush=True)
        print(f"✓ Threshold: {threshold}", flush=True)

        # Load model
        if model_path:
            self._load_model(model_path)
        else:
            raise ValueError("model_path is required for Keras detector")

    def _load_model(self, model_path: str):
        """
        Load Keras model from file or Hugging Face Hub.

        Args:
            model_path: Path to model or HF Hub model ID
        """
        import sys
        import tensorflow as tf

        # Configure GPU memory growth to avoid OOM errors on Jetson
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Enabled memory growth for {len(gpus)} GPU(s)", flush=True)
            except RuntimeError as e:
                print(f"⚠️  Could not set memory growth: {e}", flush=True)

        print("=" * 60, flush=True)
        print("🔄 Loading Keras model...", flush=True)

        try:
            # Check if it's a Hugging Face Hub model
            if "/" in model_path and not model_path.startswith("/"):
                # Remove hf:// prefix if present
                hf_model_id = model_path.replace("hf://", "")

                print("🤗 Hugging Face Hub model detected!", flush=True)
                print(f"   Repository: {hf_model_id}", flush=True)
                print(f"   Attempting to download...", flush=True)

                try:
                    from huggingface_hub import hf_hub_download

                    # Try to find .keras file
                    filenames = ["best_model.keras", "model.keras", "lane_model.keras"]
                    print(f"   Trying filenames: {', '.join(filenames)}", flush=True)

                    found_file = None
                    for filename in filenames:
                        try:
                            print(f"   → Trying {filename}...", end=" ", flush=True)
                            local_path = hf_hub_download(
                                repo_id=hf_model_id,
                                filename=filename,
                                cache_dir=".cache/huggingface"
                            )
                            model_path = local_path
                            found_file = filename
                            print(f"✓ Found!", flush=True)
                            print(f"✓ Downloaded to: {local_path}", flush=True)
                            break
                        except Exception:
                            print(f"✗ Not found", flush=True)
                            continue
                    else:
                        print(f"\n✗ ERROR: Could not find .keras file in {hf_model_id}", file=sys.stderr, flush=True)
                        print(f"   Tried: {', '.join(filenames)}", file=sys.stderr, flush=True)
                        raise FileNotFoundError(f"No .keras file found in {hf_model_id}")

                except ImportError:
                    print("✗ ERROR: huggingface_hub not installed!", file=sys.stderr, flush=True)
                    print("   Install with: pip install huggingface_hub", file=sys.stderr, flush=True)
                    raise

            # Define custom objects (common loss/metric functions for segmentation)
            custom_objects = {
                'dice_loss': self._dice_loss,
                'dice_coefficent': self._dice_coefficient,  # Note: misspelled in some models
                'dice_coefficient': self._dice_coefficient,
                'precision_smooth': self._precision_smooth,
                'recall_smooth': self._recall_smooth,
                'iou': self._iou,
            }

            # Load model without compilation (we only need inference)
            print("   Loading model architecture and weights...", flush=True)
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False  # Skip compilation to avoid custom loss/metric issues
            )

            print(f"✓ Loaded Keras model successfully!", flush=True)
            print(f"   Model type: {type(self.model)}", flush=True)
            print(f"   Input shape: {self.model.input_shape}", flush=True)
            print(f"   Output shape: {self.model.output_shape}", flush=True)
            print(f"   Total layers: {len(self.model.layers)}", flush=True)
            print("=" * 60, flush=True)

        except Exception as e:
            print(f"✗ ERROR: Failed to load Keras model: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    # Custom loss and metric functions
    @staticmethod
    def _dice_loss(y_true, y_pred, smooth=1e-6):
        """Dice loss for segmentation."""
        import tensorflow as tf
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    @staticmethod
    def _dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric."""
        import tensorflow as tf
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    @staticmethod
    def _precision_smooth(y_true, y_pred, smooth=1e-6):
        """Precision metric with smoothing."""
        import tensorflow as tf
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        true_positives = tf.reduce_sum(y_true_f * y_pred_f)
        predicted_positives = tf.reduce_sum(y_pred_f)
        return (true_positives + smooth) / (predicted_positives + smooth)

    @staticmethod
    def _recall_smooth(y_true, y_pred, smooth=1e-6):
        """Recall metric with smoothing."""
        import tensorflow as tf
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        true_positives = tf.reduce_sum(y_true_f * y_pred_f)
        actual_positives = tf.reduce_sum(y_true_f)
        return (true_positives + smooth) / (actual_positives + smooth)

    @staticmethod
    def _iou(y_true, y_pred, smooth=1e-6):
        """Intersection over Union metric."""
        import tensorflow as tf
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: RGB input image

        Returns:
            Preprocessed image ready for model
        """
        # Resize to model input size
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))

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
        Detect lanes using Keras model.

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
        prediction = self.model.predict(input_data, verbose=0)

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
        return "Deep Learning (Keras/TensorFlow U-Net)"

    def get_parameters(self) -> dict:
        """Get current parameters."""
        return {
            'framework': 'keras',
            'model_path': self.model_path,
            'input_size': self.input_size,
            'threshold': self.threshold,
        }


if __name__ == "__main__":
    # Example usage
    print("Testing Keras Lane Detector...")

    # This would load from HuggingFace
    # detector = KerasLaneDetector(
    #     model_path="Tilak1812/lane-detection-unet-tusimple",
    #     input_size=(224, 224),
    #     threshold=0.5
    # )

    print("Keras detector module loaded successfully!")
    print("Use this detector by setting framework='keras' in config.yaml")
