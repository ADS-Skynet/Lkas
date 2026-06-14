"""
Deep Learning Lane Detector using BiSeNet V2

Implements the LaneDetector interface using semantic segmentation
to detect lanes and extract left/right lane lines.
"""

import time
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
import sys

from lkas.detection.core.interfaces import LaneDetector
from common.types.models import Lane, LaneContour, DetectionResult


class DLLaneDetector(LaneDetector):
    """
    Deep Learning lane detector using BiSeNet V2 for semantic segmentation.

    Converts segmentation mask to left/right lane lines compatible with
    the existing decision pipeline (LaneAnalyzer).
    """

    # Project root and default model path — both absolute, CWD-independent
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
    DEFAULT_MODEL_PATH = _PROJECT_ROOT / "lane-detection-dl" / "inference" / "bisenet-0204.pth"

    def __init__(
        self,
        model_type: str = "pretrained",
        model_path: str | None = None,
        input_size: Tuple[int, int] = (512, 1024),  # (height, width)
        threshold: float = 0.5,
        device: str = "auto",
        n_classes: int = 2,
        smoothing_factor: float = 0.7,
        use_fp16: bool = True,  # Use half precision for faster inference
    ):
        """
        Initialize the DL lane detector.

        Args:
            model_type: Type of model ('pretrained', 'binary', 'multi')
            model_path: Path to model weights (None uses default)
            input_size: Model input size as (height, width)
            threshold: Confidence threshold for lane detection
            device: Device to run on ('cpu', 'cuda', 'auto')
            n_classes: Number of segmentation classes (2 for binary)
            smoothing_factor: Temporal smoothing factor [0, 1]
            use_fp16: Use half precision (FP16) for faster inference on GPU
        """
        self.model_type = model_type
        self.input_size = input_size
        self.threshold = threshold
        self.n_classes = n_classes
        self.smoothing_factor = smoothing_factor

        # Resolve model path — relative paths are anchored to project root
        if model_path:
            p = Path(model_path)
            self.model_path = p if p.is_absolute() else (self._PROJECT_ROOT / p).resolve()
        else:
            self.model_path = self.DEFAULT_MODEL_PATH

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # FP16 only works on CUDA
        # self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.use_fp16 = False

        # Load model
        self._load_model()

        # Temporal smoothing state
        self._prev_left_lane: Optional[Lane] = None
        self._prev_right_lane: Optional[Lane] = None
        self._frame_count = 0
        self._warmup_frames = 30

        # Cache for visualization
        self._last_mask: Optional[np.ndarray] = None

        # ImageNet normalization (as torch tensors for faster processing)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if self.use_fp16:
            self._mean = self._mean.half()
            self._std = self._std.half()

        # Pre-allocate input tensor for reuse (avoids allocation each frame)
        self._input_tensor = torch.empty(
            (1, 3, input_size[0], input_size[1]),
            dtype=torch.float16 if self.use_fp16 else torch.float32,
            device=self.device
        )

        print(f"[DLLaneDetector] Initialized on {self.device}")
        print(f"  Model: {self.model_path.name}")
        print(f"  Input size: {input_size[1]}x{input_size[0]}")
        print(f"  Classes: {n_classes}")
        print(f"  FP16: {self.use_fp16}")

    def _load_model(self):
        """Load BiSeNet V2 model."""
        # Add lane-detection-dl to path for model import
        model_dir = self.model_path.parent.parent / "model"
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from bisenetv2 import BiSeNetV2

        print(f"[DLLaneDetector] Loading model from: {self.model_path}")

        # Create model
        self.model = BiSeNetV2(n_classes=self.n_classes, aux_mode='eval')

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Convert to FP16 for faster inference
        if self.use_fp16:
            self.model = self.model.half()
            print(f"[DLLaneDetector] Model converted to FP16")

        print(f"[DLLaneDetector] Model loaded successfully")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect lanes in the given image.

        Args:
            image: RGB input image as numpy array (H, W, 3)

        Returns:
            DetectionResult with left/right Lane objects and debug image
        """
        start_time = time.time()
        self._frame_count += 1

        orig_h, orig_w = image.shape[:2]

        # Preprocess
        img_tensor = self._preprocess(image)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            logits = outputs[0]  # (1, C, H, W)

            # CRITICAL: Convert to FP32 before argmax to avoid FP16 precision issues
            # FP16 can cause small probability values to become 0, breaking argmax
            if self.use_fp16:
                logits = logits.float()

        # Debug: Log model output stats once
        if not hasattr(self, '_model_output_debug_logged'):
            # print(f"[DL Debug] logits shape={logits.shape}, dtype={logits.dtype}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
            # Check class probabilities
            probs = torch.softmax(logits, dim=1)
            # print(f"[DL Debug] probs: class0 max={probs[0,0].max().item():.4f}, class1 max={probs[0,1].max().item():.4f}")
            self._model_output_debug_logged = True

        # Get prediction mask
        pred = logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)

        # Debug: Log prediction mask stats once
        if not hasattr(self, '_pred_debug_logged'):
            nonzero = np.count_nonzero(pred)
            # print(f"[DL Debug] pred shape={pred.shape}, nonzero={nonzero}, unique={np.unique(pred)}")
            self._pred_debug_logged = True

        # Resize mask back to original size
        mask = cv2.resize(
            pred.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Store mask for visualization
        self._last_mask = mask

        # Extract multiple lane contours from mask (for DL)
        lane_contours = self._extract_lane_contours(mask)

        # Extract lane lines from mask (for backwards compatibility with decision module)
        left_lane, right_lane = self._extract_lanes_from_mask(mask)

        # Apply temporal smoothing
        left_lane = self._smooth_lane(left_lane, self._prev_left_lane)
        right_lane = self._smooth_lane(right_lane, self._prev_right_lane)

        # Update previous lanes
        self._prev_left_lane = left_lane
        self._prev_right_lane = right_lane

        # Create debug image with lane overlay
        debug_image = self._create_debug_image(image, mask, left_lane, right_lane)

        processing_time_ms = (time.time() - start_time) * 1000

        return DetectionResult(
            left_lane=left_lane,
            right_lane=right_lane,
            debug_image=debug_image,
            processing_time_ms=processing_time_ms,
            lanes=lane_contours
        )

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input (optimized)."""
        # Resize to model input size
        img_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # Convert to tensor directly on GPU (faster than numpy normalization)
        # HWC -> CHW and normalize in one step
        img_tensor = torch.from_numpy(img_resized).to(self.device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW

        # Convert to FP16 or FP32 and normalize
        if self.use_fp16:
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()

        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - self._mean) / self._std

        return img_tensor

    def _extract_lane_contours(self, mask: np.ndarray) -> list:
        """
        Extract multiple lane contours from segmentation mask.

        Args:
            mask: Segmentation mask (H, W) with class indices

        Returns:
            List of LaneContour objects
        """
        lanes = []

        # Process each class (skip background class 0)
        for cls in range(1, self.n_classes):
            # Create binary mask for this class
            class_mask = (mask == cls).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(
                class_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Filter small contours
                if len(contour) < 10:
                    continue

                # Simplify contour using Douglas-Peucker algorithm
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Reshape to list of [x, y] points
                points = approx.reshape(-1, 2).tolist()

                # Calculate confidence based on contour area
                area = cv2.contourArea(contour)
                confidence = min(1.0, area / 5000.0)  # Normalize by typical lane area

                lanes.append(LaneContour(
                    points=points,
                    class_id=cls,
                    confidence=confidence
                ))

        return lanes

    def _extract_lanes_from_mask(
        self, mask: np.ndarray
    ) -> Tuple[Optional[Lane], Optional[Lane]]:
        """
        Extract left and right lane lines from segmentation mask.

        Uses edge detection on the mask to find lane boundaries,
        then fits lines to the left and right edges.
        """
        h, w = mask.shape

        # Create binary mask for all lane pixels
        lane_mask = (mask > 0).astype(np.uint8) * 255

        # Find lane pixels
        lane_pixels = np.where(lane_mask > 0)
        if len(lane_pixels[0]) < 100:  # Not enough lane pixels
            return None, None

        # Define ROI: focus on bottom 60% of image
        roi_top = int(h * 0.4)

        # For each row in ROI, find leftmost and rightmost lane pixels
        left_points = []
        right_points = []
        center_x = w // 2

        for y in range(roi_top, h, 3):  # Sample every 3 rows
            row = lane_mask[y, :]
            lane_x = np.where(row > 0)[0]

            if len(lane_x) < 2:
                continue

            # Find leftmost and rightmost points
            leftmost = lane_x.min()
            rightmost = lane_x.max()

            # Classify as left or right lane based on position
            if leftmost < center_x:
                left_points.append((leftmost, y))
            if rightmost > center_x:
                right_points.append((rightmost, y))

        # Fit lines to points
        left_lane = self._fit_lane_line(left_points, h, roi_top, is_left=True)
        right_lane = self._fit_lane_line(right_points, h, roi_top, is_left=False)

        return left_lane, right_lane

    def _fit_lane_line(
        self,
        points: list,
        img_height: int,
        roi_top: int,
        is_left: bool
    ) -> Optional[Lane]:
        """Fit a line to lane boundary points."""
        if len(points) < 5:
            return None

        points_arr = np.array(points)
        x_vals = points_arr[:, 0]
        y_vals = points_arr[:, 1]

        # Fit polynomial (linear)
        try:
            coeffs = np.polyfit(y_vals, x_vals, 1)
        except np.RankWarning:
            return None

        # Calculate endpoints
        y1 = img_height - 1  # Bottom
        y2 = roi_top  # Top of ROI

        x1 = int(np.clip(np.polyval(coeffs, y1), 0, img_height * 2))
        x2 = int(np.clip(np.polyval(coeffs, y2), 0, img_height * 2))

        # Confidence based on number of points
        confidence = min(1.0, len(points) / 50.0)

        return Lane(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)

    def _smooth_lane(
        self,
        current: Optional[Lane],
        previous: Optional[Lane]
    ) -> Optional[Lane]:
        """Apply temporal smoothing to lane detection."""
        if current is None:
            return previous  # Keep previous if no detection

        if previous is None:
            return current  # First detection

        # Adaptive smoothing factor during warmup
        if self._frame_count < self._warmup_frames:
            factor = self.smoothing_factor * (self._frame_count / self._warmup_frames)
        else:
            factor = self.smoothing_factor

        # Exponential moving average
        x1 = int(factor * current.x1 + (1 - factor) * previous.x1)
        y1 = int(factor * current.y1 + (1 - factor) * previous.y1)
        x2 = int(factor * current.x2 + (1 - factor) * previous.x2)
        y2 = int(factor * current.y2 + (1 - factor) * previous.y2)
        confidence = factor * current.confidence + (1 - factor) * previous.confidence

        return Lane(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)

    def _create_debug_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        left_lane: Optional[Lane],
        right_lane: Optional[Lane]
    ) -> np.ndarray:
        """Create visualization with segmentation overlay and lane lines."""
        output = image.copy()

        # Create colored overlay for segmentation
        overlay = np.zeros_like(output)

        # Color lane pixels (transparent blue)
        lane_pixels = mask > 0
        overlay[lane_pixels] = [70, 130, 255]  # Light blue in RGB

        # Blend overlay with image
        alpha = 0.35
        output = cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)

        # Draw lane lines
        if left_lane and right_lane:
            # Fill lane area
            pts = np.array([
                [left_lane.x1, left_lane.y1],
                [left_lane.x2, left_lane.y2],
                [right_lane.x2, right_lane.y2],
                [right_lane.x1, right_lane.y1]
            ], np.int32)

            lane_overlay = output.copy()
            cv2.fillPoly(lane_overlay, [pts], (0, 255, 0))  # Green fill
            output = cv2.addWeighted(output, 0.7, lane_overlay, 0.3, 0)

        # Draw individual lane lines
        if left_lane:
            cv2.line(output,
                    (left_lane.x1, left_lane.y1),
                    (left_lane.x2, left_lane.y2),
                    (255, 0, 0), 3)  # Blue (RGB)

        if right_lane:
            cv2.line(output,
                    (right_lane.x1, right_lane.y1),
                    (right_lane.x2, right_lane.y2),
                    (0, 0, 255), 3)  # Red (RGB)

        return output

    def get_name(self) -> str:
        """Return the name of this detector."""
        return f"DL-BiSeNet-{self.model_type}"

    def get_parameters(self) -> dict:
        """Return current detector parameters."""
        return {
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "input_size": self.input_size,
            "threshold": self.threshold,
            "device": str(self.device),
            "n_classes": self.n_classes,
            "smoothing_factor": self.smoothing_factor,
        }

    def get_last_mask(self) -> Optional[np.ndarray]:
        """
        Get the last segmentation mask for visualization.

        Returns:
            Segmentation mask (H, W) with class indices, or None
        """
        return self._last_mask

    def update_parameter(self, name: str, value: float) -> bool:
        """
        Update a detector parameter at runtime.

        Args:
            name: Parameter name
            value: New value

        Returns:
            True if parameter was updated
        """
        if name == "threshold":
            self.threshold = value
            return True
        elif name == "smoothing_factor":
            self.smoothing_factor = max(0.0, min(1.0, value))
            return True
        return False
