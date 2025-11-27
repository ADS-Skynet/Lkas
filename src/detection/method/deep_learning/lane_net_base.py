"""
Deep Learning Lane Detection Model
CNN-based lane detection using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple

# Import pre-trained segmentation models
try:
    import segmentation_models_pytorch as smp
    SEGMENTATION_MODELS_AVAILABLE = True
except ImportError:
    SEGMENTATION_MODELS_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available. Using custom models only.")


class LaneNet(nn.Module):
    """
    Convolutional Neural Network for lane detection.
    This is a simplified UNet-style architecture for lane segmentation.
    """

    def __init__(self, input_channels: int = 3, output_channels: int = 1):
        """
        Initialize LaneNet.

        Args:
            input_channels: Number of input image channels (3 for RGB)
            output_channels: Number of output channels (1 for binary lane mask)
        """
        super(LaneNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        # Final output layer
        self.out = nn.Conv2d(64, output_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a convolutional block with two conv layers.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            Sequential conv block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Output tensor [batch, output_channels, height, width]
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)
        return torch.sigmoid(out)


class SimpleLaneNet(nn.Module):
    """
    Simplified CNN for faster inference.
    Good for real-time applications.
    """

    def __init__(self, input_channels: int = 3, output_channels: int = 1):
        """
        Initialize SimpleLaneNet.

        Args:
            input_channels: Number of input image channels
            output_channels: Number of output channels
        """
        super(SimpleLaneNet, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # Upsample
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.features(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


class DLLaneDetector:
    """Wrapper class for deep learning-based lane detection."""

    def __init__(self,
                 model_path: str | None = None,
                 model_type: str = 'pretrained',
                 device: str = 'auto',
                 input_size: Tuple[int, int] = (256, 256),
                 threshold: float = 0.5):
        """
        Initialize DL lane detector.

        Args:
            model_path: Path to pretrained model weights (optional for pretrained models)
            model_type: Type of model ('pretrained', 'simple', or 'full')
                       'pretrained' = Use pre-trained U-Net from segmentation_models_pytorch
                       'simple' = Use custom SimpleLaneNet
                       'full' = Use custom LaneNet
            device: Device to run on ('cpu', 'cuda', or 'auto')
            input_size: Input image size (height, width)
            threshold: Threshold for binary segmentation
        """
        self.input_size = input_size
        self.threshold = threshold

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialization logs go to stdout (visible in terminal)
        # Error logs go to stderr (captured in log file)
        print(f"✓ Using device: {self.device}", flush=True)
        print(f"✓ Model type: {model_type}", flush=True)
        print(f"✓ Custom weights: {model_path if model_path else 'None (using default)'}", flush=True)

        # Initialize model
        if model_type == 'pretrained' and SEGMENTATION_MODELS_AVAILABLE:
            # Use pre-trained U-Net with ResNet18 encoder
            print("=" * 60, flush=True)
            print("📦 Loading pre-trained U-Net architecture...", flush=True)
            print("   Encoder: ResNet18 (ImageNet weights)", flush=True)
            print("   Decoder: Random initialization (needs custom weights!)", flush=True)
            self.model = smp.Unet(
                encoder_name="resnet18",        # Pre-trained ResNet18 backbone
                encoder_weights="imagenet",     # Pre-trained on ImageNet
                in_channels=3,                  # RGB input
                classes=1,                      # Binary segmentation (lane/background)
                activation=None,                # We'll apply sigmoid manually
            )
            print("✓ UNet architecture loaded successfully!", flush=True)
        elif model_type == 'simple':
            print("📦 Using custom SimpleLaneNet...", flush=True)
            self.model = SimpleLaneNet()
        else:
            print("📦 Using custom LaneNet...", flush=True)
            self.model = LaneNet()

        print(f"✓ Moving model to device: {self.device}", flush=True)
        self.model.to(self.device)

        # Load custom pretrained weights if provided
        if model_path:
            print("=" * 60, flush=True)
            print(f"🔄 Loading custom weights from: {model_path}", flush=True)
            self.load_weights(model_path)
        else:
            print("=" * 60, flush=True)
            print("⚠️  WARNING: No custom weights provided!", flush=True)
            print("   Decoder has random weights - will NOT work for lane detection!", flush=True)
            print("   Add model_path to config.yaml to use trained model", flush=True)
            print("=" * 60, flush=True)

        print("✓ Setting model to evaluation mode...", flush=True)
        self.model.eval()
        print("✓ DL Lane Detector initialization complete!", flush=True)
        print("=" * 60, flush=True)

    def load_weights(self, model_path: str):
        """
        Load model weights from file or Hugging Face Hub.

        Args:
            model_path: Path to model weights or HF Hub model ID
                       Examples:
                       - Local: "models/lane_model.pth"
                       - HF Hub: "hf://Tilak1812/lane-detection-unet-tusimple"
                       - HF Hub (short): "Tilak1812/lane-detection-unet-tusimple"
        """
        import sys
        try:
            # Check if it's a Hugging Face Hub model
            if model_path.startswith("hf://") or ("/" in model_path and not model_path.startswith("/")):
                # Remove hf:// prefix if present
                hf_model_id = model_path.replace("hf://", "")

                print("🤗 Hugging Face Hub model detected!", flush=True)
                print(f"   Repository: {hf_model_id}", flush=True)
                print(f"   Attempting to download...", flush=True)
                try:
                    from huggingface_hub import hf_hub_download

                    # Download the model file from HF Hub
                    # Try common naming conventions
                    filenames = ["pytorch_model.bin", "model.pth", "best_model.pth", "lane_model.pth"]
                    print(f"   Trying filenames: {', '.join(filenames)}", flush=True)

                    for filename in filenames:
                        try:
                            print(f"   → Trying {filename}...", end=" ", flush=True)
                            local_path = hf_hub_download(
                                repo_id=hf_model_id,
                                filename=filename,
                                cache_dir=".cache/huggingface"
                            )
                            model_path = local_path
                            print(f"✓ Found!", flush=True)
                            print(f"✓ Downloaded to: {local_path}", flush=True)
                            break
                        except Exception as e:
                            print(f"✗ Not found", flush=True)
                            continue
                    else:
                        # Error: write to stderr for log file
                        print(f"\n✗ ERROR: Could not find model file in {hf_model_id}", file=sys.stderr, flush=True)
                        print(f"   Tried: {', '.join(filenames)}", file=sys.stderr, flush=True)
                        print(f"   Check the repo at: https://huggingface.co/{hf_model_id}", file=sys.stderr, flush=True)
                        raise FileNotFoundError(f"No model file found in {hf_model_id}")

                except ImportError:
                    # Error: write to stderr for log file
                    print("✗ ERROR: huggingface_hub not installed!", file=sys.stderr, flush=True)
                    print("   Install with: pip install huggingface_hub", file=sys.stderr, flush=True)
                    print("   Falling back to local path (will likely fail)...", file=sys.stderr, flush=True)

            # Load the model weights
            state_dict = torch.load(model_path, map_location=self.device)

            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            self.model.load_state_dict(state_dict)
            print(f"✓ Loaded model weights from {model_path}", flush=True)

        except Exception as e:
            # Error: write to stderr for log file
            print(f"✗ Failed to load model weights: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)

    def save_weights(self, model_path: str):
        """
        Save model weights to file.

        Args:
            model_path: Path to save weights
        """
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model weights to {model_path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: RGB image as numpy array

        Returns:
            Preprocessed tensor
        """
        # Resize
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to tensor and change dimension order
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def postprocess(self, output: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        # Remove batch dimension and convert to NumPy
        mask = output.squeeze().cpu().detach().numpy()

        # Apply threshold to obtain raw binary segmentation
        binary_mask = (mask > self.threshold).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)

        # Remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Connect broken segments
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Slightly thicken lines
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Convert 0/1 mask → 0/255
        binary_mask = binary_mask * 255

        # Resize back to original resolution
        resized_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))

        return resized_mask


    def detect(self, image: np.ndarray) -> Tuple[Tuple | None, Tuple | None, np.ndarray]:
        """
        Detect lanes in image.

        Args:
            image: RGB input image

        Returns:
            Tuple of (left_lane, right_lane, debug_image)
            Each lane is (x1, y1, x2, y2) or None
        """
        original_size = image.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocess
        lane_mask = self.postprocess(output, original_size)

        # Extract lane lines from mask
        left_lane, right_lane = self._extract_lane_lines(lane_mask, original_size)

        # Create debug image
        debug_image = self._create_debug_image(image, lane_mask, left_lane, right_lane)

        return left_lane, right_lane, debug_image

    def _extract_lane_lines(self, lane_mask: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[Tuple | None, Tuple | None]:
        """
        Extract left and right lane lines from segmentation mask.

        Args:
            lane_mask: Binary lane segmentation mask
            image_shape: (height, width) of original image

        Returns:
            Tuple of (left_lane, right_lane) where each lane is (x1, y1, x2, y2) or None
        """
        height, width = image_shape

        # Define ROI for lane detection (bottom 50% of image - broader area)
        y_min = int(height * 0.5)  # Look at bottom 50% (was 0.6 = 40%)
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
        """
        Fit a lane line from a binary mask.

        Args:
            mask: Binary mask for one lane side
            y_min: Minimum y coordinate
            y_max: Maximum y coordinate
            offset_x: X-offset to add to coordinates

        Returns:
            Lane line as (x1, y1, x2, y2) or None
        """
        # Find lane pixels
        lane_pixels = np.where(mask > 127)  # White pixels

        if len(lane_pixels[0]) < 10:  # Not enough pixels
            return None

        y_coords = lane_pixels[0]
        x_coords = lane_pixels[1]

        # Filter to ROI
        roi_mask = (y_coords >= y_min) & (y_coords <= y_max)
        y_coords = y_coords[roi_mask]
        x_coords = x_coords[roi_mask]

        if len(y_coords) < 10:  # Not enough pixels in ROI
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
        """
        Create debug visualization.

        Args:
            image: Original image
            lane_mask: Lane segmentation mask
            left_lane: Left lane line (x1, y1, x2, y2)
            right_lane: Right lane line (x1, y1, x2, y2)

        Returns:
            Debug visualization
        """
        # Create colored overlay for segmentation mask
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = lane_mask  # Green channel

        # Blend mask
        debug_image = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

        # Draw fitted lane lines on top
        if left_lane:
            cv2.line(debug_image, (left_lane[0], left_lane[1]),
                    (left_lane[2], left_lane[3]), (255, 0, 0), 5)  # Blue for left

        if right_lane:
            cv2.line(debug_image, (right_lane[0], right_lane[1]),
                    (right_lane[2], right_lane[3]), (0, 0, 255), 5)  # Red for right

        # Fill lane area if both lanes detected
        if left_lane and right_lane:
            lane_poly = np.array([[
                [left_lane[0], left_lane[1]],
                [left_lane[2], left_lane[3]],
                [right_lane[2], right_lane[3]],
                [right_lane[0], right_lane[1]]
            ]], dtype=np.int32)

            lane_overlay = debug_image.copy()
            cv2.fillPoly(lane_overlay, lane_poly, (0, 255, 255))  # Yellow fill
            debug_image = cv2.addWeighted(debug_image, 0.8, lane_overlay, 0.2, 0)

        return debug_image


if __name__ == "__main__":
    # Example usage
    print("Testing LaneNet architecture...")

    # Create sample input
    batch_size = 2
    channels = 3
    height, width = 256, 256

    x = torch.randn(batch_size, channels, height, width)

    # Test full model
    model_full = LaneNet()
    output_full = model_full(x)
    print(f"Full LaneNet output shape: {output_full.shape}")

    # Test simple model
    model_simple = SimpleLaneNet()
    output_simple = model_simple(x)
    print(f"Simple LaneNet output shape: {output_simple.shape}")

    # Test detector
    detector = DLLaneDetector(model_type='simple')
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    lane_mask, debug_image = detector.detect(test_image)
    print(f"Lane mask shape: {lane_mask.shape}")
