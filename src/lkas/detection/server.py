"""
Detection Server

Unified server that combines:
- Lane detection (CV or DL)
- Shared memory communication (input/output)
- Processing loop

This is a cleaner architecture where the server encapsulates all detection concerns.
"""

import time
import signal
import sys
from typing import Optional

from lkas.detection import LaneDetection
from lkas.detection.core.config import Config
from lkas.detection.integration.messages import ImageMessage, DetectionMessage
from lkas.detection.integration.shared_memory_detection import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel
)


class DetectionServer:
    """
    Unified detection server.

    Responsibilities:
    - Read images from shared memory
    - Detect lanes using CV or DL
    - Write detection results to shared memory
    - Manage server lifecycle

    This encapsulates the complete detection service in one cohesive class.
    """

    def __init__(
        self,
        config: Config,
        detection_method: str = "cv",
        image_shm_name: str = "camera_feed",
        detection_shm_name: str = "detection_results",
        retry_count: int = 20,
        retry_delay: float = 0.5,
    ):
        """
        Initialize detection server.

        Args:
            config: System configuration
            detection_method: Detection method ('cv' or 'dl')
            image_shm_name: Shared memory name for image input
            detection_shm_name: Shared memory name for detection output
            retry_count: Connection retry attempts
            retry_delay: Delay between retries (seconds)
        """
        print("\n" + "=" * 60)
        print("Detection Server")
        print("=" * 60)

        # 1. Initialize detector (core logic)
        print(f"\nInitializing {detection_method.upper()} detector...")
        self.detector = LaneDetection(config, detection_method)
        print(f"✓ Detector ready: {self.detector.get_detector_name()}")
        print(f"  Parameters: {self.detector.get_detector_params()}")

        # 2. Connect to image input (shared memory)
        print(f"\nConnecting to image shared memory '{image_shm_name}'...")
        self.image_channel = SharedMemoryImageChannel(
            name=image_shm_name,
            shape=(config.camera.height, config.camera.width, 3),
            create=False,  # Reader mode
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
        print(f"✓ Connected to image input")

        # 3. Create detection output (shared memory)
        print(f"Creating detection shared memory '{detection_shm_name}'...")
        self.detection_channel = SharedMemoryDetectionChannel(
            name=detection_shm_name,
            create=True,  # Writer mode
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
        print(f"✓ Created detection output")

        # Server state
        self.running = False
        self.frame_count = 0
        self.last_print_time = time.time()

        print("\n" + "=" * 60)
        print("Server initialized successfully!")
        print("=" * 60)

    def process_image(self, image_msg: ImageMessage) -> DetectionMessage:
        """
        Process one image through the detector.

        Args:
            image_msg: Image message from shared memory

        Returns:
            Detection message with lane results
        """
        return self.detector.process_image(image_msg)

    def run(self, print_stats: bool = True):
        """
        Start the detection server main loop.

        Args:
            print_stats: Whether to print performance statistics
        """
        # Register signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nReceived interrupt signal")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("\n" + "=" * 60)
        print("Detection Server Running")
        print("=" * 60)
        print(f"Reading images from: {self.image_channel.name}")
        print(f"Writing detections to: {self.detection_channel.name}")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        self.running = True

        try:
            while self.running:
                # Read image from shared memory (non-blocking with timeout)
                image_msg = self.image_channel.read_blocking(timeout=0.1, copy=True)

                if image_msg is None:
                    continue

                # Process detection
                detection_msg = self.process_image(image_msg)

                # Write results to shared memory
                self.detection_channel.write(detection_msg)

                self.frame_count += 1

                # Stats tracking and optional printing
                if time.time() - self.last_print_time > 3.0:
                    if print_stats:
                        fps = self.frame_count / (time.time() - self.last_print_time)
                        print(
                            f"\r{fps:.1f} FPS | Frame {image_msg.frame_id} | "
                            f"Processing: {detection_msg.processing_time_ms:.2f}ms | "
                            f"Lanes: L={detection_msg.left_lane is not None} "
                            f"R={detection_msg.right_lane is not None}",
                            end="",
                            flush=True,
                        )
                    self.frame_count = 0
                    self.last_print_time = time.time()

        except KeyboardInterrupt:
            print("\n\nStopping detection server...")
        finally:
            self.stop()

    def stop(self):
        """Stop the server and cleanup resources."""
        self.running = False

        # Close channels
        self.image_channel.close()
        self.detection_channel.close()

        # Unlink output channel (we created it)
        self.detection_channel.unlink()

        print("✓ Detection server stopped")

    def get_detector_name(self) -> str:
        """Get the name of the active detector."""
        return self.detector.get_detector_name()

    def get_detector_params(self) -> dict:
        """Get the parameters of the active detector."""
        return self.detector.get_detector_params()


# =============================================================================
# Architecture Benefits
# =============================================================================

"""
UNIFIED SERVER ARCHITECTURE
============================

Old (Confusing) Structure:
    detection/run.py
    └── DetectionService
        ├── self.detection_module (LaneDetection)
        └── self.server (old shared memory server class)
            └── Manually wire them together

New (Clean) Structure:
    detection/server.py
    └── DetectionServer
        ├── self.detector (LaneDetection)
        ├── self.image_channel (Input)
        ├── self.detection_channel (Output)
        └── self.run() (Processing loop)

Benefits:
1. Single Responsibility: One class handles the complete detection service
2. Encapsulation: All detection concerns in one place
3. Clear Interface: Simple to instantiate and run
4. Easy Testing: Can mock image_channel/detection_channel
5. Maintainability: Changes to communication don't affect other code

Usage:
    server = DetectionServer(
        config=config,
        detection_method="cv",
        image_shm_name="camera_feed",
        detection_shm_name="detection_results"
    )
    server.run()

Comparison to Decision Server:
    Both follow the same pattern now:
    - DetectionServer: Images → Detections
    - DecisionServer: Detections → Controls
"""
