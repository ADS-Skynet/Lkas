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
        enable_parameter_updates: bool = True,
        parameter_broker_url: str = "tcp://localhost:5560",
        defer_image_connection: bool = False,
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
            enable_parameter_updates: Enable real-time parameter updates via ZMQ
            parameter_broker_url: ZMQ URL for parameter broker
            defer_image_connection: If True, don't connect to image input yet (call connect_to_image_input() later)
        """
        print("\n" + "=" * 60)
        print("Detection Server")
        print("=" * 60)

        # Initialize attributes to None for safe cleanup
        self.image_channel = None
        self.detection_channel = None

        # Store config for deferred image connection
        self.config = config
        self.image_shm_name = image_shm_name
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        try:
            # 1. Initialize detector (core logic)
            print(f"\nInitializing {detection_method.upper()} detector...")
            self.detector = LaneDetection(config, detection_method)
            print(f"✓ Detector ready: {self.detector.get_detector_name()}")
            print(f"  Parameters: {self.detector.get_detector_params()}")

            # 2. Create detection output FIRST (so decision server can connect)
            print(f"\nCreating detection shared memory '{detection_shm_name}'...")
            self.detection_channel = SharedMemoryDetectionChannel(
                name=detection_shm_name,
                create=True,  # Writer mode
                retry_count=retry_count,
                retry_delay=retry_delay,
            )
            print(f"✓ Created detection output")

            # 3. Connect to image input (wait for simulation)
            if not defer_image_connection:
                self._connect_to_image_input()
            else:
                print(f"\n⏸ Deferring image input connection (call connect_to_image_input() later)")

            # Server state
            self.running = False
            self.frame_count = 0
            self.last_print_time = time.time()

            # Setup parameter updates if enabled
            self.param_subscriber = None
            if enable_parameter_updates:
                # Lazy import to avoid circular dependency
                from simulation.integration.zmq_broadcast import ParameterSubscriber

                print(f"\nSetting up real-time parameter updates...")
                self.param_subscriber = ParameterSubscriber(connect_url=parameter_broker_url)
                self.param_subscriber.register_callback('detection', self._on_parameter_update)
                print(f"✓ Parameter updates enabled")

            print("\n" + "=" * 60)
            print("Server initialized successfully!")
            print("=" * 60)

        except Exception as e:
            # Cleanup on initialization failure
            print(f"\n✗ Initialization failed: {e}")
            self._cleanup_on_error()
            raise

    def _connect_to_image_input(self):
        """Internal method to connect to image input."""
        print(f"\nConnecting to image shared memory '{self.image_shm_name}'...")
        self.image_channel = SharedMemoryImageChannel(
            name=self.image_shm_name,
            shape=(self.config.camera.height, self.config.camera.width, 3),
            create=False,  # Reader mode
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
        )
        print(f"✓ Connected to image input")

    def connect_to_image_input(self):
        """
        Connect to image input (for deferred connection).
        Call this after decision server is ready.
        """
        if self.image_channel is not None:
            print("⚠ Image input already connected")
            return
        self._connect_to_image_input()

    def process_image(self, image_msg: ImageMessage) -> DetectionMessage:
        """
        Process one image through the detector.

        Args:
            image_msg: Image message from shared memory

        Returns:
            Detection message with lane results
        """
        return self.detector.process_image(image_msg)

    def _on_parameter_update(self, param_name: str, value: float):
        """
        Handle real-time parameter update.

        Args:
            param_name: Parameter name
            value: New value
        """
        # Try to update the detector's underlying method
        detector_impl = self.detector.detector  # Access the actual CV/DL detector

        if hasattr(detector_impl, 'update_parameter'):
            success = detector_impl.update_parameter(param_name, value)
        #     if success:
        #         print(f"[Detection] Parameter updated: {param_name} = {value}")
        #     else:
        #         print(f"[Detection] Failed to update parameter: {param_name}")
        # else:
        #     print(f"[Detection] Detector does not support parameter updates")

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
                # Poll for parameter updates (non-blocking)
                if self.param_subscriber:
                    self.param_subscriber.poll()

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

        # Close parameter subscriber
        if self.param_subscriber:
            self.param_subscriber.close()

        # Close channels (check if they exist)
        if self.image_channel:
            self.image_channel.close()
        if self.detection_channel:
            self.detection_channel.close()
            # Unlink output channel (we created it)
            self.detection_channel.unlink()

        print("✓ Detection server stopped")

    def _cleanup_on_error(self):
        """Cleanup resources on initialization error (DON'T unlink shared memory!)."""
        # Close connections but DON'T unlink shared memory that others might be using
        if self.image_channel:
            try:
                self.image_channel.close()
            except:
                pass
        if self.detection_channel:
            try:
                self.detection_channel.close()
                # DON'T unlink here! Decision server might already be connected
            except:
                pass

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
