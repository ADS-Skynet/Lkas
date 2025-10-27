"""
Video Recorder - Manages video recording to file.

SINGLE RESPONSIBILITY: Handle video file creation and frame writing.
"""

import cv2
import numpy as np
from typing import Optional
from pathlib import Path


class VideoRecorder:
    """
    Handles video recording to file.

    WRAPPER CLASS: Wraps OpenCV VideoWriter with cleaner interface
    """

    def __init__(self, output_path: Optional[str] = None,
                 fps: float = 30.0,
                 fourcc: str = 'mp4v'):
        """
        Initialize video recorder.

        Args:
            output_path: Path to output video file (None = don't record)
            fps: Frames per second
            fourcc: Video codec ('mp4v', 'XVID', 'MJPG', etc.)

        PYTHON KEYWORDS:
            Optional[str]: Can be string or None
            fourcc: Four Character Code for video codec
        """
        self.output_path = output_path
        self.fps = fps
        self.fourcc_str = fourcc
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False

    def start(self, width: int, height: int):
        """
        Start recording video.

        Args:
            width: Video frame width
            height: Video frame height

        Returns:
            True if recording started successfully
        """
        if self.output_path is None:
            return False

        # Create parent directory if needed
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert fourcc string to code
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)

        # Create video writer
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (width, height)
        )

        if self.writer.isOpened():
            self.is_recording = True
            print(f"✓ Recording video to: {self.output_path}")
            return True
        else:
            print(f"✗ Failed to start recording: {self.output_path}")
            return False

    def write(self, frame: np.ndarray):
        """
        Write a frame to video file.

        Args:
            frame: RGB or BGR image frame
        """
        if self.is_recording and self.writer is not None:
            # OpenCV expects BGR, so convert if needed
            # (assuming input is RGB from CARLA)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr_frame)

    def stop(self):
        """Stop recording and close video file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.is_recording = False
            print(f"✓ Video saved to: {self.output_path}")

    def __enter__(self):
        """Context manager entry (for 'with' statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (automatically stops recording)."""
        self.stop()


# =============================================================================
# CONTEXT MANAGER PATTERN
# =============================================================================
"""
The Context Manager pattern ensures cleanup happens automatically.

C++ ANALOGY:
    // C++ RAII (Resource Acquisition Is Initialization)
    class VideoRecorder {
        VideoWriter* writer;
    public:
        VideoRecorder(string path) {
            writer = new VideoWriter(path);
        }
        ~VideoRecorder() {  // Destructor
            delete writer;  // Cleanup happens automatically
        }
    };

PYTHON VERSION (this class):
    class VideoRecorder:
        def __enter__(self):
            # Called when entering 'with' block
            return self

        def __exit__(self, ...):
            # Called when leaving 'with' block
            self.stop()  # Cleanup happens automatically!

USAGE:
    # With context manager (recommended)
    with VideoRecorder('output.mp4') as recorder:
        recorder.start(800, 600)
        recorder.write(frame1)
        recorder.write(frame2)
    # recorder.stop() called automatically!

    # Without context manager (manual cleanup)
    recorder = VideoRecorder('output.mp4')
    recorder.start(800, 600)
    recorder.write(frame1)
    recorder.stop()  # Must remember to call this!

BENEFITS:
    - Guaranteed cleanup (even if exception occurs)
    - Cleaner code
    - Prevents resource leaks
"""


if __name__ == "__main__":
    # Example usage
    print("Testing VideoRecorder...")

    # Create test frames
    test_frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Test without recording
    recorder1 = VideoRecorder(output_path=None)
    print(f"Recording enabled: {recorder1.is_recording}")

    # Test with recording (context manager)
    with VideoRecorder('test_output.mp4') as recorder2:
        recorder2.start(800, 600)
        for i in range(10):
            recorder2.write(test_frame)
    # Automatically stopped!

    print("✓ VideoRecorder tested successfully")
