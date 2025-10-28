"""
Flexible visualization system supporting multiple backends.

Supports:
- OpenCV (cv2) - Default, works with X11
- Pygame - Better for X11 forwarding
- Web - Best for remote/Docker (view in browser)
- None - Headless mode
"""

import numpy as np
from typing import Optional, Literal
import cv2

ViewerType = Literal['opencv', 'pygame', 'web', 'none']


class VisualizationManager:
    """
    Manages visualization across different backends.

    Automatically selects best viewer for environment.
    """

    def __init__(self,
                 viewer_type: ViewerType = 'opencv',
                 width: int = 800,
                 height: int = 600,
                 web_port: int = 8080):
        """
        Initialize visualization manager.

        Args:
            viewer_type: Type of viewer ('opencv', 'pygame', 'web', 'none')
            width: Display width
            height: Display height
            web_port: Port for web viewer
        """
        self.viewer_type = viewer_type
        self.width = width
        self.height = height
        self.web_port = web_port
        self.viewer = None

        self._initialize_viewer()

    def _initialize_viewer(self):
        """Initialize the selected viewer."""
        if self.viewer_type == 'opencv':
            print("✓ Using OpenCV viewer (X11)")
            self.viewer = OpenCVViewer(self.width, self.height)

        elif self.viewer_type == 'pygame':
            try:
                from simulation.ui.pygame_viewer import PygameViewer
                self.viewer = PygameViewer(self.width, self.height)
                print("✓ Using Pygame viewer (better X11 support)")
            except ImportError:
                print("⚠ Pygame not available, falling back to OpenCV")
                print("  Install with: pip install pygame")
                self.viewer = OpenCVViewer(self.width, self.height)

        elif self.viewer_type == 'web':
            try:
                from simulation.ui.web_viewer import WebViewer
                self.viewer = WebViewer(self.web_port)
                self.viewer.start()
                print(f"✓ Using Web viewer")
                print(f"  View at: http://localhost:{self.web_port}")
            except Exception as e:
                print(f"⚠ Web viewer failed: {e}")
                print("  Falling back to OpenCV")
                self.viewer = OpenCVViewer(self.width, self.height)

        elif self.viewer_type == 'none':
            print("✓ Headless mode (no visualization)")
            self.viewer = NullViewer()

        else:
            raise ValueError(f"Unknown viewer type: {self.viewer_type}")

    def show(self, image: np.ndarray) -> bool:
        """
        Display image.

        Args:
            image: RGB image array

        Returns:
            True if should continue, False if quit requested
        """
        if self.viewer:
            return self.viewer.show(image)
        return True

    def close(self):
        """Close viewer."""
        if self.viewer:
            self.viewer.close()

    def is_running(self) -> bool:
        """Check if viewer is still active."""
        if self.viewer:
            return self.viewer.is_running()
        return True


class OpenCVViewer:
    """OpenCV-based viewer (default)."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.window_name = "Lane Keeping System (Distributed)"
        self.running = True

    def show(self, image: np.ndarray) -> bool:
        """Show image using OpenCV."""
        if not self.running:
            return False

        # OpenCV expects BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, image_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            self.running = False
            return False

        return True

    def close(self):
        """Close OpenCV window."""
        cv2.destroyAllWindows()
        self.running = False

    def is_running(self) -> bool:
        return self.running


class NullViewer:
    """Null viewer for headless mode."""

    def show(self, image: np.ndarray) -> bool:
        return True

    def close(self):
        pass

    def is_running(self) -> bool:
        return True


def auto_select_viewer() -> ViewerType:
    """
    Automatically select best viewer for current environment.

    Returns:
        Best viewer type for environment
    """
    import os

    # Check if we're in Docker/headless
    if not os.environ.get('DISPLAY'):
        print("No DISPLAY detected, using web viewer")
        return 'web'

    # Check if we're in WSL/remote
    if 'microsoft' in os.uname().release.lower():
        print("WSL detected, using web viewer")
        return 'web'

    # Check if pygame is available (better for X11 forwarding)
    try:
        import pygame
        return 'pygame'
    except ImportError:
        pass

    # Default to OpenCV
    return 'opencv'
