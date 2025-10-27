"""
Pygame-based visualization viewer.

Better than OpenCV for remote environments:
- Works better with X11 forwarding
- More lightweight than OpenCV's highgui
- Good for Docker/remote development
"""

import pygame
import numpy as np
from typing import Optional


class PygameViewer:
    """
    Pygame-based image viewer for lane detection visualization.

    Advantages over OpenCV:
    - Better X11 forwarding support
    - Lighter weight
    - Better for remote development
    """

    def __init__(self, width: int = 800, height: int = 600, title: str = "Lane Detection"):
        """
        Initialize pygame viewer.

        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        self.width = width
        self.height = height
        self.title = title

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.running = True

    def show(self, image: np.ndarray) -> bool:
        """
        Display an image.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            True if window should continue, False if quit
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False

        # Convert numpy array to pygame surface
        # Pygame expects (width, height) but numpy is (height, width)
        image_transposed = np.transpose(image, (1, 0, 2))
        surface = pygame.surfarray.make_surface(image_transposed)

        # Scale if needed
        if surface.get_size() != (self.width, self.height):
            surface = pygame.transform.scale(surface, (self.width, self.height))

        # Draw to screen
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Limit frame rate
        self.clock.tick(30)

        return True

    def close(self):
        """Close the viewer."""
        pygame.quit()

    def is_running(self) -> bool:
        """Check if viewer is still running."""
        return self.running


# Simple test
if __name__ == "__main__":
    import time

    # Create test image
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    # Create viewer
    viewer = PygameViewer(800, 600, "Pygame Test")

    print("Showing test image. Press Q or ESC to close, or close window.")

    # Show for a few seconds
    while viewer.show(test_image):
        time.sleep(0.03)  # ~30 FPS

    viewer.close()
    print("Viewer closed")
