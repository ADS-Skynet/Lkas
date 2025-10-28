"""
Traditional Computer Vision Lane Detection
Using OpenCV for lane detection through color/edge detection and Hough transform.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class CVLaneDetector:
    """Traditional computer vision-based lane detector."""

    def __init__(self,
                 roi_vertices: Optional[List[Tuple[int, int]]] = None,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 hough_rho: int = 2,
                 hough_theta: float = np.pi / 180,
                 hough_threshold: int = 50,
                 hough_min_line_len: int = 40,
                 hough_max_line_gap: int = 100):
        """
        Initialize CV lane detector.

        Args:
            roi_vertices: Region of interest vertices (will be set dynamically if None)
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            hough_rho: Distance resolution in pixels for Hough transform
            hough_theta: Angle resolution in radians for Hough transform
            hough_threshold: Minimum votes for Hough line detection
            hough_min_line_len: Minimum line length
            hough_max_line_gap: Maximum gap between line segments
        """
        self.roi_vertices = roi_vertices
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap

        # Lane tracking
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.smoothing_factor = 0.7

    def _get_default_roi(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get default region of interest based on image shape.

        Args:
            image_shape: (height, width) of the image

        Returns:
            ROI vertices as numpy array
        """
        height, width = image_shape
        vertices = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.45), int(height * 0.6)),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.9), height)
        ]], dtype=np.int32)
        return vertices

    def _region_of_interest(self, image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """
        Apply region of interest mask to image.

        Args:
            image: Input image
            vertices: ROI vertices

        Returns:
            Masked image
        """
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(image, mask)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for lane detection.

        Args:
            image: RGB input image

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        return blur

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detection.

        Args:
            image: Preprocessed grayscale image

        Returns:
            Edge-detected image
        """
        return cv2.Canny(image, self.canny_low, self.canny_high)

    def _detect_lines(self, edge_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lines using Hough transform.

        Args:
            edge_image: Edge-detected image

        Returns:
            Array of detected lines or None
        """
        lines = cv2.HoughLinesP(
            edge_image,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap
        )
        return lines

    def _separate_lanes(self, lines: np.ndarray, image_width: int) -> Tuple[List, List]:
        """
        Separate detected lines into left and right lanes.

        Args:
            lines: Detected lines from Hough transform
            image_width: Width of the image

        Returns:
            Tuple of (left_lane_lines, right_lane_lines)
        """
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:  # Vertical line
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope and position
            if abs(slope) < 0.5:  # Too horizontal
                continue

            # Separate left and right based on slope and position
            if slope < 0 and x1 < image_width / 2:  # Left lane
                left_lines.append(line[0])
            elif slope > 0 and x1 > image_width / 2:  # Right lane
                right_lines.append(line[0])

        return left_lines, right_lines

    def _average_lane_lines(self, lines: List, y_min: int, y_max: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Average multiple line segments into a single lane line.

        Args:
            lines: List of line segments
            y_min: Minimum y coordinate
            y_max: Maximum y coordinate

        Returns:
            Averaged line as (x1, y1, x2, y2) or None
        """
        if not lines:
            return None

        # Fit a linear polynomial to all points
        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return None

        # Fit line
        poly = np.polyfit(y_coords, x_coords, 1)

        # Calculate x coordinates for y_min and y_max
        x1 = int(poly[0] * y_max + poly[1])
        x2 = int(poly[0] * y_min + poly[1])

        return (x1, y_max, x2, y_min)

    def _smooth_lane(self, current_lane: Optional[Tuple],
                     previous_lane: Optional[Tuple]) -> Optional[Tuple]:
        """
        Smooth lane detection using exponential moving average.

        Args:
            current_lane: Current detected lane
            previous_lane: Previously detected lane

        Returns:
            Smoothed lane
        """
        if current_lane is None:
            return previous_lane

        if previous_lane is None:
            return current_lane

        # Apply smoothing
        smoothed = tuple(
            int(self.smoothing_factor * curr + (1 - self.smoothing_factor) * prev)
            for curr, prev in zip(current_lane, previous_lane)
        )

        return smoothed

    def detect(self, image: np.ndarray) -> Tuple[Optional[Tuple], Optional[Tuple], np.ndarray]:
        """
        Detect lane lines in the image.

        Args:
            image: RGB input image

        Returns:
            Tuple of (left_lane, right_lane, debug_image)
            Each lane is (x1, y1, x2, y2) or None
        """
        height, width = image.shape[:2]

        # Preprocess
        preprocessed = self._preprocess_image(image)

        # Edge detection
        edges = self._detect_edges(preprocessed)

        # Apply ROI
        roi_vertices = self.roi_vertices if self.roi_vertices else self._get_default_roi((height, width))
        roi_edges = self._region_of_interest(edges, roi_vertices)

        # Detect lines
        lines = self._detect_lines(roi_edges)

        # Separate left and right lanes
        left_lines, right_lines = self._separate_lanes(lines, width)

        # Average and smooth lanes
        y_min = int(height * 0.6)
        y_max = height

        left_lane = self._average_lane_lines(left_lines, y_min, y_max)
        right_lane = self._average_lane_lines(right_lines, y_min, y_max)

        # Apply temporal smoothing
        left_lane = self._smooth_lane(left_lane, self.prev_left_lane)
        right_lane = self._smooth_lane(right_lane, self.prev_right_lane)

        # Update previous lanes
        self.prev_left_lane = left_lane
        self.prev_right_lane = right_lane

        # Create debug image
        debug_image = self._create_debug_image(image, left_lane, right_lane, roi_vertices)

        return left_lane, right_lane, debug_image

    def _create_debug_image(self, image: np.ndarray,
                           left_lane: Optional[Tuple],
                           right_lane: Optional[Tuple],
                           roi_vertices: np.ndarray) -> np.ndarray:
        """
        Create debug visualization image.

        Args:
            image: Original image
            left_lane: Left lane line
            right_lane: Right lane line
            roi_vertices: ROI vertices

        Returns:
            Debug visualization image
        """
        debug_img = image.copy()

        # Draw ROI
        cv2.polylines(debug_img, roi_vertices, True, (0, 255, 0), 2)

        # Draw lane lines
        if left_lane:
            cv2.line(debug_img, (left_lane[0], left_lane[1]),
                    (left_lane[2], left_lane[3]), (255, 0, 0), 5)

        if right_lane:
            cv2.line(debug_img, (right_lane[0], right_lane[1]),
                    (right_lane[2], right_lane[3]), (0, 0, 255), 5)

        # Fill lane area
        if left_lane and right_lane:
            lane_poly = np.array([[
                [left_lane[0], left_lane[1]],
                [left_lane[2], left_lane[3]],
                [right_lane[2], right_lane[3]],
                [right_lane[0], right_lane[1]]
            ]], dtype=np.int32)

            overlay = debug_img.copy()
            cv2.fillPoly(overlay, lane_poly, (0, 255, 0))
            debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)

        return debug_img


if __name__ == "__main__":
    # Example usage
    detector = CVLaneDetector()

    # Test with a sample image (you would load a real image here)
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)

    left_lane, right_lane, debug_image = detector.detect(test_image)

    print(f"Left lane: {left_lane}")
    print(f"Right lane: {right_lane}")
