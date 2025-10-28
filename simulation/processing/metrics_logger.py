"""
Metrics Logger - Tracks and logs performance metrics.

SINGLE RESPONSIBILITY: Log performance data and provide statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Optional
from detection.core.models import LaneMetrics


class MetricsLogger:
    """
    Logs performance metrics and FPS tracking.

    SIMPLE CLASS: Just data storage and computation
    No complex logic - easy to understand!
    """

    def __init__(self):
        """Initialize metrics logger."""
        # FPS tracking
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()

        # Performance stats
        self.total_frames = 0
        self.successful_detections = 0

    def update_fps(self):
        """
        Update FPS counter.

        Call this once per frame in main loop.
        """
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time

        # Update FPS every second
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

    def log_frame(self, metrics: LaneMetrics):
        """
        Log metrics from a processed frame.

        Args:
            metrics: Lane metrics from this frame
        """
        self.total_frames += 1
        if metrics.has_both_lanes:
            self.successful_detections += 1

    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps

    def get_detection_rate(self) -> float:
        """
        Get detection success rate.

        Returns:
            Percentage of frames with successful lane detection (0-100)
        """
        if self.total_frames == 0:
            return 0.0
        return (self.successful_detections / self.total_frames) * 100.0

    def print_metrics(self, metrics: LaneMetrics, steering: Optional[float]):
        """
        Print metrics to console.

        Args:
            metrics: Current lane metrics
            steering: Steering correction value
        """
        offset_str = f"{metrics.lateral_offset_meters:.2f}m" if metrics.lateral_offset_meters else "N/A"
        heading_str = f"{metrics.heading_angle_deg:.1f}°" if metrics.heading_angle_deg else "N/A"
        steering_str = f"{steering:.3f}" if steering else "N/A"

        print(
            f"\rStatus: {metrics.departure_status.value:<20} | "
            f"Offset: {offset_str:>8} | "
            f"Heading: {heading_str:>7} | "
            f"Steering: {steering_str:>7} | "
            f"FPS: {self.fps:>5.1f}",
            end=""
        )

    def reset(self):
        """Reset all statistics."""
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.total_frames = 0
        self.successful_detections = 0
