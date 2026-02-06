"""
Segmentation Lane Parser

Parses binary segmentation masks from deep learning lane detection
into structured lane information using polynomial fitting.

Approach (standard in OpenPilot, Apollo, Autoware):
    1. ROI selection (bottom portion of image where road is visible)
    2. Row-by-row scanning to find lane boundary pixels
    3. Outlier filtering (IQR method)
    4. 2nd-degree polynomial fitting: x = f(y)
    5. Center path computation from left/right boundaries
    6. Metrics calculation (lateral offset, heading, lane width)

The polynomial representation captures road curvature, enabling
lookahead-based controllers (Pure Pursuit) to anticipate curves
rather than only reacting to the current lateral offset.
"""

import numpy as np
from common.types.models import LaneMetrics, LaneDepartureStatus


class SegmentationLaneParser:
    """
    Parses binary segmentation masks into lane polynomials and metrics.

    Pipeline:
        mask (H, W) uint8  -->  boundary points  -->  polynomial fit  -->  LaneMetrics
           0=background          per-row scan          np.polyfit          offset, heading,
           1=lane pixel           + IQR filter          x = f(y)           lane width, etc.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        roi_ratio: float = 0.6,
        poly_degree: int = 2,
        min_points: int = 20,
        smoothing_factor: float = 0.3,
        lane_width_meters: float = 3.7,
        drift_threshold: float = 0.15,
        departure_threshold: float = 0.35,
        row_step: int = 2,
    ):
        """
        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
            roi_ratio: Fraction of image height to use as ROI (from bottom)
            poly_degree: Degree of polynomial fit (2 = quadratic)
            min_points: Minimum boundary points required for polynomial fitting
            smoothing_factor: Temporal smoothing [0, 1] (0 = no smoothing)
            lane_width_meters: Standard lane width in meters (3.7m for US highways)
            drift_threshold: Fraction of lane width for drift warning
            departure_threshold: Fraction of lane width for departure warning
            row_step: Step size for row scanning (skip rows for performance)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.roi_ratio = roi_ratio
        self.poly_degree = poly_degree
        self.min_points = min_points
        self.smoothing_factor = smoothing_factor
        self.lane_width_meters = lane_width_meters
        self.drift_threshold = drift_threshold
        self.departure_threshold = departure_threshold
        self.row_step = row_step

        self.vehicle_center_x = image_width // 2
        self.roi_top = int(image_height * (1 - roi_ratio))

        # Fitted polynomial coefficients (updated each frame)
        self._left_poly = None    # x = f(y) for left boundary
        self._right_poly = None   # x = f(y) for right boundary
        self._center_poly = None  # x = f(y) for center path

        # Previous frame polynomials for temporal smoothing
        self._prev_left_poly = None
        self._prev_right_poly = None
        self._prev_center_poly = None

        # Frame counter for warmup
        self._frame_count = 0
        self._warmup_frames = 10

    def parse(self, mask: np.ndarray) -> LaneMetrics:
        """
        Parse binary segmentation mask into lane metrics.

        Args:
            mask: Binary segmentation mask (H, W) uint8, 0=background, 1=lane

        Returns:
            LaneMetrics with computed lane information
        """
        self._frame_count += 1

        # 1. Extract boundary points via row scanning
        left_points, right_points = self._extract_boundaries(mask)

        has_left = len(left_points) >= self.min_points
        has_right = len(right_points) >= self.min_points

        # 2. Fit polynomials to boundary points
        self._left_poly = None
        self._right_poly = None
        self._center_poly = None

        if has_left:
            left_y, left_x = left_points[:, 0], left_points[:, 1]
            self._left_poly = np.polyfit(left_y, left_x, self.poly_degree)
            self._left_poly = self._smooth_poly(self._left_poly, self._prev_left_poly)
            self._prev_left_poly = self._left_poly.copy()

        if has_right:
            right_y, right_x = right_points[:, 0], right_points[:, 1]
            self._right_poly = np.polyfit(right_y, right_x, self.poly_degree)
            self._right_poly = self._smooth_poly(self._right_poly, self._prev_right_poly)
            self._prev_right_poly = self._right_poly.copy()

        # 3. Compute center path (from already-smoothed boundaries, no extra smoothing)
        if has_left and has_right:
            self._center_poly = (self._left_poly + self._right_poly) / 2.0
        elif has_left:
            # Estimate center from left boundary + assumed half lane width
            self._center_poly = self._left_poly.copy()
            self._center_poly[-1] += self._estimate_half_lane_width()
        elif has_right:
            self._center_poly = self._right_poly.copy()
            self._center_poly[-1] -= self._estimate_half_lane_width()

        if self._center_poly is not None:
            self._prev_center_poly = self._center_poly.copy()

        # 4. Compute metrics from fitted polynomials
        return self._compute_metrics(has_left, has_right)

    def _extract_boundaries(self, mask: np.ndarray):
        """
        Extract left and right lane boundary points from mask via row scanning.

        Uses contiguous pixel clustering instead of a fixed center split.
        This handles curves correctly where both lane markings may shift
        to the same side of the image center.

        For each row in the ROI:
        - Find all lane pixel indices
        - Group into contiguous clusters (separated by gaps)
        - 2+ clusters: leftmost cluster's inner edge = left boundary,
                       rightmost cluster's inner edge = right boundary
        - 1 cluster: classify as left or right using previous frame's center

        Returns:
            Tuple of (left_points, right_points) as Nx2 arrays of [y, x]
        """
        left_points = []
        right_points = []

        # Minimum gap between clusters to be considered separate lane markings
        min_gap = max(5, int(self.image_width * 0.03))

        for y in range(self.image_height - 1, self.roi_top, -self.row_step):
            lane_pixels = np.nonzero(mask[y])[0]

            if len(lane_pixels) < 2:
                continue

            # Find contiguous clusters by detecting gaps
            gaps = np.diff(lane_pixels)
            split_indices = np.where(gaps > min_gap)[0]

            if len(split_indices) >= 1:
                # 2+ clusters: use leftmost and rightmost
                # Left boundary: inner (rightmost) edge of leftmost cluster
                first_end = split_indices[0]
                left_points.append([y, int(lane_pixels[first_end])])

                # Right boundary: inner (leftmost) edge of rightmost cluster
                last_start = split_indices[-1] + 1
                right_points.append([y, int(lane_pixels[last_start])])
            else:
                # Single cluster: classify using previous frame's center path
                cluster_center = (lane_pixels[0] + lane_pixels[-1]) / 2.0
                ref_x = self._get_split_reference(y)

                if cluster_center < ref_x:
                    left_points.append([y, int(lane_pixels[-1])])
                else:
                    right_points.append([y, int(lane_pixels[0])])

        left_arr = np.array(left_points, dtype=np.float64) if left_points else np.empty((0, 2))
        right_arr = np.array(right_points, dtype=np.float64) if right_points else np.empty((0, 2))

        # Filter outliers using IQR method
        if len(left_arr) >= self.min_points:
            left_arr = self._filter_outliers(left_arr)
        if len(right_arr) >= self.min_points:
            right_arr = self._filter_outliers(right_arr)

        return left_arr, right_arr

    def _get_split_reference(self, y: float) -> float:
        """
        Get reference x-position for classifying a single-cluster row.

        Uses previous frame's center path if available, otherwise image center.
        """
        if self._prev_center_poly is not None:
            return float(np.polyval(self._prev_center_poly, y))
        return float(self.vehicle_center_x)

    def _filter_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        Filter outlier boundary points using IQR method on x-values.

        Removes points whose x-coordinate falls outside 1.5 * IQR from
        the quartile boundaries. Standard statistical outlier detection.
        """
        x_vals = points[:, 1]
        q1 = np.percentile(x_vals, 25)
        q3 = np.percentile(x_vals, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        valid = (x_vals >= lower) & (x_vals <= upper)
        return points[valid]

    def _smooth_poly(
        self, current: np.ndarray, previous: np.ndarray | None
    ) -> np.ndarray:
        """Temporal smoothing of polynomial coefficients (exponential moving average)."""
        if previous is None or self._frame_count <= self._warmup_frames:
            return current
        alpha = self.smoothing_factor
        return alpha * previous + (1 - alpha) * current

    def _estimate_half_lane_width(self) -> float:
        """
        Estimate half lane width in pixels when only one boundary is visible.

        Uses ~15% of image width as a default estimate, which corresponds
        to roughly half a lane width in typical camera configurations.
        """
        return self.image_width * 0.15

    def _compute_metrics(self, has_left: bool, has_right: bool) -> LaneMetrics:
        """Compute LaneMetrics from fitted polynomials."""
        if self._center_poly is None:
            return LaneMetrics(
                vehicle_center_x=float(self.vehicle_center_x),
                departure_status=LaneDepartureStatus.NO_LANES,
                has_left_lane=False,
                has_right_lane=False,
                has_both_lanes=False,
            )

        # Evaluate at vehicle position (bottom of image)
        y_vehicle = float(self.image_height - 1)

        # Lane center at vehicle position
        lane_center_x = float(np.polyval(self._center_poly, y_vehicle))

        # Lateral offset (positive = vehicle is right of center)
        lateral_offset_pixels = float(self.vehicle_center_x - lane_center_x)

        # Lane width from left/right polynomials
        lane_width_pixels = None
        if self._left_poly is not None and self._right_poly is not None:
            left_x = np.polyval(self._left_poly, y_vehicle)
            right_x = np.polyval(self._right_poly, y_vehicle)
            lane_width_pixels = float(abs(right_x - left_x))

        # Lateral offset in meters
        lateral_offset_meters = None
        if lane_width_pixels is not None and lane_width_pixels > 0:
            pixels_per_meter = lane_width_pixels / self.lane_width_meters
            lateral_offset_meters = lateral_offset_pixels / pixels_per_meter

        # Normalized offset [-1, 1]
        lateral_offset_normalized = None
        if lane_width_pixels is not None and lane_width_pixels > 0:
            lateral_offset_normalized = lateral_offset_pixels / lane_width_pixels
        elif has_left or has_right:
            # Fallback: normalize by half image width
            lateral_offset_normalized = lateral_offset_pixels / (self.image_width / 2.0)

        # Heading angle from polynomial first derivative
        heading_angle_deg = self._compute_heading_angle(y_vehicle)

        # Departure status
        departure_status = self._get_departure_status(
            lateral_offset_pixels, lane_width_pixels, has_left, has_right
        )

        return LaneMetrics(
            vehicle_center_x=float(self.vehicle_center_x),
            lane_center_x=lane_center_x,
            lane_width_pixels=lane_width_pixels,
            lateral_offset_pixels=lateral_offset_pixels,
            lateral_offset_meters=lateral_offset_meters,
            lateral_offset_normalized=lateral_offset_normalized,
            heading_angle_deg=heading_angle_deg,
            departure_status=departure_status,
            has_left_lane=has_left,
            has_right_lane=has_right,
            has_both_lanes=(has_left and has_right),
        )

    def _compute_heading_angle(self, y: float) -> float | None:
        """
        Compute heading angle from center polynomial derivative.

        The polynomial gives x = f(y). The derivative dx/dy is the lateral
        slope of the road at position y. The heading angle is arctan(slope).

        Convention (matches existing PD/PID controllers):
            0 deg  = road goes straight ahead
            +N deg = road curves right (vehicle heading left of road)
            -N deg = road curves left (vehicle heading right of road)
        """
        if self._center_poly is None:
            return None

        deriv_poly = np.polyder(self._center_poly)
        slope = np.polyval(deriv_poly, y)
        angle_deg = float(np.degrees(np.arctan(slope)))

        return angle_deg

    def _get_departure_status(
        self,
        offset_pixels: float | None,
        lane_width: float | None,
        has_left: bool,
        has_right: bool,
    ) -> LaneDepartureStatus:
        """Determine lane departure status based on lateral offset."""
        if not has_left and not has_right:
            return LaneDepartureStatus.NO_LANES

        if offset_pixels is None:
            return LaneDepartureStatus.NO_LANES

        if lane_width is None or lane_width == 0:
            return LaneDepartureStatus.UNKNOWN

        offset_fraction = abs(offset_pixels) / lane_width

        if offset_fraction >= self.departure_threshold:
            if offset_pixels > 0:
                return LaneDepartureStatus.RIGHT_DEPARTURE
            else:
                return LaneDepartureStatus.LEFT_DEPARTURE
        elif offset_fraction >= self.drift_threshold:
            if offset_pixels > 0:
                return LaneDepartureStatus.RIGHT_DRIFT
            else:
                return LaneDepartureStatus.LEFT_DRIFT
        else:
            return LaneDepartureStatus.CENTERED

    # ------------------------------------------------------------------
    # Public accessors for controller integration
    # ------------------------------------------------------------------

    def get_center_poly(self) -> np.ndarray | None:
        """Get current center path polynomial coefficients."""
        return self._center_poly

    def get_curvature(self, y: float | None = None) -> float | None:
        """
        Get road curvature at given y position.

        Curvature = |x''| / (1 + x'^2)^(3/2)
        """
        if self._center_poly is None or self.poly_degree < 2:
            return None

        if y is None:
            y = float(self.image_height - 1)

        d1 = np.polyder(self._center_poly)
        dx = np.polyval(d1, y)

        d2 = np.polyder(d1)
        ddx = np.polyval(d2, y)

        curvature = abs(ddx) / (1 + dx**2) ** 1.5
        return float(curvature)

    def evaluate_center_at(self, y: float) -> float | None:
        """Evaluate center path polynomial at given y position."""
        if self._center_poly is None:
            return None
        return float(np.polyval(self._center_poly, y))

    def reset(self):
        """Reset temporal smoothing state."""
        self._left_poly = None
        self._right_poly = None
        self._center_poly = None
        self._prev_left_poly = None
        self._prev_right_poly = None
        self._prev_center_poly = None
        self._frame_count = 0
