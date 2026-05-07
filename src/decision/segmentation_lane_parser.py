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
        smoothing_factor: float = 0.5,
        lane_width_meters: float = 3.7,
        drift_threshold: float = 0.15,
        departure_threshold: float = 0.35,
        row_step: int = 2,
        camera_offset_x: int = 0,
        min_confidence: float = 0.3,
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
            camera_offset_x: Pixel offset of camera center from vehicle center
                             (negative = shift reference left)
            min_confidence: Minimum confidence to accept a boundary [0, 1].
                           Boundaries below this are treated as noise/track edges.
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
        self.camera_offset_x = camera_offset_x
        self.min_confidence = min_confidence

        self.vehicle_center_x = image_width // 2 + camera_offset_x
        self.roi_top = int(image_height * (1 - roi_ratio))

        # Fitted polynomial coefficients (updated each frame)
        self._left_poly = None    # x = f(y) for left boundary
        self._right_poly = None   # x = f(y) for right boundary
        self._center_poly = None  # x = f(y) for center path

        # Boundary confidence scores (updated each frame)
        self._left_confidence = 0.0
        self._right_confidence = 0.0

        # Previous frame polynomials for temporal smoothing
        self._prev_left_poly = None
        self._prev_right_poly = None
        self._prev_center_poly = None

        # Frame counter for warmup
        self._frame_count = 0
        self._warmup_frames = 10

        # Staleness counter: reset _prev_center_poly after prolonged absence
        # Prevents stale reference from causing misclassification after
        # track discontinuities (crossroads, gaps in detection)
        self._center_stale_frames = 0
        self._max_center_stale = 30

    def parse(self, mask: np.ndarray) -> LaneMetrics:
        """
        Parse binary segmentation mask into lane metrics.

        Args:
            mask: Binary segmentation mask (H, W) uint8, 0=background, 1=lane

        Returns:
            LaneMetrics with computed lane information
        """
        self._frame_count += 1

        # Adapt to actual mask dimensions (camera may differ from config)
        h, w = mask.shape[:2]
        if h != self.image_height or w != self.image_width:
            self.image_height = h
            self.image_width = w
            self.vehicle_center_x = w // 2 + self.camera_offset_x
            self.roi_top = int(h * (1 - self.roi_ratio))

        # 1. Extract boundary points via row scanning
        left_points, right_points = self._extract_boundaries(mask)

        has_left = len(left_points) >= self.min_points
        has_right = len(right_points) >= self.min_points

        # 2. Fit polynomials to boundary points with confidence filtering
        self._left_poly = None
        self._right_poly = None
        self._center_poly = None

        if has_left:
            left_y, left_x = left_points[:, 0], left_points[:, 1]
            raw_poly = np.polyfit(left_y, left_x, self.poly_degree)
            self._left_confidence = self._compute_fit_confidence(left_points, raw_poly)
            if self._left_confidence >= self.min_confidence:
                self._left_poly = self._smooth_poly(raw_poly, self._prev_left_poly)
                self._prev_left_poly = self._left_poly.copy()
            else:
                has_left = False
        else:
            self._left_confidence = 0.0

        if has_right:
            right_y, right_x = right_points[:, 0], right_points[:, 1]
            raw_poly = np.polyfit(right_y, right_x, self.poly_degree)
            self._right_confidence = self._compute_fit_confidence(right_points, raw_poly)
            if self._right_confidence >= self.min_confidence:
                self._right_poly = self._smooth_poly(raw_poly, self._prev_right_poly)
                self._prev_right_poly = self._right_poly.copy()
            else:
                has_right = False
        else:
            self._right_confidence = 0.0

        # 2b. Single-boundary reclassification
        # When only one boundary survives, verify its left/right classification
        # using vehicle_center_x. The per-row classification uses _prev_center_poly
        # which can be stale after track discontinuities, causing persistent
        # misclassification (e.g., a right-side boundary stuck as "left").
        if has_left and not has_right and self._left_poly is not None:
            x_at_bottom = float(np.polyval(self._left_poly, self.image_height - 1))
            if x_at_bottom >= self.vehicle_center_x:
                # Boundary is right of vehicle → reclassify as right
                self._right_poly = self._left_poly
                self._left_poly = None
                self._prev_right_poly = self._right_poly.copy()
                has_left, has_right = False, True
                self._left_confidence, self._right_confidence = (
                    self._right_confidence, self._left_confidence
                )
        elif has_right and not has_left and self._right_poly is not None:
            x_at_bottom = float(np.polyval(self._right_poly, self.image_height - 1))
            if x_at_bottom < self.vehicle_center_x:
                # Boundary is left of vehicle → reclassify as left
                self._left_poly = self._right_poly
                self._right_poly = None
                self._prev_left_poly = self._left_poly.copy()
                has_left, has_right = True, False
                self._left_confidence, self._right_confidence = (
                    self._right_confidence, self._left_confidence
                )

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
            self._center_stale_frames = 0
        else:
            self._center_stale_frames += 1
            if self._center_stale_frames >= self._max_center_stale:
                # Stale for too long — clear so _get_split_reference
                # falls back to vehicle_center_x for fresh classification
                self._prev_center_poly = None

        # 4. Compute metrics from fitted polynomials
        return self._compute_metrics(has_left, has_right)

    def _extract_boundaries(self, mask: np.ndarray):
        """
        Extract left and right lane boundary points from mask via row scanning.

        Uses contiguous pixel clustering and picks the best adjacent pair of
        clusters whose midpoint is closest to the reference center. This correctly
        handles cases where both lane boundaries are on the same side of the
        image center (e.g., vehicle offset within the lane).

        For each row in the ROI:
        - Find all lane pixel indices
        - Group into contiguous clusters (separated by gaps)
        - 2+ clusters: pick the adjacent pair whose midpoint is closest
          to the reference, left cluster inner edge = left boundary,
          right cluster inner edge = right boundary
        - 1 cluster: classify as left or right using reference

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
                # 2+ clusters: build list of (left_edge, right_edge) per cluster
                clusters = []
                start = 0
                for si in split_indices:
                    end = si
                    clusters.append((int(lane_pixels[start]), int(lane_pixels[end])))
                    start = si + 1
                clusters.append((int(lane_pixels[start]), int(lane_pixels[-1])))

                ref_x = self._get_split_reference(y)

                # Pick the best adjacent pair: midpoint closest to reference
                best_pair = None  # (left_inner_edge, right_inner_edge, distance)
                for i in range(len(clusters) - 1):
                    left_inner = clusters[i][1]      # right edge of left cluster
                    right_inner = clusters[i + 1][0]  # left edge of right cluster
                    midpoint = (left_inner + right_inner) / 2.0
                    dist = abs(midpoint - ref_x)
                    if best_pair is None or dist < best_pair[2]:
                        best_pair = (left_inner, right_inner, dist)

                if best_pair is not None:
                    left_points.append([y, best_pair[0]])
                    right_points.append([y, best_pair[1]])
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

    def _compute_fit_confidence(self, points: np.ndarray, poly: np.ndarray) -> float:
        """
        Compute confidence score for a lane boundary polynomial fit.

        Combines two signals:
        - Fit quality: RMSE of polynomial fit (lower = better).
          Real lane markings follow a smooth curve; track edges scatter.
        - Coverage: fraction of ROI rows with boundary points.
          Real lanes are detected consistently; noise is intermittent.

        Returns:
            Confidence in [0, 1]. Higher = more likely a real lane.
        """
        y_vals = points[:, 0]
        x_vals = points[:, 1]

        # RMSE of polynomial fit (not R², which fails for straight lanes)
        x_pred = np.polyval(poly, y_vals)
        rmse = float(np.sqrt(np.mean((x_vals - x_pred) ** 2)))
        max_rmse = 25.0  # pixels — above this, fit quality drops to 0
        fit_quality = max(0.0, 1.0 - rmse / max_rmse)

        # Coverage: what fraction of scannable ROI rows have points
        total_rows = max(1, (self.image_height - 1 - self.roi_top) // self.row_step)
        coverage = min(1.0, len(points) / total_rows)

        return float(fit_quality * coverage)

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

        # Heading angle from polynomial derivative — evaluate slightly above
        # the vehicle (15% up) where the polynomial is more stable.
        # At the very bottom edge, the quadratic tangent can be wildly exaggerated.
        y_heading = float(self.image_height * 0.85)
        heading_angle_deg = self._compute_heading_angle(y_heading)

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

        Convention (matches CV LaneAnalyzer):
            0 deg  = road goes straight ahead
            +N deg = road/lane tilts right in image (dx/dy > 0 going down)
            -N deg = road/lane tilts left in image (dx/dy < 0 going down)
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

    def get_confidences(self) -> tuple[float, float]:
        """Get current boundary confidence scores (left, right)."""
        return (self._left_confidence, self._right_confidence)

    def reset(self):
        """Reset temporal smoothing state."""
        self._left_poly = None
        self._right_poly = None
        self._center_poly = None
        self._prev_left_poly = None
        self._prev_right_poly = None
        self._prev_center_poly = None
        self._left_confidence = 0.0
        self._right_confidence = 0.0
        self._frame_count = 0
        self._center_stale_frames = 0
