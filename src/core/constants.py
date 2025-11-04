"""
Core Constants

Central location for all magic numbers and configuration constants.
Following clean code principles: NO MAGIC NUMBERS!
"""


class SimulationConstants:
    """Constants for CARLA simulation."""

    # Tick rate and timing
    TICK_RATE_HZ = 20
    FIXED_DELTA_SECONDS = 0.05  # 1/20 = 0.05 seconds per tick

    # Retry configuration
    DEFAULT_RETRY_COUNT = 20
    RETRY_DELAY_SECONDS = 0.5

    # Warmup and initialization
    WARMUP_FRAMES = 50
    WARMUP_DURATION_SECONDS = 2.5  # 50 frames / 20 Hz

    # Status reporting
    STATUS_PRINT_INTERVAL_FRAMES = 30
    LATENCY_REPORT_INTERVAL_FRAMES = 90

    # Control defaults
    DEFAULT_BASE_THROTTLE = 0.3
    DEFAULT_DETECTOR_TIMEOUT_MS = 1000

    # Pause delay when paused
    PAUSE_SLEEP_SECONDS = 0.1


class CVDetectionConstants:
    """Constants for Computer Vision lane detection."""

    # Region of Interest (ROI) ratios
    ROI_BOTTOM_HEIGHT_RATIO = 0.5  # Start ROI from 50% of image height
    ROI_LEFT_WIDTH_RATIO = 0.05    # Left boundary at 5% of width
    ROI_TOP_LEFT_RATIO = 0.35       # Top-left corner at 35% width
    ROI_TOP_RIGHT_RATIO = 0.65      # Top-right corner at 65% width
    ROI_RIGHT_WIDTH_RATIO = 0.95    # Right boundary at 95% width

    # Lane detection thresholds
    MIN_LANE_SLOPE = 0.5            # Minimum absolute slope to be considered a lane
    MAX_HORIZONTAL_SLOPE = 0.5      # Maximum slope to reject horizontal lines

    # JPEG compression quality
    JPEG_QUALITY = 85               # Quality for image compression (0-100)

    # Hough transform parameters
    HOUGH_RHO = 2                   # Distance resolution in pixels
    HOUGH_THETA_DEGREES = 1         # Angle resolution in degrees
    HOUGH_THRESHOLD = 15            # Minimum votes to detect a line
    HOUGH_MIN_LINE_LENGTH = 40      # Minimum line length in pixels
    HOUGH_MAX_LINE_GAP = 20         # Maximum gap between line segments


class CommunicationConstants:
    """Constants for inter-process communication."""

    # ZMQ message topics
    TOPIC_FRAME = b'frame'
    TOPIC_DETECTION = b'detection'
    TOPIC_STATE = b'state'
    TOPIC_ACTION = b'action'

    # Shared memory defaults
    DEFAULT_IMAGE_SHM_NAME = "camera_feed"
    DEFAULT_DETECTION_SHM_NAME = "detection_results"

    # ZMQ default ports
    DEFAULT_DETECTION_PORT = 5556
    DEFAULT_BROADCAST_PORT = 5557
    DEFAULT_ACTION_PORT = 5558

    # Stream rates
    WEB_VIEWER_FPS = 30
    STREAM_FRAME_DELAY_MS = 33  # ~30 FPS (1000ms / 30)


class ControllerConstants:
    """Constants for PD controller and decision making."""

    # Throttle policy defaults
    DEFAULT_THROTTLE_BASE = 0.4
    DEFAULT_THROTTLE_MIN = 0.2
    DEFAULT_STEER_THRESHOLD = 0.3
    DEFAULT_STEER_MAX = 0.8

    # Control limits
    MAX_STEERING = 1.0
    MIN_STEERING = -1.0
    MAX_THROTTLE = 1.0
    MIN_THROTTLE = 0.0
    MAX_BRAKE = 1.0
    MIN_BRAKE = 0.0

    # PD controller defaults
    DEFAULT_KP = 0.5
    DEFAULT_KD = 0.1


class VisualizationConstants:
    """Constants for visualization and rendering."""

    # Colors (BGR format for OpenCV)
    COLOR_LEFT_LANE = (255, 0, 0)      # Blue
    COLOR_RIGHT_LANE = (0, 0, 255)     # Red
    COLOR_CENTER_LINE = (0, 255, 0)    # Green
    COLOR_WARNING = (0, 165, 255)      # Orange
    COLOR_DANGER = (0, 0, 255)         # Red
    COLOR_SAFE = (0, 255, 0)           # Green
    COLOR_TEXT = (255, 255, 255)       # White

    # Line thickness
    LANE_LINE_THICKNESS = 5
    CENTER_LINE_THICKNESS = 2
    THIN_LINE_THICKNESS = 1

    # Font settings
    FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    # HUD positioning
    HUD_MARGIN_X = 10
    HUD_MARGIN_Y = 30
    HUD_LINE_SPACING = 30


class DetectorTypes:
    """Detector type identifiers."""

    COMPUTER_VISION = "cv"
    DEEP_LEARNING = "dl"


class ActionTypes:
    """Action type identifiers for event system."""

    RESPAWN = "respawn"
    PAUSE = "pause"
    RESUME = "resume"
    QUIT = "quit"


class ControlModes:
    """Control mode identifiers."""

    MANUAL = "manual"
    AUTOPILOT = "autopilot"
    LANE_KEEPING = "lane_keeping"
    EMERGENCY_STOP = "emergency_stop"


# Convenience exports
__all__ = [
    'SimulationConstants',
    'CVDetectionConstants',
    'CommunicationConstants',
    'ControllerConstants',
    'VisualizationConstants',
    'DetectorTypes',
    'ActionTypes',
    'ControlModes',
]
