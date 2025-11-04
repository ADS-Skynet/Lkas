"""
Simulation Constants

CARLA simulation infrastructure constants.
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


class ActionTypes:
    """Action type identifiers for event system."""

    RESPAWN = "respawn"
    PAUSE = "pause"
    RESUME = "resume"
    QUIT = "quit"


# Convenience exports
__all__ = [
    'SimulationConstants',
    'CommunicationConstants',
    'ActionTypes',
]
