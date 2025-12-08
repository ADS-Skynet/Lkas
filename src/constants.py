"""
LKAS Constants

Lane Keeping Assist System specific constants.
Module-specific configuration values and type identifiers.
Common configuration values are in common.config.
"""


class DetectorTypes:
    """Detector type identifiers."""

    COMPUTER_VISION = "cv"
    DEEP_LEARNING = "dl"


class ControlModes:
    """Control mode identifiers."""

    MANUAL = "manual"
    AUTOPILOT = "autopilot"
    LANE_KEEPING = "lane_keeping"
    EMERGENCY_STOP = "emergency_stop"


class Launcher:
    """LKAS launcher configuration."""

    ENABLE_FOOTER = True  # Enable persistent terminal footer display


# Convenience exports
__all__ = [
    'DetectorTypes',
    'ControlModes',
    'Launcher',
]
