"""
LKAS Constants

Lane Keeping Assist System specific constants.
Only enum-like type identifiers are kept here.
All configurable values have been moved to config.yaml (common or module-specific).

Use ConfigManager.load() to access configuration values.
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


# Convenience exports
__all__ = [
    'DetectorTypes',
    'ControlModes',
]
