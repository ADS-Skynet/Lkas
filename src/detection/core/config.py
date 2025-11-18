"""
DEPRECATED: Configuration has moved to skynet_common.config

This file provides backward compatibility. Please update your imports to:
    from skynet_common.config import ConfigManager, Config, etc.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from lkas.detection.core.config is deprecated. "
    "Please use 'from skynet_common.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location for backward compatibility
from skynet_common.config import (
    Config,
    ConfigManager,
    CommunicationConfig,
    CARLAConfig,
    CameraConfig,
    CVDetectorConfig,
    DLDetectorConfig,
    AnalyzerConfig,
    ControllerConfig,
    ThrottlePolicyConfig,
    VisualizationConfig,
    get_project_root,
    DEFAULT_CONFIG_PATH,
)

__all__ = [
    "Config",
    "ConfigManager",
    "CommunicationConfig",
    "CARLAConfig",
    "CameraConfig",
    "CVDetectorConfig",
    "DLDetectorConfig",
    "AnalyzerConfig",
    "ControllerConfig",
    "ThrottlePolicyConfig",
    "VisualizationConfig",
    "get_project_root",
    "DEFAULT_CONFIG_PATH",
]
