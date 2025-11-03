"""
Configuration management for lane detection system.
Loads from YAML and provides type-safe access to settings.
"""

from dataclasses import dataclass, field
from typing import Tuple
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root directory by locating pyproject.toml.

    Returns:
        Path to project root directory
    """
    # Start from this file's location and walk up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume project root is 2 levels up from this file
    # (detection/core/config.py -> project root)
    return Path(__file__).resolve().parent.parent.parent


# Default config path at project root
DEFAULT_CONFIG_PATH = get_project_root() / "config.yaml"


@dataclass
class CARLAConfig:
    """CARLA simulator configuration."""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    vehicle_type: str = "vehicle.tesla.model3"


@dataclass
class CameraConfig:
    """Camera sensor configuration."""
    width: int = 800
    height: int = 600
    fov: float = 90.0
    position: Tuple[float, float, float] = (2.0, 0.0, 1.5)  # x, y, z
    rotation: Tuple[float, float, float] = (-10.0, 0.0, 0.0)  # pitch, yaw, roll


@dataclass
class CVDetectorConfig:
    """Computer Vision detector parameters."""
    canny_low: int = 50
    canny_high: int = 150
    hough_rho: int = 2
    hough_theta: float = 0.017453  # pi/180
    hough_threshold: int = 50
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 100
    smoothing_factor: float = 0.7
    min_slope: float = 0.5

    # ROI configuration (broader detection area)
    roi_bottom_left_x: float = 0.05  # fraction of width (bottom-left corner)
    roi_top_left_x: float = 0.35     # fraction of width (top-left corner) - wider than before
    roi_top_right_x: float = 0.65    # fraction of width (top-right corner) - wider than before
    roi_bottom_right_x: float = 0.95 # fraction of width (bottom-right corner)
    roi_top_y: float = 0.5           # fraction of height (look at top 50% of image)


@dataclass
class DLDetectorConfig:
    """Deep Learning detector parameters."""
    model_type: str = "pretrained"  # 'pretrained', 'simple', 'full'
    input_size: Tuple[int, int] = (256, 256)
    threshold: float = 0.5
    device: str = "auto"  # 'cpu', 'cuda', 'auto'


@dataclass
class AnalyzerConfig:
    """Lane analyzer configuration."""
    drift_threshold: float = 0.15
    departure_threshold: float = 0.35
    lane_width_meters: float = 3.7
    max_heading_degrees: float = 30.0


@dataclass
class ControllerConfig:
    """PD controller parameters."""
    kp: float = 0.5  # Proportional gain
    kd: float = 0.1  # Derivative gain


@dataclass
class ThrottlePolicyConfig:
    """Adaptive throttle policy configuration."""
    base: float = 0.14              # Base throttle when steering is minimal
    min: float = 0.05               # Minimum throttle during sharp turns
    steer_threshold: float = 0.15   # Steering magnitude to start reducing throttle
    steer_max: float = 0.70         # Maximum expected steering magnitude


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    show_spectator_overlay: bool = True
    follow_with_spectator: bool = False
    alert_blink_frequency: int = 10

    # Colors (BGR format for OpenCV)
    color_left_lane: Tuple[int, int, int] = (255, 0, 0)  # Blue
    color_right_lane: Tuple[int, int, int] = (0, 0, 255)  # Red
    color_lane_fill: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_centered: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_drift: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    color_departure: Tuple[int, int, int] = (0, 0, 255)  # Red

    # HUD settings
    hud_font_scale: float = 0.6
    hud_thickness: int = 2
    hud_margin: int = 20


@dataclass
class Config:
    """
    Master configuration container.

    Aggregates all subsystem configurations.
    """
    carla: CARLAConfig = field(default_factory=CARLAConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    cv_detector: CVDetectorConfig = field(default_factory=CVDetectorConfig)
    dl_detector: DLDetectorConfig = field(default_factory=DLDetectorConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    throttle_policy: ThrottlePolicyConfig = field(default_factory=ThrottlePolicyConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # General settings
    detection_method: str = "cv"  # 'cv' or 'dl'


class ConfigManager:
    """
    Configuration manager with YAML loading.

    Usage:
        # Load from project root config.yaml (default)
        config = ConfigManager.load()

        # Load from specific path
        config = ConfigManager.load('path/to/config.yaml')

        # Use built-in defaults only
        config = ConfigManager.load('default')

        host = config.carla.host
    """

    @staticmethod
    def load(config_path: str | Path | None = None) -> Config:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file.
                        If None, tries to load from project root config.yaml.
                        If "default", uses built-in defaults without loading file.

        Returns:
            Config object with loaded settings
        """
        # If explicitly asked for defaults, return without loading
        if config_path == "default":
            return Config()

        # If no path given, use project root config.yaml
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        path = Path(config_path)
        if not path.exists():
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return Config()

        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            if data is None:
                return Config()

            # Parse CARLA config
            carla_cfg = CARLAConfig()
            if 'carla' in data:
                carla_data = data['carla']
                carla_cfg = CARLAConfig(
                    host=carla_data.get('host', carla_cfg.host),
                    port=carla_data.get('port', carla_cfg.port),
                    timeout=carla_data.get('timeout', carla_cfg.timeout),
                    vehicle_type=carla_data.get('vehicle_type', carla_cfg.vehicle_type),
                )

            # Parse camera config
            camera_cfg = CameraConfig()
            if 'camera' in data:
                cam_data = data['camera']

                # Convert position dict to tuple if needed
                position = cam_data.get('position', camera_cfg.position)
                if isinstance(position, dict):
                    position = (position['x'], position['y'], position['z'])
                elif not isinstance(position, tuple):
                    position = tuple(position)

                # Convert rotation dict to tuple if needed
                rotation = cam_data.get('rotation', camera_cfg.rotation)
                if isinstance(rotation, dict):
                    rotation = (rotation['pitch'], rotation['yaw'], rotation['roll'])
                elif not isinstance(rotation, tuple):
                    rotation = tuple(rotation)

                camera_cfg = CameraConfig(
                    width=cam_data.get('width', camera_cfg.width),
                    height=cam_data.get('height', camera_cfg.height),
                    fov=cam_data.get('fov', camera_cfg.fov),
                    position=position,
                    rotation=rotation,
                )

            # Parse CV detector config
            cv_cfg = CVDetectorConfig()
            if 'lane_detection' in data and 'cv_params' in data['lane_detection']:
                cv_data = data['lane_detection']['cv_params']
                cv_cfg = CARLAConfig(**{k: v for k, v in cv_data.items() if hasattr(cv_cfg, k)})

            # Parse analyzer config
            analyzer_cfg = AnalyzerConfig()
            if 'lane_analyzer' in data:
                ana_data = data['lane_analyzer']
                analyzer_cfg = AnalyzerConfig(
                    drift_threshold=ana_data.get('drift_threshold', analyzer_cfg.drift_threshold),
                    departure_threshold=ana_data.get('departure_threshold', analyzer_cfg.departure_threshold),
                    lane_width_meters=ana_data.get('lane_width_meters', analyzer_cfg.lane_width_meters),
                )

            # Parse controller config
            controller_cfg = ControllerConfig()
            if 'lane_analyzer' in data:
                ctrl_data = data['lane_analyzer']
                controller_cfg = ControllerConfig(
                    kp=ctrl_data.get('kp', controller_cfg.kp),
                    kd=ctrl_data.get('kd', controller_cfg.kd),
                )

            # Parse throttle policy config
            throttle_cfg = ThrottlePolicyConfig()
            if 'throttle_policy' in data:
                throttle_data = data['throttle_policy']
                throttle_cfg = ThrottlePolicyConfig(
                    base=throttle_data.get('base', throttle_cfg.base),
                    min=throttle_data.get('min', throttle_cfg.min),
                    steer_threshold=throttle_data.get('steer_threshold', throttle_cfg.steer_threshold),
                    steer_max=throttle_data.get('steer_max', throttle_cfg.steer_max),
                )

            # Create config object
            config = Config(
                carla=carla_cfg,
                camera=camera_cfg,
                cv_detector=cv_cfg,
                analyzer=analyzer_cfg,
                controller=controller_cfg,
                throttle_policy=throttle_cfg,
            )

            return config

        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration.")
            return Config()

    @staticmethod
    def save(config: Config, config_path: str) -> bool:
        """
        Save configuration to YAML file.

        Args:
            config: Config object to save
            config_path: Path to save YAML file

        Returns:
            True if successful
        """
        try:
            data = {
                'carla': {
                    'host': config.carla.host,
                    'port': config.carla.port,
                    'timeout': config.carla.timeout,
                    'vehicle_type': config.carla.vehicle_type,
                },
                'camera': {
                    'width': config.camera.width,
                    'height': config.camera.height,
                    'fov': config.camera.fov,
                    'position': list(config.camera.position),
                    'rotation': list(config.camera.rotation),
                },
                'lane_detection': {
                    'method': config.detection_method,
                },
                'lane_analysis': {
                    'drift_threshold': config.analyzer.drift_threshold,
                    'departure_threshold': config.analyzer.departure_threshold,
                    'lane_width_meters': config.analyzer.lane_width_meters,
                },
            }

            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)

            return True

        except Exception as e:
            print(f"Error saving config: {e}")
            return False
