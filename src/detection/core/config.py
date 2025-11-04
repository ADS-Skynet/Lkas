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
            if 'cv_detector' in data:
                cv_data = data['cv_detector']
                cv_cfg = CVDetectorConfig(
                    canny_low=cv_data.get('canny_low', cv_cfg.canny_low),
                    canny_high=cv_data.get('canny_high', cv_cfg.canny_high),
                    hough_rho=cv_data.get('hough_rho', cv_cfg.hough_rho),
                    hough_theta=cv_data.get('hough_theta', cv_cfg.hough_theta),
                    hough_threshold=cv_data.get('hough_threshold', cv_cfg.hough_threshold),
                    hough_min_line_len=cv_data.get('hough_min_line_len', cv_cfg.hough_min_line_len),
                    hough_max_line_gap=cv_data.get('hough_max_line_gap', cv_cfg.hough_max_line_gap),
                    smoothing_factor=cv_data.get('smoothing_factor', cv_cfg.smoothing_factor),
                    min_slope=cv_data.get('min_slope', cv_cfg.min_slope),
                    roi_bottom_left_x=cv_data.get('roi_bottom_left_x', cv_cfg.roi_bottom_left_x),
                    roi_top_left_x=cv_data.get('roi_top_left_x', cv_cfg.roi_top_left_x),
                    roi_top_right_x=cv_data.get('roi_top_right_x', cv_cfg.roi_top_right_x),
                    roi_bottom_right_x=cv_data.get('roi_bottom_right_x', cv_cfg.roi_bottom_right_x),
                    roi_top_y=cv_data.get('roi_top_y', cv_cfg.roi_top_y),
                )

            # Parse DL detector config
            dl_cfg = DLDetectorConfig()
            if 'dl_detector' in data:
                dl_data = data['dl_detector']

                # Convert input_size list to tuple if needed
                input_size = dl_data.get('input_size', dl_cfg.input_size)
                if isinstance(input_size, list):
                    input_size = tuple(input_size)

                dl_cfg = DLDetectorConfig(
                    model_type=dl_data.get('model_type', dl_cfg.model_type),
                    input_size=input_size,
                    threshold=dl_data.get('threshold', dl_cfg.threshold),
                    device=dl_data.get('device', dl_cfg.device),
                )

            # Parse analyzer config
            analyzer_cfg = AnalyzerConfig()
            if 'lane_analyzer' in data:
                ana_data = data['lane_analyzer']
                analyzer_cfg = AnalyzerConfig(
                    drift_threshold=ana_data.get('drift_threshold', analyzer_cfg.drift_threshold),
                    departure_threshold=ana_data.get('departure_threshold', analyzer_cfg.departure_threshold),
                    lane_width_meters=ana_data.get('lane_width_meters', analyzer_cfg.lane_width_meters),
                    max_heading_degrees=ana_data.get('max_heading_degrees', analyzer_cfg.max_heading_degrees),
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

            # Parse visualization config
            viz_cfg = VisualizationConfig()
            if 'visualization' in data:
                viz_data = data['visualization']

                # Parse colors if present (convert list to tuple)
                def parse_color(color_val, default):
                    if isinstance(color_val, list):
                        return tuple(color_val)
                    return default

                viz_cfg = VisualizationConfig(
                    show_spectator_overlay=viz_data.get('show_spectator_overlay', viz_cfg.show_spectator_overlay),
                    follow_with_spectator=viz_data.get('follow_with_spectator', viz_cfg.follow_with_spectator),
                    alert_blink_frequency=viz_data.get('alert_blink_frequency', viz_cfg.alert_blink_frequency),
                    color_left_lane=parse_color(viz_data.get('color_left_lane'), viz_cfg.color_left_lane),
                    color_right_lane=parse_color(viz_data.get('color_right_lane'), viz_cfg.color_right_lane),
                    color_lane_fill=parse_color(viz_data.get('color_lane_fill'), viz_cfg.color_lane_fill),
                    color_centered=parse_color(viz_data.get('color_centered'), viz_cfg.color_centered),
                    color_drift=parse_color(viz_data.get('color_drift'), viz_cfg.color_drift),
                    color_departure=parse_color(viz_data.get('color_departure'), viz_cfg.color_departure),
                    hud_font_scale=viz_data.get('hud_font_scale', viz_cfg.hud_font_scale),
                    hud_thickness=viz_data.get('hud_thickness', viz_cfg.hud_thickness),
                    hud_margin=viz_data.get('hud_margin', viz_cfg.hud_margin),
                )

            # Parse detection method from system section
            detection_method = "cv"
            if 'system' in data:
                detection_method = data['system'].get('detection_method', 'cv')

            # Create config object
            config = Config(
                carla=carla_cfg,
                camera=camera_cfg,
                cv_detector=cv_cfg,
                dl_detector=dl_cfg,
                analyzer=analyzer_cfg,
                controller=controller_cfg,
                throttle_policy=throttle_cfg,
                visualization=viz_cfg,
                detection_method=detection_method,
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
            # Convert position and rotation tuples to dict format
            position = config.camera.position
            rotation = config.camera.rotation

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
                    'position': {
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                    },
                    'rotation': {
                        'pitch': rotation[0],
                        'yaw': rotation[1],
                        'roll': rotation[2],
                    },
                },
                'cv_detector': {
                    'canny_low': config.cv_detector.canny_low,
                    'canny_high': config.cv_detector.canny_high,
                    'hough_rho': config.cv_detector.hough_rho,
                    'hough_theta': config.cv_detector.hough_theta,
                    'hough_threshold': config.cv_detector.hough_threshold,
                    'hough_min_line_len': config.cv_detector.hough_min_line_len,
                    'hough_max_line_gap': config.cv_detector.hough_max_line_gap,
                    'smoothing_factor': config.cv_detector.smoothing_factor,
                    'min_slope': config.cv_detector.min_slope,
                    'roi_bottom_left_x': config.cv_detector.roi_bottom_left_x,
                    'roi_top_left_x': config.cv_detector.roi_top_left_x,
                    'roi_top_right_x': config.cv_detector.roi_top_right_x,
                    'roi_bottom_right_x': config.cv_detector.roi_bottom_right_x,
                    'roi_top_y': config.cv_detector.roi_top_y,
                },
                'dl_detector': {
                    'model_type': config.dl_detector.model_type,
                    'input_size': list(config.dl_detector.input_size),
                    'threshold': config.dl_detector.threshold,
                    'device': config.dl_detector.device,
                },
                'lane_analyzer': {
                    'drift_threshold': config.analyzer.drift_threshold,
                    'departure_threshold': config.analyzer.departure_threshold,
                    'lane_width_meters': config.analyzer.lane_width_meters,
                    'max_heading_degrees': config.analyzer.max_heading_degrees,
                    'kp': config.controller.kp,
                    'kd': config.controller.kd,
                },
                'throttle_policy': {
                    'base': config.throttle_policy.base,
                    'min': config.throttle_policy.min,
                    'steer_threshold': config.throttle_policy.steer_threshold,
                    'steer_max': config.throttle_policy.steer_max,
                },
                'visualization': {
                    'show_spectator_overlay': config.visualization.show_spectator_overlay,
                    'follow_with_spectator': config.visualization.follow_with_spectator,
                    'alert_blink_frequency': config.visualization.alert_blink_frequency,
                    'color_left_lane': list(config.visualization.color_left_lane),
                    'color_right_lane': list(config.visualization.color_right_lane),
                    'color_lane_fill': list(config.visualization.color_lane_fill),
                    'color_centered': list(config.visualization.color_centered),
                    'color_drift': list(config.visualization.color_drift),
                    'color_departure': list(config.visualization.color_departure),
                    'hud_font_scale': config.visualization.hud_font_scale,
                    'hud_thickness': config.visualization.hud_thickness,
                    'hud_margin': config.visualization.hud_margin,
                },
                'system': {
                    'detection_method': config.detection_method,
                },
            }

            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)

            return True

        except Exception as e:
            print(f"Error saving config: {e}")
            return False
