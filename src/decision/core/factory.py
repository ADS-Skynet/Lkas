"""
Factory pattern for creating steering controllers.

Centralizes controller instantiation and configuration.
"""

from .interfaces import SteeringController


class ControllerFactory:
    """
    Factory for creating steering controller instances.

    Usage:
        factory = ControllerFactory()
        controller = factory.create('pure_pursuit', gain=0.8)
    """

    def __init__(self, config=None):
        """
        Initialize factory with optional configuration.

        Args:
            config: System configuration object (optional)
                    If provided, default parameters are loaded from config
        """
        self.config = config

    def create(self, controller_type: str | None = None, **kwargs) -> SteeringController:
        """
        Create a steering controller instance.

        Args:
            controller_type: Type of controller ('pure_pursuit')
                           If None and config provided, uses config default
            **kwargs: Controller-specific parameters to override defaults

        Returns:
            SteeringController instance

        Raises:
            ValueError: If controller_type is invalid
        """
        if controller_type is None:
            if self.config and hasattr(self.config, 'decision'):
                controller_type = getattr(self.config.decision, 'controller_method', 'pure_pursuit')
            else:
                controller_type = 'pure_pursuit'

        controller_type = controller_type.lower()

        if controller_type == "pure_pursuit":
            return self._create_pure_pursuit_controller(**kwargs)
        else:
            raise ValueError(
                f"Unknown controller type: {controller_type}. "
                f"Available: {self.list_available_controllers()}"
            )

    def _create_pure_pursuit_controller(self, **kwargs) -> SteeringController:
        """
        Create Pure Pursuit path tracking controller.

        Designed for DL segmentation-based lane detection. Uses a lookahead
        point on the center path polynomial to compute steering.

        Args:
            **kwargs: Override default parameters
                gain (or kp): Main steering gain (default: 0.8)
                lookahead_ratio: Lookahead distance as fraction of image height (default: 0.4)
                heading_gain (or kd): Heading angle correction gain (default: 0.15)
                image_width: Camera image width (default: 1280)
                image_height: Camera image height (default: 720)

        Returns:
            PurePursuitController instance
        """
        from lkas.decision.method.pure_pursuit_controller import PurePursuitController

        if self.config and hasattr(self.config, 'decision'):
            cfg = getattr(self.config.decision, 'pure_pursuit', None)
            if cfg:
                params = {
                    "gain": kwargs.get("gain", kwargs.get("kp", getattr(cfg, 'gain', 0.8))),
                    "lookahead_ratio": kwargs.get("lookahead_ratio", getattr(cfg, 'lookahead_ratio', 0.4)),
                    "heading_gain": kwargs.get("heading_gain", kwargs.get("kd", getattr(cfg, 'heading_gain', 0.15))),
                    "image_width": kwargs.get("image_width", getattr(cfg, 'image_width', 1280)),
                    "image_height": kwargs.get("image_height", getattr(cfg, 'image_height', 720)),
                    "camera_offset_x": kwargs.get("camera_offset_x", 0),
                }
            else:
                params = {
                    "gain": kwargs.get("gain", kwargs.get("kp", 0.8)),
                    "lookahead_ratio": kwargs.get("lookahead_ratio", 0.4),
                    "heading_gain": kwargs.get("heading_gain", kwargs.get("kd", 0.15)),
                    "image_width": kwargs.get("image_width", 1280),
                    "image_height": kwargs.get("image_height", 720),
                    "camera_offset_x": kwargs.get("camera_offset_x", 0),
                }
        else:
            params = {
                "gain": kwargs.get("gain", kwargs.get("kp", 0.8)),
                "lookahead_ratio": kwargs.get("lookahead_ratio", 0.4),
                "heading_gain": kwargs.get("heading_gain", kwargs.get("kd", 0.15)),
                "image_width": kwargs.get("image_width", 1280),
                "image_height": kwargs.get("image_height", 720),
                "camera_offset_x": kwargs.get("camera_offset_x", 0),
            }

        return PurePursuitController(**params)

    @staticmethod
    def list_available_controllers() -> list[str]:
        """
        List all available controller types.

        Returns:
            List of controller type strings
        """
        return ["pure_pursuit"]
