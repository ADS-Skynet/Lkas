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
        controller = factory.create('pid', kp=0.5, ki=0.01, kd=0.1)
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
            controller_type: Type of controller ('pd', 'pid', 'mpc', 'pfc')
                           If None and config provided, uses config default
            **kwargs: Controller-specific parameters to override defaults

        Returns:
            SteeringController instance

        Raises:
            ValueError: If controller_type is invalid

        Examples:
            # Create PD controller with default params
            controller = factory.create('pd')

            # Create PID with custom gains
            controller = factory.create('pid', kp=0.8, ki=0.02, kd=0.15)

            # Create MPC with custom horizon
            controller = factory.create('mpc', horizon=15)
        """
        # Use config default if available and controller_type not specified
        if controller_type is None:
            if self.config and hasattr(self.config, 'decision'):
                controller_type = getattr(self.config.decision, 'controller_method', 'pid')
            else:
                controller_type = 'pid'  # Fallback default

        controller_type = controller_type.lower()

        if controller_type == "pd":
            return self._create_pd_controller(**kwargs)
        elif controller_type == "pid":
            return self._create_pid_controller(**kwargs)
        elif controller_type == "mpc":
            return self._create_mpc_controller(**kwargs)
        elif controller_type == "pfc":
            return self._create_pfc_controller(**kwargs)
        else:
            raise ValueError(
                f"Unknown controller type: {controller_type}. "
                f"Available: {self.list_available_controllers()}"
            )

    def _create_pd_controller(self, **kwargs) -> SteeringController:
        """
        Create PD (Proportional-Derivative) controller.

        Args:
            **kwargs: Override default parameters
                kp: Proportional gain (default: 0.5)
                kd: Derivative gain (default: 0.1)

        Returns:
            PDController instance
        """
        from lkas.decision.method.pd_controller import PDController

        # Get defaults from config if available
        if self.config and hasattr(self.config, 'decision'):
            cfg = getattr(self.config.decision, 'pd', None)
            if cfg:
                params = {
                    "kp": kwargs.get("kp", getattr(cfg, 'kp', 0.5)),
                    "kd": kwargs.get("kd", getattr(cfg, 'kd', 0.1)),
                }
            else:
                params = {
                    "kp": kwargs.get("kp", 0.5),
                    "kd": kwargs.get("kd", 0.1),
                }
        else:
            params = {
                "kp": kwargs.get("kp", 0.5),
                "kd": kwargs.get("kd", 0.1),
            }

        return PDController(**params)

    def _create_pid_controller(self, **kwargs) -> SteeringController:
        """
        Create PID (Proportional-Integral-Derivative) controller.

        Args:
            **kwargs: Override default parameters
                kp: Proportional gain (default: 0.5)
                ki: Integral gain (default: 0.01)
                kd: Derivative gain (default: 0.1)

        Returns:
            PIDController instance
        """
        from lkas.decision.method.pid_controller import PIDController

        # Get defaults from config if available
        if self.config and hasattr(self.config, 'decision'):
            cfg = getattr(self.config.decision, 'pid', None)
            if cfg:
                params = {
                    "kp": kwargs.get("kp", getattr(cfg, 'kp', 0.5)),
                    "ki": kwargs.get("ki", getattr(cfg, 'ki', 0.01)),
                    "kd": kwargs.get("kd", getattr(cfg, 'kd', 0.1)),
                }
            else:
                params = {
                    "kp": kwargs.get("kp", 0.5),
                    "ki": kwargs.get("ki", 0.01),
                    "kd": kwargs.get("kd", 0.1),
                }
        else:
            params = {
                "kp": kwargs.get("kp", 0.5),
                "ki": kwargs.get("ki", 0.01),
                "kd": kwargs.get("kd", 0.1),
            }

        return PIDController(**params)

    def _create_mpc_controller(self, **kwargs) -> SteeringController:
        """
        Create MPC (Model Predictive Control) controller.

        Args:
            **kwargs: Override default parameters
                prediction_horizon: Number of steps to predict (default: 10)
                control_horizon: Number of control steps (default: 5)
                dt: Time step in seconds (default: 0.1)
                q_lateral: Lateral offset cost weight (default: 1.0)
                q_heading: Heading angle cost weight (default: 0.5)
                r_steering: Steering effort cost weight (default: 0.1)

        Returns:
            MPCController instance
        """
        from lkas.decision.method.mpc_controller import MPCController

        # Get defaults from config if available
        if self.config and hasattr(self.config, 'decision'):
            cfg = getattr(self.config.decision, 'mpc', None)
            if cfg:
                params = {
                    "prediction_horizon": kwargs.get("prediction_horizon", getattr(cfg, 'prediction_horizon', 10)),
                    "control_horizon": kwargs.get("control_horizon", getattr(cfg, 'control_horizon', 5)),
                    "dt": kwargs.get("dt", getattr(cfg, 'dt', 0.1)),
                    "q_lateral": kwargs.get("q_lateral", getattr(cfg, 'q_lateral', 1.0)),
                    "q_heading": kwargs.get("q_heading", getattr(cfg, 'q_heading', 0.5)),
                    "r_steering": kwargs.get("r_steering", getattr(cfg, 'r_steering', 0.1)),
                }
            else:
                params = {
                    "prediction_horizon": kwargs.get("prediction_horizon", 10),
                    "control_horizon": kwargs.get("control_horizon", 5),
                    "dt": kwargs.get("dt", 0.1),
                    "q_lateral": kwargs.get("q_lateral", 1.0),
                    "q_heading": kwargs.get("q_heading", 0.5),
                    "r_steering": kwargs.get("r_steering", 0.1),
                }
        else:
            params = {
                "prediction_horizon": kwargs.get("prediction_horizon", 10),
                "control_horizon": kwargs.get("control_horizon", 5),
                "dt": kwargs.get("dt", 0.1),
                "q_lateral": kwargs.get("q_lateral", 1.0),
                "q_heading": kwargs.get("q_heading", 0.5),
                "r_steering": kwargs.get("r_steering", 0.1),
            }

        return MPCController(**params)

    def _create_pfc_controller(self, **kwargs) -> SteeringController:
        """
        Create PFC (Predictive Functional Control) controller.

        Args:
            **kwargs: Override default parameters
                prediction_horizon: Number of steps to predict (default: 8)
                n_basis_functions: Number of basis functions (default: 3)
                dt: Time step in seconds (default: 0.1)

        Returns:
            PFCController instance
        """
        # Import will be added when PFC is implemented
        raise NotImplementedError(
            "PFC controller is not yet implemented. "
            "Available controllers: pd, pid, mpc"
        )

    @staticmethod
    def list_available_controllers() -> list[str]:
        """
        List all available controller types.

        Returns:
            List of controller type strings
        """
        return ["pd", "pid", "mpc"]  # Add "pfc" when implemented
