"""
Configuration Repository Interface

Abstract interface for configuration loading.
Supports multiple sources: YAML, Environment, Dict (testing)
"""

from abc import ABC, abstractmethod


class IConfigRepository(ABC):
    """
    Abstract interface for configuration repositories.

    Implementations: YAML, Environment, Dictionary (for testing)
    """

    @abstractmethod
    def load_config(self) -> object:
        """
        Load configuration.

        Returns:
            Config object
        """
        pass

    @abstractmethod
    def save_config(self, config: object) -> None:
        """
        Save configuration.

        Args:
            config: Config object to save
        """
        pass

    @abstractmethod
    def get_config_path(self) -> str:
        """Get configuration file path."""
        pass
