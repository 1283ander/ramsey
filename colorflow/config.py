"""
Configuration management for ColorFlow pipeline.
Loads settings from YAML file with validation and defaults.
"""

import yaml
import os
from typing import Dict, Any


class ColorFlowConfig:
    """Configuration manager for the ColorFlow pipeline."""

    def __init__(self, config_path: str = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        self.config_path = config_path
        self._config = None
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'sampling.frames_per_video')."""
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save_config(self, config_path: str = None):
        """Save current configuration to YAML file."""
        if config_path is None:
            config_path = self.config_path

        with open(config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

    @property
    def sampling_frames_per_video(self) -> int:
        """Number of frames to sample per video."""
        return self.get('sampling.frames_per_video', 200)

    @property
    def delta_e_threshold(self) -> float:
        """ΔE standard deviation threshold for convergence."""
        return self.get('convergence.delta_e_std_threshold', 1.5)

    @property
    def max_iterations(self) -> int:
        """Maximum number of optimization iterations."""
        return self.get('convergence.max_iterations', 3)

    @property
    def metrics_space(self) -> str:
        """Color space for metrics computation."""
        return self.get('convergence.metrics_space', 'lab')

    @property
    def ev_ema_alpha(self) -> float:
        """EMA smoothing factor for exposure adjustments."""
        return self.get('temporal_smoothing.ev_ema_alpha', 0.3)

    @property
    def wb_ema_alpha(self) -> float:
        """EMA smoothing factor for white balance adjustments."""
        return self.get('temporal_smoothing.wb_ema_alpha', 0.2)

    @property
    def saturation_ema_alpha(self) -> float:
        """EMA smoothing factor for saturation adjustments."""
        return self.get('temporal_smoothing.saturation_ema_alpha', 0.25)


# Global configuration instance
_config_instance = None


def get_config(config_path: str = None) -> ColorFlowConfig:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ColorFlowConfig(config_path)
    return _config_instance


def reload_config():
    """Reload global configuration."""
    global _config_instance
    _config_instance = ColorFlowConfig()