"""Configuration management for the faker system."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class FakerConfig:
    """Main configuration for the faker system."""

    # Server settings
    host: str = "localhost"
    port: int = 8899
    debug: bool = False

    # Model definitions directory
    models_dir: str = "configs/models"
    scenarios_dir: str = "configs/scenarios"

    # Default model capabilities
    default_capabilities: Dict[str, Any] = field(default_factory=lambda: {
        "supports_json_mode": True,
        "supports_temperature": True,
        "supports_system": True,
        "max_tokens": 4096
    })

    def __post_init__(self):
        """Set absolute paths for directories."""
        base_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(base_dir, self.models_dir)
        self.scenarios_dir = os.path.join(base_dir, self.scenarios_dir)

    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        model_path = os.path.join(self.models_dir, f"{model_name}.yaml")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model config '{model_name}' not found at {model_path}")

        with open(model_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def list_available_models(self) -> list[str]:
        """List all available model configuration files."""
        if not os.path.exists(self.models_dir):
            return []

        model_files = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                model_name = filename.replace('.yaml', '').replace('.yml', '')
                model_files.append(model_name)

        return model_files

    def list_available_scenarios(self) -> list[str]:
        """List all available scenario configuration files."""
        if not os.path.exists(self.scenarios_dir):
            return []

        scenario_files = []
        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                scenario_name = filename.replace('.yaml', '').replace('.yml', '')
                scenario_files.append(scenario_name)

        return scenario_files

    def create_model_config(self, model_name: str, config: Dict[str, Any]):
        """Create a new model configuration file."""
        os.makedirs(self.models_dir, exist_ok=True)

        model_path = os.path.join(self.models_dir, f"{model_name}.yaml")
        with open(model_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    def create_scenario_config(self, scenario_name: str, config: Dict[str, Any]):
        """Create a new scenario configuration file."""
        os.makedirs(self.scenarios_dir, exist_ok=True)

        scenario_path = os.path.join(self.scenarios_dir, f"{scenario_name}.yaml")
        with open(scenario_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    @classmethod
    def from_yaml(cls, config_path: str) -> "FakerConfig":
        """Load faker configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, output_path: str):
        """Save faker configuration to YAML file."""
        config_dict = {
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            'models_dir': self.models_dir,
            'scenarios_dir': self.scenarios_dir,
            'default_capabilities': self.default_capabilities
        }

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)