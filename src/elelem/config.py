"""
Configuration loading system for Elelem
"""

import json
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Configuration loader for Elelem settings."""
    
    def __init__(self):
        self._config = self._load_config()
        self._models_config = self._load_models_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        config_path = Path(__file__).parent / "config.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load Elelem configuration: {e}")
    
    def _load_models_config(self) -> Dict[str, Any]:
        """Load models and providers configuration from YAML file."""
        models_path = Path(__file__).parent / "models.yaml"
        
        try:
            with open(models_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except (FileNotFoundError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load Elelem models configuration: {e}")
    
    @property
    def retry_settings(self) -> Dict[str, Any]:
        """Get retry settings."""
        return self._config.get("retry_settings", {})
        
    @property
    def providers(self) -> Dict[str, Any]:
        """Get provider configurations."""
        return self._models_config.get("providers", {})
        
    @property
    def timeout_seconds(self) -> int:
        """Get timeout setting."""
        return self._config.get("timeout_seconds", 120)
        
    @property
    def logging_level(self) -> str:
        """Get logging level."""
        return self._config.get("logging_level", "INFO")
        
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider, {})
        
    def get_provider_endpoint(self, provider: str) -> str:
        """Get endpoint for a specific provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("endpoint", "")
        
    def get_fallback_model(self, provider: str) -> str:
        """Get fallback model for a specific provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("fallback_model", "")
        
    def get_provider_default_params(self, provider: str) -> Dict[str, Any]:
        """Get default parameters for a specific provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("default_params", {})