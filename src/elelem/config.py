"""
Configuration loading system for Elelem
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
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
        """Load models and providers configuration from YAML files.
        
        Loads main models.yaml and auto-discovers provider files in providers/ directory.
        """
        models_path = Path(__file__).parent / "models.yaml"
        providers_dir = Path(__file__).parent / "providers"
        
        # Initialize merged configuration
        merged_config = {"providers": {}, "models": {}}
        
        try:
            # Load main models.yaml if it exists
            if models_path.exists():
                with open(models_path, 'r') as f:
                    main_config = yaml.safe_load(f) or {}
                
                # Merge main config
                merged_config["providers"].update(main_config.get("providers", {}))
                merged_config["models"].update(main_config.get("models", {}))
            
            # Auto-discover and merge provider files
            if providers_dir.exists():
                for yaml_file in providers_dir.glob("*.yaml"):
                    with open(yaml_file, 'r') as f:
                        provider_config = yaml.safe_load(f) or {}
                    
                    provider_name = yaml_file.stem  # filename without .yaml
                    
                    # Merge provider config
                    if 'provider' in provider_config:
                        merged_config['providers'][provider_name] = provider_config['provider']
                    
                    # Merge models
                    if 'models' in provider_config:
                        merged_config['models'].update(provider_config['models'])
            
            return merged_config
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
        
    def get_provider_default_params(self, provider: str) -> Dict[str, Any]:
        """Get default parameters for a specific provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("default_params", {})
        
    def get_provider_max_tokens_default(self, provider: str) -> Optional[int]:
        """Get default max_tokens for a specific provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("max_tokens_default")
        
    def get_model_extra_body(self, model_name: str) -> Dict[str, Any]:
        """Get extra_body parameters for a specific model."""
        models = self.models
        if model_name not in models:
            return {}
        return models[model_name].get("extra_body", {})
        
    @property
    def models(self) -> Dict[str, Any]:
        """Get all model configurations."""
        return self._models_config.get("models", {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get unified candidate structure for any model (regular or virtual).
        
        Args:
            model_name: Model name (e.g., "openai:gpt-4.1" or "virtual:gpt-oss-120b")
            
        Returns:
            Dict with 'candidates' list and optional 'timeout'
        """
        models = self.models
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
            
        config = models[model_name]
        
        if 'candidates' in config:
            # Virtual model - resolve model references
            resolved_candidates = []
            for candidate in config['candidates']:
                if 'model' in candidate:
                    # Reference to another model
                    ref_model_name = candidate['model']
                    if ref_model_name not in models:
                        raise ValueError(f"Referenced model '{ref_model_name}' not found")
                    
                    ref_config = models[ref_model_name]
                    resolved = {
                        'original_model_ref': ref_model_name,  # Keep track of original reference
                        'provider': ref_config['provider'],
                        'model_id': ref_config['model_id'],
                        'capabilities': ref_config.get('capabilities', {}),
                        'cost': ref_config.get('cost'),
                        'default_params': ref_config.get('default_params', {}),
                        'timeout': candidate.get('timeout')  # Candidate timeout override
                    }
                else:
                    # Inline definition (shouldn't happen in new format, but handle it)
                    resolved = candidate.copy()
                
                resolved_candidates.append(resolved)
            
            return {
                'candidates': resolved_candidates,
                'timeout': config.get('timeout')
            }
        else:
            # Regular model - wrap in single candidate structure
            return {
                'candidates': [{
                    'provider': config['provider'],
                    'model_id': config['model_id'],
                    'capabilities': config.get('capabilities', {}),
                    'cost': config.get('cost'),
                    'default_params': config.get('default_params', {}),
                    'timeout': config.get('timeout')
                }],
                'timeout': config.get('timeout')
            }
    
    def get_candidate_timeout(self, candidate: Dict[str, Any], model_config: Dict[str, Any]) -> int:
        """Get timeout for a specific candidate using hierarchy.
        
        Timeout hierarchy:
        1. Candidate-level timeout
        2. Model-level timeout
        3. Global timeout
        
        Args:
            candidate: Candidate configuration dict
            model_config: Model configuration dict from get_model_config()
            
        Returns:
            Timeout in seconds
        """
        # Parse timeout strings to seconds
        def parse_timeout(timeout_str):
            if not timeout_str:
                return None
            if isinstance(timeout_str, int):
                return timeout_str
            if isinstance(timeout_str, str) and timeout_str.endswith('s'):
                return int(timeout_str[:-1])
            return int(timeout_str)
        
        # Try candidate timeout first
        candidate_timeout = parse_timeout(candidate.get('timeout'))
        if candidate_timeout is not None:
            return candidate_timeout
            
        # Try model timeout
        model_timeout = parse_timeout(model_config.get('timeout'))
        if model_timeout is not None:
            return model_timeout
            
        # Fall back to global timeout
        return self.timeout_seconds