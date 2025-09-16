"""
Configuration loading system for Elelem
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


class Config:
    """Configuration loader for Elelem settings."""
    
    def __init__(self, extra_provider_dirs: Optional[List[str]] = None):
        self._config = self._load_config()
        # Check for extra provider dirs from environment variable
        if extra_provider_dirs is None:
            import os
            env_dirs = os.environ.get('ELELEM_EXTRA_PROVIDER_DIRS')
            if env_dirs:
                extra_provider_dirs = [d.strip() for d in env_dirs.split(',')]
        self._models_config = self._load_models_config(extra_provider_dirs)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        config_path = Path(__file__).parent / "config.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load Elelem configuration: {e}")
    
    def _load_models_config(self, extra_provider_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load models and providers configuration from YAML files.
        
        Loads main models.yaml, metadata, and auto-discovers provider files in providers/ directory.
        """
        models_path = Path(__file__).parent / "models.yaml"
        providers_dir = Path(__file__).parent / "providers"
        metadata_path = providers_dir / "_metadata.yaml"
        
        # Initialize merged configuration
        merged_config = {"providers": {}, "models": {}}
        
        # Load metadata definitions
        metadata_definitions = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_config = yaml.safe_load(f) or {}
                metadata_definitions = metadata_config.get("model_metadata", {})
            except (FileNotFoundError, yaml.YAMLError) as e:
                # Metadata is optional, continue without it
                pass
        
        try:
            # Load main models.yaml if it exists
            if models_path.exists():
                with open(models_path, 'r') as f:
                    main_config = yaml.safe_load(f) or {}
                
                # Merge main config
                merged_config["providers"].update(main_config.get("providers", {}))
                merged_config["models"].update(main_config.get("models", {}))
            
            # Auto-discover and merge provider files from main and extra directories
            provider_dirs_to_scan = []
            if providers_dir.exists():
                provider_dirs_to_scan.append(providers_dir)

            # Add extra provider directories if specified
            if extra_provider_dirs:
                for extra_dir in extra_provider_dirs:
                    extra_path = Path(extra_dir)
                    if extra_path.exists():
                        provider_dirs_to_scan.append(extra_path)

            # Scan all provider directories
            for current_dir in provider_dirs_to_scan:
                for yaml_file in current_dir.glob("*.yaml"):
                    # Skip metadata file
                    if yaml_file.name.startswith("_"):
                        continue
                        
                    with open(yaml_file, 'r') as f:
                        provider_config = yaml.safe_load(f) or {}
                    
                    provider_name = yaml_file.stem  # filename without .yaml
                    
                    # Merge provider config
                    if 'provider' in provider_config:
                        merged_config['providers'][provider_name] = provider_config['provider']
                    
                    # Merge models and resolve metadata references
                    if 'models' in provider_config:
                        for model_key, model_config in provider_config['models'].items():
                            # Check for metadata_ref and resolve it
                            if 'metadata_ref' in model_config and model_config['metadata_ref'] in metadata_definitions:
                                metadata = metadata_definitions[model_config['metadata_ref']].copy()
                                # Merge metadata into display_metadata
                                model_config.setdefault('display_metadata', {}).update(metadata)
                                # Remove the reference
                                del model_config['metadata_ref']
                            
                            merged_config['models'][model_key] = model_config
            
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
        """Get unified candidate structure for any model (regular, virtual, or dynamic).

        Args:
            model_name: Model name (e.g., "openai:gpt-4.1", "virtual:gpt-oss-120b", or "dynamic:{...}")

        Returns:
            Dict with 'candidates' list and optional 'timeout'
        """
        # Handle dynamic models
        if model_name.startswith("dynamic:{"):
            return self._parse_dynamic_model(model_name)

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
                        'display_metadata': ref_config.get('display_metadata', {}),
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
                    'display_metadata': config.get('display_metadata', {}),
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

    def _parse_dynamic_model(self, model_string: str) -> Dict[str, Any]:
        """Parse dynamic model specification from YAML syntax.

        Args:
            model_string: Dynamic model string starting with "dynamic:{...}"

        Returns:
            Dict with 'candidates' list and optional 'timeout' (same format as virtual models)
        """
        import yaml

        # Properly split on first colon to extract YAML content
        if not model_string.startswith("dynamic:"):
            raise ValueError("Dynamic model string must start with 'dynamic:'")

        _, yaml_content = model_string.split(":", 1)

        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in dynamic model specification: {e}")

        if not isinstance(config, dict):
            raise ValueError("Dynamic model specification must be a YAML dictionary")

        if 'candidates' not in config:
            raise ValueError("Dynamic model specification must include 'candidates' list")

        # Transform string candidates to dict format (consistent with virtual models)
        candidates = []
        for candidate in config['candidates']:
            if isinstance(candidate, str):
                # Simple string -> convert to dict format
                candidates.append({'model': candidate})
            elif isinstance(candidate, dict):
                # Already in dict format
                candidates.append(candidate)
            else:
                raise ValueError(f"Invalid candidate format: {candidate}")

        # Create config in same format as static virtual models
        parsed_config = {
            'candidates': candidates,
            'timeout': config.get('timeout'),
            'metadata_ref': config.get('metadata_ref')
        }

        # Apply the same resolution logic as virtual models
        models = self.models
        resolved_candidates = []

        for candidate in candidates:
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
                    'display_metadata': ref_config.get('display_metadata', {}),
                    'timeout': candidate.get('timeout')  # Candidate timeout override
                }
            else:
                # Inline definition (shouldn't happen, but handle it)
                resolved = candidate.copy()

            resolved_candidates.append(resolved)

        return {
            'candidates': resolved_candidates,
            'timeout': parsed_config.get('timeout'),
            'metadata_ref': parsed_config.get('metadata_ref')
        }