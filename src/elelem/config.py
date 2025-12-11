"""
Configuration loading system for Elelem
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


def expand_parameterized_models(models: Dict[str, Any]) -> Dict[str, Any]:
    """Expand parameterized model definitions into individual models."""
    expanded_models = {}

    for model_key, model_config in models.items():
        # Check for parameterization syntax: @[value1,value2,value3] or @b[true,false] for booleans
        param_pattern = r'@(b)?\[([^\]]+)\]'
        matches = list(re.finditer(param_pattern, model_key))

        if not matches:
            # No parameterization, keep as-is
            expanded_models[model_key] = model_config
            continue

        # Extract parameter values and generate all combinations
        param_values = []
        param_names = []

        for match in matches:
            is_boolean = match.group(1) == 'b'  # Check if @b syntax was used
            values_str = match.group(2)  # Now group 2 contains the values
            values = [v.strip() for v in values_str.split(',')]

            # Convert boolean strings to actual booleans if @b syntax was used
            if is_boolean:
                boolean_values = []
                for v in values:
                    if v.lower() == 'true':
                        boolean_values.append(True)
                    elif v.lower() == 'false':
                        boolean_values.append(False)
                    else:
                        raise ValueError(f"Invalid boolean value '{v}' in @b[...] syntax. Use 'true' or 'false'.")
                param_values.append(boolean_values)
            else:
                param_values.append(values)

            # Extract parameter name from the pattern before @
            start_pos = match.start()
            # Look backwards to find parameter name (e.g., "reasoning=" before "@[low,high]")
            param_part = model_key[:start_pos]

            # Find the parameter name by looking for "key=" pattern (accounting for @b syntax)
            param_match = re.search(r'([^?&=]+)=@b?\[', model_key[max(0, start_pos-20):start_pos+10])
            if param_match:
                param_name = param_match.group(1)
            else:
                param_name = f"param{len(param_names)}"

            param_names.append(param_name)

        # Generate all combinations
        from itertools import product

        for combination in product(*param_values):
            # Create new model key by substituting parameter values
            new_model_key = model_key
            new_model_config = model_config.copy()

            for i, (match, param_name, param_value) in enumerate(zip(matches, param_names, combination)):
                # Replace @[...] or @b[...] with actual value (convert booleans to lowercase strings for model key)
                key_value = str(param_value).lower() if isinstance(param_value, bool) else str(param_value)
                new_model_key = new_model_key.replace(match.group(0), key_value, 1)

                # Substitute $parameter_name in config values (param_value can now be boolean or string)
                new_model_config = substitute_parameters(new_model_config, {param_name: param_value})

            expanded_models[new_model_key] = new_model_config

    return expanded_models


def substitute_parameters(config: Any, params: Dict[str, Any]) -> Any:
    """Recursively substitute $parameter references in config values."""
    if isinstance(config, dict):
        return {k: substitute_parameters(v, params) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_parameters(item, params) for item in config]
    elif isinstance(config, str):
        # Replace $parameter_name with actual value
        for param_name, param_value in params.items():
            placeholder = f"${param_name}"
            if placeholder in config:
                # If the entire string is just the placeholder, return the actual value (preserving type)
                if config == placeholder:
                    return param_value
                else:
                    # For partial replacements, convert to string
                    config = config.replace(placeholder, str(param_value))
        return config
    else:
        return config


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
                            # Handle new metadata structure with model_reference
                            if 'metadata' in model_config and isinstance(model_config['metadata'], dict):
                                metadata_obj = model_config['metadata']
                                model_reference = metadata_obj.get('model_reference')
                                model_configuration = metadata_obj.get('model_configuration')

                                if model_reference and model_reference in metadata_definitions:
                                    metadata = metadata_definitions[model_reference].copy()
                                    # Merge metadata into display_metadata
                                    model_config.setdefault('display_metadata', {}).update(metadata)

                                # Add model_configuration to display_metadata if present
                                if model_configuration:
                                    model_config.setdefault('display_metadata', {})['model_configuration'] = model_configuration

                            merged_config['models'][model_key] = model_config

            # Expand parameterized models after all models are loaded
            merged_config['models'] = expand_parameterized_models(merged_config['models'])

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
                        'timeout': candidate.get('timeout'),  # Candidate timeout override
                        'priority': candidate.get('priority')  # For benchmark routing (e.g., always_first)
                    }
                else:
                    # Inline definition (shouldn't happen in new format, but handle it)
                    resolved = candidate.copy()
                
                resolved_candidates.append(resolved)
            
            # Extract routing configuration for benchmark-based reordering
            routing = config.get('routing', {})

            return {
                'candidates': resolved_candidates,
                'timeout': config.get('timeout'),
                'routing': {
                    'speed_weight': routing.get('speed_weight', 1.0),
                    'min_tokens_per_sec': routing.get('min_tokens_per_sec', 0.0)
                }
            }
        else:
            # Regular model - wrap in single candidate structure (no routing needed)
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
                'timeout': config.get('timeout'),
                'routing': None  # Regular models don't use routing
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

    @property
    def chunk_timeout_seconds(self) -> int:
        """Get chunk timeout setting for streaming cold start detection.

        Default is 15 seconds - enough for normal operations but fast enough
        to detect serverless cold starts.
        """
        return self._config.get("chunk_timeout_seconds", 15)

    def get_candidate_chunk_timeout(self, candidate: Dict[str, Any], model_config: Dict[str, Any]) -> int:
        """Get chunk_timeout for streaming with a specific candidate.

        For streaming responses, this timeout applies per-chunk to detect cold starts
        and stream stalls. Uses global chunk_timeout_seconds (default 15s) if not
        overridden at candidate or model level.

        Timeout hierarchy:
        1. Candidate-level chunk_timeout
        2. Model-level chunk_timeout
        3. Global chunk_timeout_seconds (default 15s)

        Args:
            candidate: Candidate configuration dict
            model_config: Model configuration dict from get_model_config()

        Returns:
            Chunk timeout in seconds
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

        # Try candidate chunk_timeout first
        candidate_chunk_timeout = parse_timeout(candidate.get('chunk_timeout'))
        if candidate_chunk_timeout is not None:
            return candidate_chunk_timeout

        # Try model chunk_timeout
        model_chunk_timeout = parse_timeout(model_config.get('chunk_timeout'))
        if model_chunk_timeout is not None:
            return model_chunk_timeout

        # Fall back to global chunk timeout
        return self.chunk_timeout_seconds

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