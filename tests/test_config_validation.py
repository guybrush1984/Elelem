"""
Configuration validation tests for Elelem providers and models.
Validates that all provider configurations are properly formed and consistent.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List, Set
import yaml

from elelem.config import Config


@pytest.fixture
def config():
    """Create Config instance for testing."""
    return Config()


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""

    def test_configuration_loads_successfully(self, config):
        """Test that configuration loads without errors."""
        assert config is not None
        assert len(config.models) > 0
        assert len(config.providers) > 0

    def test_all_models_have_valid_structure(self, config):
        """Test that all models have proper structure."""
        models = config.models
        issues = []

        for model_name, model_config in models.items():
            model_issues = self._validate_model_structure(model_name, model_config, models)
            issues.extend(model_issues)

        if issues:
            pytest.fail(f"Model structure issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

    def test_all_metadata_references_exist(self, config):
        """Test that all metadata.model_reference values reference existing metadata."""
        models = config.models
        issues = []

        # Load metadata definitions from _metadata.yaml
        metadata_path = Path(__file__).parent.parent / "src/elelem/providers/_metadata.yaml"
        metadata_definitions = set()

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_config = yaml.safe_load(f) or {}
                metadata_definitions = set(metadata_config.get("model_metadata", {}).keys())
            except Exception as e:
                pytest.fail(f"Could not load metadata definitions: {e}")

        # Check each model's metadata.model_reference
        for model_name, model_config in models.items():
            if 'metadata' in model_config and isinstance(model_config['metadata'], dict):
                model_reference = model_config['metadata'].get('model_reference')
                if model_reference and model_reference not in metadata_definitions:
                    issues.append(f"Model '{model_name}' references non-existent metadata '{model_reference}'")

        if issues:
            pytest.fail(f"Metadata reference issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

    def test_virtual_model_references_are_valid(self, config):
        """Test that virtual models reference existing base models."""
        models = config.models
        issues = []

        virtual_models = []
        regular_models = []

        for model_name, model_config in models.items():
            if 'candidates' in model_config:
                virtual_models.append(model_name)
            else:
                regular_models.append(model_name)

        # Check virtual model references (allow chaining, detect cycles)
        def detect_cycle(model_name: str, visited: set, path: list) -> list:
            """Detect circular references in virtual model chains."""
            if model_name in path:
                return path + [model_name]  # Cycle found
            if model_name in visited or model_name not in virtual_models:
                return []  # Already checked or not a virtual model

            visited.add(model_name)
            path = path + [model_name]

            for candidate in models[model_name].get('candidates', []):
                if 'model' in candidate:
                    ref = candidate['model']
                    if ref in virtual_models:
                        cycle = detect_cycle(ref, visited, path)
                        if cycle:
                            return cycle
            return []

        for virtual_model in virtual_models:
            model_config = models[virtual_model]

            for i, candidate in enumerate(model_config['candidates']):
                if 'model' in candidate:
                    ref_model = candidate['model']

                    if ref_model not in models:
                        issues.append(f"Virtual model '{virtual_model}' candidate {i} references non-existent model '{ref_model}'")

        # Check for circular references across all virtual models
        visited = set()
        for virtual_model in virtual_models:
            cycle = detect_cycle(virtual_model, visited, [])
            if cycle:
                issues.append(f"Circular reference detected: {' -> '.join(cycle)}")

        if issues:
            pytest.fail(f"Virtual model reference issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

    def test_all_provider_references_exist(self, config):
        """Test that all models reference existing providers."""
        models = config.models
        providers = config.providers
        issues = []

        provider_names = set(providers.keys())

        for model_name, model_config in models.items():
            # Skip virtual models (they don't directly reference providers)
            if 'candidates' in model_config:
                continue

            if 'provider' in model_config:
                provider = model_config['provider']
                if provider not in provider_names:
                    issues.append(f"Model '{model_name}' references non-existent provider '{provider}'")

        if issues:
            pytest.fail(f"Provider reference issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

    def test_model_keys_are_unique(self, config):
        """Test that all model keys are unique (should be guaranteed by YAML parsing)."""
        models = config.models

        # This check is largely redundant since YAML parsing would fail with duplicate keys,
        # but we verify the keys are well-formed
        model_keys = set()
        duplicates = []

        for model_name in models.keys():
            if model_name in model_keys:
                duplicates.append(model_name)
            model_keys.add(model_name)

        if duplicates:
            pytest.fail(f"Duplicate model keys found: {duplicates}")

        # Ensure we have reasonable number of models
        assert len(model_keys) > 10, f"Expected more than 10 models, found {len(model_keys)}"

    def test_cost_configurations_are_valid(self, config):
        """Test that cost configurations are properly formatted."""
        models = config.models
        issues = []

        models_with_cost = 0
        models_without_cost = []

        for model_name, model_config in models.items():
            # Skip virtual models
            if 'candidates' in model_config:
                continue

            if 'cost' in model_config:
                models_with_cost += 1
                cost_config = model_config['cost']

                # Handle runtime cost determination
                if cost_config == "runtime":
                    # Runtime cost is valid - skip detailed validation
                    continue

                # Check required cost fields for static cost configs
                required_cost_fields = ['input_cost_per_1m', 'output_cost_per_1m', 'currency']
                for field in required_cost_fields:
                    if field not in cost_config:
                        issues.append(f"Model '{model_name}' cost config missing '{field}'")

                # Check that costs are numbers
                for field in ['input_cost_per_1m', 'output_cost_per_1m']:
                    if field in cost_config and not isinstance(cost_config[field], (int, float)):
                        issues.append(f"Model '{model_name}' cost field '{field}' must be numeric")
            else:
                models_without_cost.append(model_name)

        if issues:
            pytest.fail(f"Cost configuration issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

        # Most models should have cost configuration
        assert models_with_cost > 0, "No models have cost configuration"

    def test_provider_configurations_are_complete(self, config):
        """Test that provider configurations have required fields."""
        providers = config.providers
        issues = []

        for provider_name, provider_config in providers.items():
            # Each provider should have an endpoint (single), endpoints (multiple), or auth_type (cloud providers)
            # Cloud providers with auth_type get their endpoint from the token provider
            if 'endpoint' not in provider_config and 'endpoints' not in provider_config and 'auth_type' not in provider_config:
                issues.append(f"Provider '{provider_name}' missing 'endpoint', 'endpoints', or 'auth_type' field")

            # Single endpoint should be a valid URL
            endpoint = provider_config.get('endpoint', '')
            if endpoint and not (endpoint.startswith('http://') or endpoint.startswith('https://')):
                issues.append(f"Provider '{provider_name}' endpoint '{endpoint}' should be a valid HTTP(S) URL")

            # Multiple endpoints should all be valid URLs
            endpoints = provider_config.get('endpoints', [])
            if endpoints:
                if not isinstance(endpoints, list):
                    issues.append(f"Provider '{provider_name}' endpoints should be a list")
                else:
                    for idx, ep in enumerate(endpoints):
                        if not (ep.startswith('http://') or ep.startswith('https://')):
                            issues.append(f"Provider '{provider_name}' endpoints[{idx}] '{ep}' should be a valid HTTP(S) URL")

        if issues:
            pytest.fail(f"Provider configuration issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

        # Should have reasonable number of providers
        assert len(providers) >= 5, f"Expected at least 5 providers, found {len(providers)}"

    def test_capability_flags_are_boolean(self, config):
        """Test that capability flags are properly formatted as booleans."""
        models = config.models
        issues = []

        expected_capabilities = ['supports_json_mode', 'supports_temperature', 'supports_system']

        for model_name, model_config in models.items():
            # Skip virtual models
            if 'candidates' in model_config:
                continue

            if 'capabilities' in model_config:
                capabilities = model_config['capabilities']
                if not isinstance(capabilities, dict):
                    issues.append(f"Model '{model_name}' capabilities must be a dict")
                    continue

                for cap in expected_capabilities:
                    if cap in capabilities and not isinstance(capabilities[cap], bool):
                        issues.append(f"Model '{model_name}' capability '{cap}' must be boolean, got {type(capabilities[cap])}")

        if issues:
            pytest.fail(f"Capability configuration issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

    def _validate_model_structure(self, model_name: str, model_config: Dict[str, Any], all_models: Dict[str, Any]) -> List[str]:
        """Validate that a model has proper structure."""
        issues = []

        # Check if it's a virtual model
        if 'candidates' in model_config:
            # Virtual model validation
            if not isinstance(model_config['candidates'], list):
                issues.append(f"Virtual model '{model_name}' candidates must be a list")
            elif len(model_config['candidates']) == 0:
                issues.append(f"Virtual model '{model_name}' has empty candidates list")

            # Check each candidate
            for i, candidate in enumerate(model_config['candidates']):
                if not isinstance(candidate, dict):
                    issues.append(f"Virtual model '{model_name}' candidate {i} must be a dict")
                    continue

                if 'model' not in candidate:
                    issues.append(f"Virtual model '{model_name}' candidate {i} missing 'model' field")
                    continue

                # Check if referenced model exists
                ref_model = candidate['model']
                if ref_model not in all_models:
                    issues.append(f"Virtual model '{model_name}' references non-existent model '{ref_model}'")

        else:
            # Regular model validation
            required_fields = ['provider', 'model_id']
            for field in required_fields:
                if field not in model_config:
                    issues.append(f"Model '{model_name}' missing required field '{field}'")

            # Check capabilities structure
            if 'capabilities' in model_config:
                capabilities = model_config['capabilities']
                if not isinstance(capabilities, dict):
                    issues.append(f"Model '{model_name}' capabilities must be a dict")

        return issues