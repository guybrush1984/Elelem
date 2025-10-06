"""Test the parameterized model expansion system."""

import pytest
from unittest.mock import patch
from elelem.config import expand_parameterized_models


@pytest.mark.unit
class TestParameterizedModels:
    """Test parameterized model expansion functionality."""

    def test_basic_parameter_expansion(self):
        """Test basic parameter expansion with single parameter."""
        models = {
            "test:model?param=@[a,b,c]": {
                "provider": "test",
                "model_id": "model",
                "default_params": {
                    "parameter": "$param"
                },
                "metadata": {
                    "config": "param=$param"
                }
            }
        }

        expanded = expand_parameterized_models(models)

        # Should expand to 3 models
        assert len(expanded) == 3

        expected_models = {
            "test:model?param=a",
            "test:model?param=b",
            "test:model?param=c"
        }
        assert set(expanded.keys()) == expected_models

        # Check parameter substitution
        assert expanded["test:model?param=a"]["default_params"]["parameter"] == "a"
        assert expanded["test:model?param=b"]["default_params"]["parameter"] == "b"
        assert expanded["test:model?param=c"]["default_params"]["parameter"] == "c"

        # Check metadata substitution
        assert expanded["test:model?param=a"]["metadata"]["config"] == "param=a"

    def test_reasoning_parameter_expansion(self):
        """Test reasoning parameter expansion (real-world scenario)."""
        models = {
            "faker:reasoning-test?reasoning=@[low,medium,high]": {
                "provider": "faker",
                "model_id": "reasoning-test",
                "capabilities": {
                    "supports_json_mode": True,
                    "supports_temperature": True,
                    "supports_system": True
                },
                "cost": {
                    "input_cost_per_1m": 0.15,
                    "output_cost_per_1m": 0.75,
                    "currency": "USD"
                },
                "default_params": {
                    "reasoning_effort": "$reasoning"
                },
                "metadata": {
                    "model_reference": "reasoning_test_model",
                    "model_configuration": "reasoning=$reasoning"
                }
            }
        }

        expanded = expand_parameterized_models(models)

        # Should expand to 3 reasoning models
        assert len(expanded) == 3

        expected_models = {
            "faker:reasoning-test?reasoning=low",
            "faker:reasoning-test?reasoning=medium",
            "faker:reasoning-test?reasoning=high"
        }
        assert set(expanded.keys()) == expected_models

        # Verify reasoning_effort is correctly substituted
        assert expanded["faker:reasoning-test?reasoning=low"]["default_params"]["reasoning_effort"] == "low"
        assert expanded["faker:reasoning-test?reasoning=medium"]["default_params"]["reasoning_effort"] == "medium"
        assert expanded["faker:reasoning-test?reasoning=high"]["default_params"]["reasoning_effort"] == "high"

        # Verify other fields remain unchanged
        for model in expanded.values():
            assert model["provider"] == "faker"
            assert model["model_id"] == "reasoning-test"
            assert model["capabilities"]["supports_json_mode"] is True
            assert model["cost"]["input_cost_per_1m"] == 0.15

    def test_no_parameterization(self):
        """Test that non-parameterized models remain unchanged."""
        models = {
            "faker:basic": {
                "provider": "faker",
                "model_id": "basic"
            },
            "faker:json-temp-test": {
                "provider": "faker",
                "model_id": "json-temp-test"
            }
        }

        expanded = expand_parameterized_models(models)

        # Should remain the same
        assert expanded == models
        assert len(expanded) == 2

    def test_mixed_parameterized_and_normal(self):
        """Test mix of parameterized and normal models."""
        models = {
            "faker:param?level=@[a,b]": {
                "provider": "faker",
                "default_params": {
                    "level": "$level"
                }
            },
            "faker:normal": {
                "provider": "faker",
                "model_id": "normal"
            }
        }

        expanded = expand_parameterized_models(models)

        # Should have 3 models total (2 expanded + 1 normal)
        assert len(expanded) == 3

        expected_models = {
            "faker:param?level=a",
            "faker:param?level=b",
            "faker:normal"
        }
        assert set(expanded.keys()) == expected_models

        # Normal model unchanged
        assert expanded["faker:normal"]["model_id"] == "normal"

        # Parameterized models expanded correctly
        assert expanded["faker:param?level=a"]["default_params"]["level"] == "a"
        assert expanded["faker:param?level=b"]["default_params"]["level"] == "b"

    def test_deep_parameter_substitution(self):
        """Test parameter substitution in nested structures."""
        models = {
            "test:model?mode=@[fast,slow]": {
                "provider": "test",
                "default_params": {
                    "speed": "$mode",
                    "nested": {
                        "config": "mode=$mode",
                        "values": ["$mode", "static"]
                    }
                },
                "metadata": {
                    "description": "Model with $mode processing"
                }
            }
        }

        expanded = expand_parameterized_models(models)

        fast_model = expanded["test:model?mode=fast"]
        slow_model = expanded["test:model?mode=slow"]

        # Check nested substitution
        assert fast_model["default_params"]["nested"]["config"] == "mode=fast"
        assert fast_model["default_params"]["nested"]["values"] == ["fast", "static"]
        assert fast_model["metadata"]["description"] == "Model with fast processing"

        assert slow_model["default_params"]["nested"]["config"] == "mode=slow"
        assert slow_model["default_params"]["nested"]["values"] == ["slow", "static"]
        assert slow_model["metadata"]["description"] == "Model with slow processing"

    def test_parameterized_models_pass_validation(self):
        """Test that parameterized models generate valid configurations that pass normal validation."""
        # Create a realistic parameterized model that should pass all validation checks
        models = {
            "test-provider:gpt-test?reasoning=@[low,high]": {
                "provider": "test-provider",
                "model_id": "gpt-test",
                "capabilities": {
                    "supports_json_mode": True,
                    "supports_temperature": True,
                    "supports_system": True
                },
                "cost": {
                    "input_cost_per_1m": 0.15,
                    "output_cost_per_1m": 0.75,
                    "currency": "USD"
                },
                "default_params": {
                    "reasoning_effort": "$reasoning"
                },
                "metadata": {
                    "model_reference": "gpt_test",
                    "model_configuration": "reasoning=$reasoning"
                }
            }
        }

        expanded = expand_parameterized_models(models)

        # Verify both expanded models have all required fields and correct types
        for model_key, model_config in expanded.items():
            # Check required top-level fields
            assert "provider" in model_config
            assert "model_id" in model_config
            assert "capabilities" in model_config
            assert "cost" in model_config

            # Check capabilities are boolean
            capabilities = model_config["capabilities"]
            assert isinstance(capabilities["supports_json_mode"], bool)
            assert isinstance(capabilities["supports_temperature"], bool)
            assert isinstance(capabilities["supports_system"], bool)

            # Check cost structure
            cost = model_config["cost"]
            assert "input_cost_per_1m" in cost
            assert "output_cost_per_1m" in cost
            assert "currency" in cost
            assert isinstance(cost["input_cost_per_1m"], (int, float))
            assert isinstance(cost["output_cost_per_1m"], (int, float))
            assert cost["currency"] == "USD"

            # Check parameter substitution worked
            assert "default_params" in model_config
            reasoning_effort = model_config["default_params"]["reasoning_effort"]
            assert reasoning_effort in ["low", "high"]

            # Check model key format is correct
            assert model_key in ["test-provider:gpt-test?reasoning=low", "test-provider:gpt-test?reasoning=high"]

        print("✅ Parameterized models generate valid configurations that pass validation checks")

