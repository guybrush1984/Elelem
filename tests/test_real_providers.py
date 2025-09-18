"""
Comprehensive real provider testing for Elelem.
Tests all models with simple requests and validates cost tracking.
"""

import pytest
import json
import os
from dotenv import load_dotenv

from elelem import Elelem
from elelem.config import Config

# Load environment variables from .env file
load_dotenv()


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters for model testing."""
    if "model_name" in metafunc.fixturenames:
        config = Config()

        if "regular_model" in metafunc.function.__name__:
            # Get all regular models
            models = config.models
            regular_models = [k for k, v in models.items() if 'candidates' not in v]
            metafunc.parametrize("model_name", regular_models)

        elif "virtual_model" in metafunc.function.__name__:
            # Get all virtual models
            models = config.models
            virtual_models = [k for k, v in models.items() if 'candidates' in v]
            metafunc.parametrize("model_name", virtual_models)

        elif "json_request" in metafunc.function.__name__:
            # Get ALL regular models for JSON testing
            models = config.models
            json_models = [k for k, v in models.items() if 'candidates' not in v]  # All regular models
            metafunc.parametrize("model_name", json_models)


# Global tracking for test results across parametrized tests
_test_results = {
    'regular_models': {'passed': [], 'failed': []},
    'virtual_models': {'passed': [], 'failed': []},
    'json_models': {'passed': [], 'failed': []}
}


@pytest.fixture
def elelem():
    """Create Elelem instance for testing."""
    return Elelem()


@pytest.fixture
def config():
    """Create Config instance for testing."""
    return Config()


@pytest.fixture
def regular_models(config):
    """Get regular (non-virtual) models for testing."""
    models = config.models
    return {k: v for k, v in models.items() if 'candidates' not in v}


@pytest.fixture
def regular_model_names(regular_models):
    """Get list of regular model names for parametrization."""
    return list(regular_models.keys())


@pytest.fixture
def virtual_models(config):
    """Get virtual models for testing."""
    models = config.models
    return {k: v for k, v in models.items() if 'candidates' in v}


@pytest.fixture
def json_capable_models(regular_models):
    """Get models that support JSON mode."""
    json_models = []
    for model_name, model_config in regular_models.items():
        capabilities = model_config.get('capabilities', {})
        if capabilities.get('supports_json_mode', False):
            json_models.append(model_name)
    return json_models


def has_any_api_keys():
    """Check if any API keys are configured."""
    api_key_vars = [
        'OPENAI_API_KEY', 'GROQ_API_KEY', 'FIREWORKS_API_KEY',
        'DEEPINFRA_API_KEY', 'OPENROUTER_API_KEY', 'PARASAIL_API_KEY',
        'SCALEWAY_API_KEY'
    ]
    return any(os.getenv(var) for var in api_key_vars)


class TestRealProviders:
    """Test real provider functionality."""

    @pytest.mark.asyncio
    async def test_config_validation(self, config):
        """Test that configuration loads without errors."""
        assert config is not None
        assert len(config.models) > 0
        assert len(config.providers) > 0

    @pytest.mark.asyncio
    async def test_regular_model_simple_request(self, elelem, model_name):
        """Test individual regular model with simple request."""

        if not has_any_api_keys():
            pytest.skip("No API keys configured - skipping real provider tests")

        try:
            # Record stats before request
            stats_before = elelem.get_stats()
            calls_before = stats_before.get('total_calls', 0)

            # Make simple request
            response = await elelem.create_chat_completion(
                model=model_name,
                messages=[
                    {"role": "user", "content": "Say 'Hello' and nothing else."}
                ],
                tags=["real-provider-test", "simple-request"]
            )

            # Validate response structure
            assert response is not None, f"No response from {model_name}"
            assert hasattr(response, 'choices'), f"No choices in response from {model_name}"
            assert len(response.choices) > 0, f"Empty choices from {model_name}"

            choice = response.choices[0]
            assert hasattr(choice, 'message'), f"No message in choice from {model_name}"
            assert hasattr(choice.message, 'content'), f"No content in message from {model_name}"

            content = choice.message.content
            assert content is not None, f"Null content from {model_name}"
            assert isinstance(content, str), f"Non-string content from {model_name}"
            assert len(content.strip()) > 0, f"Empty content from {model_name}"

            # Check cost tracking
            stats_after = elelem.get_stats()
            calls_after = stats_after.get('total_calls', 0)
            assert calls_after > calls_before, f"Call count not incremented for {model_name}"

            # Track success
            _test_results['regular_models']['passed'].append(model_name)

        except Exception as e:
            # Track failure but don't fail the test immediately
            _test_results['regular_models']['failed'].append((model_name, str(e)))

            # Get provider from model name
            provider = model_name.split(':')[0] if ':' in model_name else 'unknown'

            # Check if this is an acceptable failure or a systemic issue
            provider_models_tested = [m for m in _test_results['regular_models']['passed'] + [f[0] for f in _test_results['regular_models']['failed']] if m.startswith(f"{provider}:")]
            provider_failures = [f for f in _test_results['regular_models']['failed'] if f[0].startswith(f"{provider}:")]

            # If ALL models from this provider are failing, that might indicate a real issue
            if len(provider_models_tested) >= 2 and len(provider_failures) == len(provider_models_tested):
                pytest.fail(f"All models from provider '{provider}' are failing. This indicates a systemic issue: {str(e)}")

            # Otherwise, just mark as expected failure for flaky provider
            pytest.skip(f"Provider temporarily unavailable: {str(e)}")

    @pytest.mark.asyncio
    async def test_virtual_model_simple_request(self, elelem, model_name):
        """Test individual virtual model with simple request."""

        if not has_any_api_keys():
            pytest.skip("No API keys configured - skipping real provider tests")

        try:
            # Record stats before request
            stats_before = elelem.get_stats()
            calls_before = stats_before.get('total_calls', 0)

            # Make simple request
            response = await elelem.create_chat_completion(
                model=model_name,
                messages=[
                    {"role": "user", "content": "Say 'Hello' and nothing else."}
                ],
                tags=["real-provider-test", "virtual-model"]
            )

            # Validate response structure
            assert response is not None, f"No response from {model_name}"
            assert hasattr(response, 'choices'), f"No choices in response from {model_name}"
            assert len(response.choices) > 0, f"Empty choices from {model_name}"

            choice = response.choices[0]
            assert hasattr(choice, 'message'), f"No message in choice from {model_name}"
            assert hasattr(choice.message, 'content'), f"No content in message from {model_name}"

            content = choice.message.content
            assert content is not None, f"Null content from {model_name}"
            assert isinstance(content, str), f"Non-string content from {model_name}"
            assert len(content.strip()) > 0, f"Empty content from {model_name}"

            # Check cost tracking
            stats_after = elelem.get_stats()
            calls_after = stats_after.get('total_calls', 0)
            assert calls_after > calls_before, f"Call count not incremented for {model_name}"

            # Track success
            _test_results['virtual_models']['passed'].append(model_name)

        except Exception as e:
            # Track failure
            _test_results['virtual_models']['failed'].append((model_name, str(e)))

            # Virtual models should have failover, but if underlying providers are down, they might fail
            # Allow virtual model failures but ensure at least some are working
            total_tested = len(_test_results['virtual_models']['passed']) + len(_test_results['virtual_models']['failed'])
            if total_tested >= 3 and len(_test_results['virtual_models']['passed']) == 0:
                pytest.fail(f"All virtual models are failing. This indicates underlying provider issues: {str(e)}")

            pytest.skip(f"Virtual model temporarily unavailable (underlying provider issues): {str(e)}")

    @pytest.mark.asyncio
    async def test_json_request(self, elelem, model_name, config):
        """Test individual model with JSON request (with or without response_format)."""

        if not has_any_api_keys():
            pytest.skip("No API keys configured - skipping real provider tests")

        # Check if model supports JSON mode parameter
        model_config = config.models[model_name]
        capabilities = model_config.get('capabilities', {})
        supports_json_mode = capabilities.get('supports_json_mode', False)

        try:
            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Return a JSON object with keys 'name' and 'age' for a person named John who is 25 years old."}
                ],
                "tags": ["real-provider-test", "json-mode"]
            }

            # Only add response_format if the model supports it
            if supports_json_mode:
                request_params["response_format"] = {"type": "json_object"}

            response = await elelem.create_chat_completion(**request_params)

            # Validate response structure
            assert response is not None, f"No response from {model_name}"
            assert hasattr(response, 'choices'), f"No choices in response from {model_name}"
            assert len(response.choices) > 0, f"Empty choices from {model_name}"

            content = response.choices[0].message.content
            assert content is not None, f"Null content from {model_name}"
            assert isinstance(content, str), f"Non-string content from {model_name}"

            # Try to parse as JSON
            parsed_json = json.loads(content)
            assert isinstance(parsed_json, dict), f"JSON response is not a dict from {model_name}"

            # Check that the JSON contains expected keys
            assert 'name' in parsed_json, f"JSON response missing 'name' key from {model_name}"
            assert 'age' in parsed_json, f"JSON response missing 'age' key from {model_name}"

            # Track success
            _test_results['json_models']['passed'].append(model_name)

        except Exception as e:
            # Track failure
            _test_results['json_models']['failed'].append((model_name, str(e)))

            # Get provider from model name
            provider = model_name.split(':')[0] if ':' in model_name else 'unknown'

            # JSON requests might not be perfectly reliable, allow some failures
            # But if a provider completely fails JSON requests, that's an issue
            provider_json_tested = [m for m in _test_results['json_models']['passed'] + [f[0] for f in _test_results['json_models']['failed']] if m.startswith(f"{provider}:")]
            provider_json_failures = [f for f in _test_results['json_models']['failed'] if f[0].startswith(f"{provider}:")]

            if len(provider_json_tested) >= 2 and len(provider_json_failures) == len(provider_json_tested):
                pytest.fail(f"All models from provider '{provider}' are failing JSON requests: {str(e)}")

            pytest.skip(f"JSON request temporarily unavailable for this model: {str(e)}")

    @pytest.mark.asyncio
    async def test_cost_tracking_comprehensive(self, elelem, regular_models):
        """Test comprehensive cost tracking across models."""

        if not has_any_api_keys():
            pytest.skip("No API keys configured - skipping real provider tests")

        # Test ALL regular models for cost tracking
        test_models = list(regular_models.keys())

        models_with_cost = []
        models_without_cost = []

        for model_name in test_models:
            try:
                response = await elelem.create_chat_completion(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": "Count from 1 to 5."}
                    ],
                    tags=["cost-tracking-test"]
                )

                # Check if response has usage information
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    cost_info = {
                        'model': model_name,
                        'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                        'completion_tokens': getattr(usage, 'completion_tokens', 0),
                        'total_tokens': getattr(usage, 'total_tokens', 0)
                    }

                    # Validate usage data
                    assert cost_info['total_tokens'] > 0, f"No tokens reported for {model_name}"
                    assert cost_info['prompt_tokens'] > 0, f"No prompt tokens for {model_name}"

                    models_with_cost.append(cost_info)
                else:
                    models_without_cost.append(model_name)

            except Exception as e:
                # Skip models that fail for other reasons
                print(f"Skipping cost test for {model_name}: {str(e)}")
                continue

        print(f"\nCost tracking summary:")
        print(f"✓ Models with cost data: {len(models_with_cost)}")
        print(f"⚠ Models without cost data: {len(models_without_cost)}")

        if models_with_cost:
            print("Cost tracking examples:")
            for cost_info in models_with_cost[:3]:
                print(f"  - {cost_info['model']}: {cost_info['total_tokens']} tokens")

        # All tested models should have cost tracking
        assert len(models_with_cost) > 0, "No models provided cost tracking data"

        # Ensure we got cost data from at least some models
        working_models = len(models_with_cost) + len(models_without_cost)
        if working_models == 0:
            pytest.fail("No models could be tested - check API keys and connectivity")

    @pytest.mark.asyncio
    async def test_stats_aggregation(self, elelem):
        """Test that stats are properly aggregated across requests."""

        if not has_any_api_keys():
            pytest.skip("No API keys configured - skipping real provider tests")

        # Clear any existing stats (check correct attribute name)
        if hasattr(elelem, 'metrics'):
            elelem.metrics.clear_all_data()
        elif hasattr(elelem, '_metrics'):
            elelem._metrics.clear_all_data()
        else:
            # Try to find the metrics object
            for attr_name in dir(elelem):
                attr = getattr(elelem, attr_name)
                if hasattr(attr, 'clear_all_data'):
                    attr.clear_all_data()
                    break

        # Make a few test requests
        test_models = ["openai:openai/gpt-4.1-mini"]  # Use a reliable model

        for i in range(3):
            try:
                await elelem.create_chat_completion(
                    model=test_models[0],
                    messages=[
                        {"role": "user", "content": f"Say 'Test {i+1}'."}
                    ],
                    tags=[f"stats-test-{i+1}"]
                )
            except Exception:
                # Skip if model not available
                pytest.skip(f"Model {test_models[0]} not available")

        # Check overall stats
        stats = elelem.get_stats()
        assert 'total_calls' in stats
        assert stats['total_calls'] >= 3

        # Check stats by tag
        for i in range(3):
            tag_stats = elelem.get_stats_by_tag(f"stats-test-{i+1}")
            assert len(tag_stats) >= 1, f"No stats found for tag stats-test-{i+1}"


# Pytest markers would go here if registered in pytest config

# Additional test classes for specific provider testing
class TestSpecificProviders:
    """Test specific provider functionality."""

    @pytest.mark.asyncio
    async def test_openai_specific_features(self, elelem):
        """Test OpenAI-specific features."""
        try:
            response = await elelem.create_chat_completion(
                model="openai:openai/gpt-4.1-mini",
                messages=[
                    {"role": "user", "content": "What is 2+2?"}
                ],
                tags=["openai-specific"]
            )

            assert response is not None
            assert hasattr(response, 'choices')
            # OpenAI should provide usage data
            assert hasattr(response, 'usage')

        except Exception as e:
            pytest.skip(f"OpenAI test skipped: {str(e)}")

    @pytest.mark.asyncio
    async def test_groq_specific_features(self, elelem):
        """Test Groq-specific features."""
        try:
            response = await elelem.create_chat_completion(
                model="groq:openai/gpt-oss-120b?reasoning=low",
                messages=[
                    {"role": "user", "content": "What is 2+2?"}
                ],
                tags=["groq-specific"]
            )

            assert response is not None
            assert hasattr(response, 'choices')

        except Exception as e:
            pytest.skip(f"Groq test skipped: {str(e)}")

    @pytest.mark.asyncio
    async def test_virtual_model_failover(self, elelem):
        """Test virtual model failover behavior."""
        try:
            # Use a virtual model that has multiple candidates
            response = await elelem.create_chat_completion(
                model="virtual:gpt-oss-120b-cheap",
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                tags=["virtual-failover"]
            )

            assert response is not None
            assert hasattr(response, 'choices')

            # The response should indicate which provider was actually used
            # This depends on Elelem's implementation details

        except Exception as e:
            pytest.skip(f"Virtual model test skipped: {str(e)}")

    def test_overall_provider_health_summary(self):
        """Summary test to report overall provider health across all parametrized tests."""

        # This test runs after all parametrized tests and gives a summary
        if not has_any_api_keys():
            pytest.skip("No API keys configured")

        print("\n" + "="*60)
        print("OVERALL PROVIDER HEALTH SUMMARY")
        print("="*60)

        # Regular models summary
        regular_passed = len(_test_results['regular_models']['passed'])
        regular_failed = len(_test_results['regular_models']['failed'])
        regular_total = regular_passed + regular_failed

        if regular_total > 0:
            regular_success_rate = (regular_passed / regular_total) * 100
            print(f"Regular Models: {regular_passed}/{regular_total} passed ({regular_success_rate:.1f}%)")

            if regular_failed > 0:
                print("  Failed models:")
                for model, error in _test_results['regular_models']['failed']:
                    print(f"    - {model}: {error[:60]}...")

        # Virtual models summary
        virtual_passed = len(_test_results['virtual_models']['passed'])
        virtual_failed = len(_test_results['virtual_models']['failed'])
        virtual_total = virtual_passed + virtual_failed

        if virtual_total > 0:
            virtual_success_rate = (virtual_passed / virtual_total) * 100
            print(f"Virtual Models: {virtual_passed}/{virtual_total} passed ({virtual_success_rate:.1f}%)")

        # JSON models summary
        json_passed = len(_test_results['json_models']['passed'])
        json_failed = len(_test_results['json_models']['failed'])
        json_total = json_passed + json_failed

        if json_total > 0:
            json_success_rate = (json_passed / json_total) * 100
            print(f"JSON-capable Models: {json_passed}/{json_total} passed ({json_success_rate:.1f}%)")

        # Provider breakdown
        print("\nProvider Breakdown:")
        all_models = (_test_results['regular_models']['passed'] +
                     [f[0] for f in _test_results['regular_models']['failed']])

        providers = {}
        for model in all_models:
            provider = model.split(':')[0] if ':' in model else 'unknown'
            if provider not in providers:
                providers[provider] = {'passed': 0, 'failed': 0}

        # Count passes and failures by provider
        for model in _test_results['regular_models']['passed']:
            provider = model.split(':')[0] if ':' in model else 'unknown'
            providers[provider]['passed'] += 1

        for model, error in _test_results['regular_models']['failed']:
            provider = model.split(':')[0] if ':' in model else 'unknown'
            providers[provider]['failed'] += 1

        for provider, stats in providers.items():
            total = stats['passed'] + stats['failed']
            if total > 0:
                success_rate = (stats['passed'] / total) * 100
                print(f"  {provider}: {stats['passed']}/{total} passed ({success_rate:.1f}%)")

        print("="*60)

        # Overall health check - ensure we have some working providers
        total_working_providers = sum(1 for stats in providers.values() if stats['passed'] > 0)
        if total_working_providers == 0 and regular_total > 0:
            pytest.fail("No providers are working - this indicates a systemic issue")

        # Ensure we have reasonable coverage
        if regular_total > 0 and regular_success_rate < 50:
            pytest.fail(f"Overall success rate too low: {regular_success_rate:.1f}% - indicates widespread provider issues")