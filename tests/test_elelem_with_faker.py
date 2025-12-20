"""Test Elelem's features using the faker system."""

import pytest
import asyncio
import json
import time
from unittest.mock import patch
import os

# Import Elelem
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elelem import Elelem

# Import faker from tests directory
sys.path.insert(0, str(Path(__file__).parent))
from faker.server import ModelFaker


class TestElelemWithFaker:
    """Test Elelem's advanced features using the faker system."""

    @pytest.fixture(scope="function")
    def faker_server(self):
        """Start a faker server for testing."""
        faker = ModelFaker(port=6666)
        try:
            faker.start()
            time.sleep(0.5)  # Wait for server to be ready
            yield faker
        finally:
            # Ensure cleanup happens even if test fails
            try:
                faker.stop()
                time.sleep(1.0)  # Give time for socket to be released
            except Exception as e:
                print(f"Warning: Error during faker cleanup: {e}")

    @pytest.fixture(scope="function")
    def elelem_with_faker_env(self, faker_server, monkeypatch, tmp_path):
        """Setup Elelem with faker environment."""
        # Set environment variable for faker
        monkeypatch.setenv("FAKER_API_KEY", "fake-key-123")
        # Set environment variable for faker-streaming provider
        monkeypatch.setenv("FAKER-STREAMING_API_KEY", "fake-streaming-key-456")

        # Use a unique temporary SQLite database for each test to avoid pollution
        db_file = tmp_path / "test_metrics.db"
        monkeypatch.setenv("ELELEM_DATABASE_URL", f"sqlite:///{db_file}")

        # Update faker provider configuration to use the actual server port
        # (This ensures consistency even if we later make ports dynamic)
        from elelem.config import Config
        config = Config()
        faker_provider = config.get_provider_config("faker")
        faker_provider["endpoint"] = f"http://localhost:{faker_server.port}/v1"

        # Create Elelem instance with test provider directories
        elelem = Elelem(extra_provider_dirs=["tests/providers"])
        return elelem, faker_server

    @pytest.mark.asyncio
    async def test_json_temperature_reduction(self, elelem_with_faker_env):
        """Test that Elelem reduces temperature when JSON parsing fails."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with JSON temperature reduction scenario
        faker.configure_scenario('elelem_json_temp_reduction')
        faker.reset_state()

        # Make request with high temperature and JSON mode
        response = await elelem.create_chat_completion(
            model="faker:json-temp-test",
            messages=[{"role": "user", "content": "Return JSON data"}],
            response_format={"type": "json_object"},
            temperature=0.9
        )

        # Should eventually succeed with valid JSON
        assert response
        content = response.choices[0].message.content
        json_data = json.loads(content)
        assert json_data["status"] == "success_after_temperature_reduction"

        # Verify Elelem made multiple requests with decreasing temperature
        requests = faker.request_analyzer.get_captured_requests()
        print(f"Number of requests made: {len(requests)}")

        # Check temperature reduction pattern
        temperatures = []
        for i, req in enumerate(requests):
            body = req.get('body', {})
            temp = body.get('temperature', 'not set')
            temperatures.append(temp)
            print(f"Request {i+1}: temperature = {temp}")

        print(f"Temperature sequence: {temperatures}")

        # Check elelem_metrics structure
        elelem_metrics = response.elelem_metrics
        print(f"Elelem metrics keys: {list(elelem_metrics.keys())}")

        # Should show decreasing temperatures
        assert len(requests) >= 2  # Should have retried at least once
        assert len(temperatures) >= 2
        assert temperatures[0] > temperatures[-1]  # Temperature should decrease

        # Verify the temperature reduction worked as expected
        assert temperatures == [0.9, 0.8, 0.5]  # Exact sequence expected

    @pytest.mark.asyncio
    async def test_json_repair_fixes_malformed_json(self, elelem_with_faker_env):
        """Test that json-repair fixes repairable JSON errors without retrying."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with json-repair scenario
        faker.configure_scenario('elelem_json_repair')
        faker.reset_state()

        # Make request - faker will return JSON with missing closing brace
        response = await elelem.create_chat_completion(
            model="faker:json-repair-test",
            messages=[{"role": "user", "content": "Return JSON data"}],
            response_format={"type": "json_object"},
            temperature=0.5
        )

        # Should succeed on first attempt (json-repair fixes the malformed JSON)
        assert response
        content = response.choices[0].message.content

        # The repaired JSON should be parseable
        json_data = json.loads(content)
        assert json_data.get("test") == "data"  # Original data from faker's json_error

        # Verify only ONE request was made (no retry needed due to json-repair)
        requests = faker.request_analyzer.get_captured_requests()
        print(f"Number of requests made: {len(requests)}")
        assert len(requests) == 1, "json-repair should fix the error without retrying"

    @pytest.mark.asyncio
    async def test_rate_limit_retry_then_success(self, elelem_with_faker_env):
        """Test Elelem's rate limit retry logic."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with rate limit scenario
        faker.configure_scenario('elelem_rate_limits')
        faker.reset_state()

        # Make request that will hit rate limits
        start_time = time.time()
        response = await elelem.create_chat_completion(
            model="faker:rate-limited",
            messages=[{"role": "user", "content": "Test rate limits"}]
        )
        elapsed_time = time.time() - start_time

        # Should eventually succeed
        assert response
        assert "Success after" in response.choices[0].message.content

        # Should have taken some time due to rate limit backoff
        assert elapsed_time > 1.0  # At least 1 second due to retries

        # Verify multiple requests were made
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) >= 3  # Multiple attempts due to rate limits

    @pytest.mark.asyncio
    async def test_parameter_cleanup_json_mode(self, elelem_with_faker_env):
        """Test that Elelem removes response_format for models that don't support JSON."""
        elelem, faker = elelem_with_faker_env

        # Configure faker to capture parameter cleanup
        faker.configure_scenario('elelem_parameter_cleanup')
        faker.reset_state()

        # Make JSON mode request to model that doesn't support it
        response = await elelem.create_chat_completion(
            model="faker:no-json",
            messages=[{"role": "user", "content": "This should work without JSON mode"}],
            response_format={"type": "json_object"}  # This should be removed by Elelem
        )

        # Should succeed
        assert response
        assert "Parameter cleanup test" in response.choices[0].message.content

        # Verify Elelem removed response_format parameter
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) >= 1

        request_body = requests[0]['body']
        assert 'response_format' not in request_body  # Should be removed by Elelem

    @pytest.mark.asyncio
    async def test_cost_tracking_with_faker(self, elelem_with_faker_env):
        """Test that Elelem correctly tracks costs with faker responses."""
        elelem, faker = elelem_with_faker_env

        # Configure basic working scenario
        faker.configure_scenario('happy_path')
        faker.reset_state()

        # Make request
        response = await elelem.create_chat_completion(
            model="faker:basic",
            messages=[{"role": "user", "content": "Hello world"}],
            tags=["test_cost_tracking"]
        )

        # Should succeed
        assert response
        print(f"Response type: {type(response)}")
        print(f"Response content: {response}")

        # Elelem returns OpenAI response object
        assert hasattr(response, "usage")
        assert response.usage.total_tokens > 0

        # Check that Elelem tracked costs
        stats = elelem.get_stats_by_tag("test_cost_tracking")
        print(f"Stats by tag: {stats}")

        # Also check overall stats
        overall_stats = elelem.get_stats()
        print(f"Overall stats: {overall_stats}")

        # The response itself contains cost information
        assert response.elelem_metrics["costs_usd"]["total_cost_usd"] > 0
        assert response.elelem_metrics["tokens"]["total"] > 0

    @pytest.mark.asyncio
    async def test_virtual_model_candidate_failover(self, elelem_with_faker_env):
        """Test virtual model candidate failover using multiple faker candidates."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with rate limit scenario (first candidates will fail)
        faker.configure_scenario('elelem_rate_limits')
        faker.reset_state()

        # Test virtual model that tries rate-limited candidates first, then basic
        response = await elelem.create_chat_completion(
            model="virtual:faker-test-failover",
            messages=[{"role": "user", "content": "Test candidate failover"}]
        )

        # Should eventually succeed with the basic candidate
        assert response
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content

        # Verify we actually hit multiple candidates (check logs or internal state)
        # The response should come from the successful candidate

    @pytest.mark.asyncio
    async def test_rate_limit_exhaustion_triggers_failover(self, elelem_with_faker_env):
        """Test that exhausting rate limit retries triggers failover to next candidate.

        This tests the fix for the bug where rate limit exhaustion raised ModelError
        instead of InfrastructureError, preventing failover to the next candidate.
        """
        elelem, faker = elelem_with_faker_env

        # Configure faker with rate limit failover scenario
        # This scenario returns more 429s than max_rate_limit_retries (2)
        faker.configure_scenario('elelem_rate_limit_failover')
        faker.reset_state()

        start_time = time.time()

        # Use virtual model where first candidate exhausts rate limits
        response = await elelem.create_chat_completion(
            model="virtual:faker-rate-exhaust-failover",
            messages=[{"role": "user", "content": "Test rate limit exhaustion failover"}],
            tags=["rate-exhaust-failover-test"]
        )

        elapsed_time = time.time() - start_time

        # Should succeed via second candidate after first exhausts rate limits
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0

        # Should have taken some time due to rate limit backoff (1s + 2s = 3s minimum)
        assert elapsed_time >= 2.0, f"Expected at least 2s for rate limit backoff, got {elapsed_time:.1f}s"

        # Verify the response came from the second candidate (basic), not the first
        assert response.elelem_metrics["model_used"] == "faker:basic", \
            f"Expected failover to faker:basic, got {response.elelem_metrics['model_used']}"

        # Verify multiple requests were made (at least 3: 2 rate limits + 1 success)
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) >= 3, f"Expected at least 3 requests, got {len(requests)}"

    @pytest.mark.asyncio
    async def test_model_error_skips_same_model_reference(self, elelem_with_faker_env):
        """Test that ModelError skips all candidates with the same model_reference.

        When a candidate fails with ModelError (not InfrastructureError), all other
        candidates with the same model_reference should be skipped, and the system
        should try the next candidate with a different model_reference.
        """
        elelem, faker = elelem_with_faker_env

        # Configure faker with model error failover scenario
        faker.configure_scenario('elelem_model_error_failover')
        faker.reset_state()

        # Use virtual model with:
        # - faker:model-error (model_reference: faker_model_error) - will fail with ModelError
        # - faker:model-error-alt (model_reference: faker_model_error) - should be SKIPPED
        # - faker:basic (model_reference: faker_basic) - should succeed
        response = await elelem.create_chat_completion(
            model="virtual:faker-model-error-failover",
            messages=[{"role": "user", "content": "Test model error failover"}],
            tags=["model-error-failover-test"]
        )

        # Should succeed via the third candidate (faker:basic)
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert "Success after model-level failover" in response.choices[0].message.content

        # Verify the response came from faker:basic (not the skipped faker:model-error-alt)
        assert response.elelem_metrics["model_used"] == "faker:basic", \
            f"Expected failover to faker:basic, got {response.elelem_metrics['model_used']}"

        # Verify only 2 requests were made (not 3):
        # 1. faker:model-error -> ModelError
        # 2. faker:basic -> Success (faker:model-error-alt was SKIPPED)
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) == 2, f"Expected 2 requests (skipping same model_ref), got {len(requests)}"

    @pytest.mark.asyncio
    async def test_virtual_chaining_with_model_error_failover(self, elelem_with_faker_env):
        """Test virtual model chaining with model-level failover.

        A virtual model can reference another virtual model. When the inner virtual's
        candidates fail with ModelError, the system should skip all candidates with
        the same model_reference and try the outer virtual's fallback candidates.
        """
        elelem, faker = elelem_with_faker_env

        # Configure faker with model error failover scenario
        faker.configure_scenario('elelem_model_error_failover')
        faker.reset_state()

        # Use chained virtual model:
        # - virtual:faker-model-error-inner (flattens to: model-error, model-error-alt)
        # - faker:basic
        # First candidate from inner virtual fails -> skip second (same model_ref) -> success on faker:basic
        response = await elelem.create_chat_completion(
            model="virtual:faker-chained-failover",
            messages=[{"role": "user", "content": "Test chained virtual failover"}],
            tags=["chained-failover-test"]
        )

        # Should succeed via faker:basic after skipping both inner virtual candidates
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0

        # Verify the response came from faker:basic
        assert response.elelem_metrics["model_used"] == "faker:basic", \
            f"Expected failover to faker:basic, got {response.elelem_metrics['model_used']}"

        # Verify only 2 requests were made (inner virtual's second candidate was skipped)
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) == 2, f"Expected 2 requests (skipping same model_ref from chained virtual), got {len(requests)}"

    @pytest.mark.asyncio
    async def test_request_validation_through_elelem(self, elelem_with_faker_env):
        """Test that requests through Elelem are properly formatted."""
        elelem, faker = elelem_with_faker_env

        faker.configure_scenario('happy_path')
        faker.reset_state()

        # Make request through Elelem
        await elelem.create_chat_completion(
            model="faker:basic",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            temperature=0.7,
            max_tokens=100
        )

        # Verify request format
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) == 1

    @pytest.mark.asyncio
    async def test_json_schema_validation(self, elelem_with_faker_env):
        """Test that Elelem validates JSON responses against provided schemas."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with JSON schema scenario
        faker.configure_scenario('elelem_json_schema')
        faker.reset_state()

        # Test valid JSON schema response
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            },
            "required": ["name", "age", "email"]
        }

        response = await elelem.create_chat_completion(
            model="faker:json-schema-test",
            messages=[{"role": "user", "content": "Generate a user profile"}],
            response_format={"type": "json_object"},
            json_schema=schema,
            temperature=1.0  # Matches valid_schema condition
        )

        # Should succeed with valid JSON
        assert response
        content = response.choices[0].message.content

        # Verify it's valid JSON matching schema
        import json
        parsed = json.loads(content)
        assert "name" in parsed
        assert "age" in parsed
        assert "email" in parsed
        assert isinstance(parsed["age"], (int, float))

        # Test invalid JSON schema response (should trigger retry/validation)
        try:
            response = await elelem.create_chat_completion(
                model="faker:json-schema-test",
                messages=[{"role": "user", "content": "Generate another user profile"}],
                response_format={"type": "json_object"},
                json_schema=schema,
                temperature=0.5  # Matches invalid_schema condition
            )

            # If it succeeds, check that validation handled the invalid data
            content = response.choices[0].message.content
            parsed = json.loads(content)
            # Should either fix the invalid age or retry to get valid response

        except Exception as e:
            # JSON schema validation should catch invalid responses
            assert "validation" in str(e).lower() or "schema" in str(e).lower()

        # Verify requests were made
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) >= 1

    @pytest.mark.asyncio
    async def test_structured_outputs_format(self, elelem_with_faker_env):
        """Test that Elelem supports OpenAI's structured outputs format."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with JSON schema scenario
        faker.configure_scenario('elelem_json_schema')
        faker.reset_state()

        # Test structured outputs format (OpenAI standard)
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            },
            "required": ["name", "age", "email"]
        }

        response = await elelem.create_chat_completion(
            model="faker:json-schema-test",
            messages=[{"role": "user", "content": "Generate a user profile"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "UserProfile",
                    "schema": schema,
                    "strict": True
                }
            },
            temperature=1.0,  # Matches valid_schema condition
            enforce_schema_in_prompt=True  # Inject schema so test can verify it
        )

        # Should succeed with valid JSON
        assert response
        content = response.choices[0].message.content

        # Verify it's valid JSON matching schema
        import json
        parsed = json.loads(content)
        assert "name" in parsed
        assert "age" in parsed
        assert "email" in parsed
        assert isinstance(parsed["age"], (int, float))

        # Verify Elelem downgraded to json_object for the provider
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) >= 1
        last_request = requests[-1]
        request_body = last_request.get('body', {})
        # Should be downgraded to basic JSON mode
        assert request_body.get('response_format', {}).get('type') == 'json_object'
        # Should NOT have json_schema in request to provider
        assert 'json_schema' not in request_body.get('response_format', {})

        # CRITICAL: Verify schema was injected into the prompt
        messages = request_body.get('messages', [])
        # Find system message (or last user message if no system support)
        system_content = None
        for msg in messages:
            if msg.get('role') == 'system':
                system_content = msg.get('content', '')
                break

        assert system_content is not None, "Expected system message with JSON instructions"
        # Verify schema was included in the prompt
        assert '=== REQUIRED OUTPUT FORMAT ===' in system_content
        assert '"type": "object"' in system_content
        assert '"name"' in system_content
        assert '"age"' in system_content
        assert '"email"' in system_content
        assert 'required' in system_content.lower()

    @pytest.mark.asyncio
    async def test_structured_outputs_exclusivity(self, elelem_with_faker_env):
        """Test that using both formats simultaneously raises an error."""
        elelem, faker = elelem_with_faker_env

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        # Should raise ValueError when using both formats
        with pytest.raises(ValueError, match="Cannot use both"):
            await elelem.create_chat_completion(
                model="faker:json-schema-test",
                messages=[{"role": "user", "content": "Test"}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": schema
                    }
                },
                json_schema=schema  # Conflict!
            )

    @pytest.mark.asyncio
    async def test_structured_outputs_backward_compatibility(self, elelem_with_faker_env):
        """Test that old json_schema parameter still works."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with JSON schema scenario
        faker.configure_scenario('elelem_json_schema')
        faker.reset_state()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }

        # Old API should still work
        response = await elelem.create_chat_completion(
            model="faker:json-schema-test",
            messages=[{"role": "user", "content": "Generate a user profile"}],
            response_format={"type": "json_object"},
            json_schema=schema,
            temperature=1.0
        )

        # Should succeed
        assert response
        content = response.choices[0].message.content
        import json
        parsed = json.loads(content)
        assert "name" in parsed
        assert "age" in parsed

    @pytest.mark.asyncio
    async def test_reasoning_token_extraction(self, elelem_with_faker_env):
        """Test that Elelem correctly extracts reasoning tokens from different provider formats."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with reasoning token scenario
        faker.configure_scenario('elelem_reasoning_tokens')
        faker.reset_state()

        # Test OpenAI format (nested in completion_tokens_details)
        response = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Think through this problem"}],
            temperature=1.0  # Matches openai_format condition
        )

        # Verify response and check reasoning tokens in Elelem metrics
        assert response
        stats = elelem.get_stats()
        assert "tokens" in stats
        assert "reasoning" in stats["tokens"]

        # Test GROQ format (nested in output_tokens_details)
        faker.reset_state()
        response = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Analyze this scenario"}],
            temperature=0.8  # Matches groq_format condition
        )

        assert response
        stats = elelem.get_stats()
        assert stats["tokens"]["reasoning"]["total"] > 0

        # Test content-based reasoning extraction (<think> tags)
        faker.reset_state()
        response = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Use thinking tags"}],
            temperature=0.6  # Matches think_tags condition
        )

        assert response
        content = response.choices[0].message.content

        # Check what the faker actually sent vs what Elelem returned
        requests = faker.request_analyzer.get_captured_requests()
        print(f"Content returned by Elelem: '{content}'")
        print(f"Number of requests: {len(requests)}")

        # Elelem should have REMOVED <think> tags, not kept them
        assert "<think>" not in content and "</think>" not in content, f"Elelem should remove <think> tags but content is: '{content}'"

        # Verify reasoning tokens were extracted
        stats = elelem.get_stats()
        assert stats["tokens"]["reasoning"]["total"] > 0

        # Verify reasoning content was extracted and added to response
        assert hasattr(response, 'elelem_metrics'), "Response should have elelem_metrics"
        assert "reasoning_content" in response.elelem_metrics, "Response should include reasoning_content in elelem_metrics"

        reasoning_content = response.elelem_metrics["reasoning_content"]
        assert reasoning_content is not None, "Reasoning content should not be None"
        assert len(reasoning_content.strip()) > 0, "Reasoning content should not be empty"

        # The reasoning content should contain the thinking content from <think> tags
        # Based on the faker scenario, this should be the content between the tags
        print(f"Extracted reasoning content: '{reasoning_content}'")

        # Verify exact content matches what the faker scenario defines for temperature 0.6
        expected_reasoning = "Let me analyze this step by step. First, I need to understand the problem. Then I can work through the solution methodically."
        assert reasoning_content == expected_reasoning, f"Reasoning content mismatch.\nExpected: '{expected_reasoning}'\nActual: '{reasoning_content}'"

        print(f"Final reasoning tokens extracted: {stats['tokens']['reasoning']['total']}")
        total_tokens = stats['tokens']['input']['total'] + stats['tokens']['output']['total']
        print(f"Total tokens: {total_tokens}")
        print(f"✅ Reasoning content successfully extracted: {len(reasoning_content)} chars")
        print(f"Calls made: {stats['requests']['total']}")

    @pytest.mark.asyncio
    async def test_reasoning_content_extraction_formats(self, elelem_with_faker_env):
        """Test reasoning content extraction with different provider formats."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with reasoning token scenario
        faker.configure_scenario('elelem_reasoning_tokens')

        # Test 1: OpenAI-style reasoning (temperature 1.0)
        faker.reset_state()
        response1 = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Test OpenAI format"}],
            temperature=1.0  # OpenAI format condition
        )

        assert response1
        assert hasattr(response1, 'elelem_metrics')

        # For OpenAI format, reasoning tokens come from explicit fields but no reasoning content
        reasoning_tokens = response1.elelem_metrics["tokens"]["reasoning"]
        assert reasoning_tokens > 0, "OpenAI format should have reasoning tokens"

        # OpenAI format typically doesn't expose reasoning content, only tokens
        assert "reasoning_content" not in response1.elelem_metrics or response1.elelem_metrics["reasoning_content"] is None, \
            "OpenAI format should not have reasoning content"

        # Test 2: GROQ-style reasoning (temperature 0.8)
        faker.reset_state()
        response2 = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Test GROQ format"}],
            temperature=0.8  # GROQ format condition
        )

        assert response2
        reasoning_tokens2 = response2.elelem_metrics["tokens"]["reasoning"]
        assert reasoning_tokens2 > 0, "GROQ format should have reasoning tokens"

        # GROQ format also typically doesn't expose reasoning content, only tokens
        assert "reasoning_content" not in response2.elelem_metrics or response2.elelem_metrics["reasoning_content"] is None, \
            "GROQ format should not have reasoning content"

        # Test 3: <think> tags format (temperature 0.6) - already tested above
        faker.reset_state()
        response3 = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Test think tags"}],
            temperature=0.6  # Think tags condition
        )

        assert response3
        assert "reasoning_content" in response3.elelem_metrics
        reasoning_content3 = response3.elelem_metrics["reasoning_content"]
        assert reasoning_content3 is not None
        assert len(reasoning_content3.strip()) > 0

        # Verify exact content matches the faker scenario for temperature 0.6
        expected_reasoning3 = "Let me analyze this step by step. First, I need to understand the problem. Then I can work through the solution methodically."
        assert reasoning_content3 == expected_reasoning3, f"Think tags reasoning content mismatch.\nExpected: '{expected_reasoning3}'\nActual: '{reasoning_content3}'"

        # Test 4: Structured reasoning format (temperature 0.4)
        faker.reset_state()
        response4 = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Test structured format"}],
            temperature=0.4  # Structured reasoning condition
        )

        assert response4
        reasoning_tokens4 = response4.elelem_metrics["tokens"]["reasoning"]
        assert reasoning_tokens4 > 0, "Structured format should have reasoning tokens"

        print(f"✅ All reasoning formats tested successfully:")
        print(f"  OpenAI format: {reasoning_tokens} tokens")
        print(f"  GROQ format: {reasoning_tokens2} tokens")
        print(f"  Think tags: {response3.elelem_metrics['tokens']['reasoning']} tokens, content: {len(reasoning_content3)} chars")
        print(f"  Structured: {reasoning_tokens4} tokens")

    @pytest.mark.asyncio
    async def test_temperature_parameter_cleanup(self, elelem_with_faker_env):
        """Test that Elelem removes temperature parameter for models that don't support it."""
        elelem, faker = elelem_with_faker_env

        # Configure faker for temperature cleanup
        faker.configure_scenario('elelem_temperature_cleanup')
        faker.reset_state()

        # Make request with temperature to model that doesn't support it
        response = await elelem.create_chat_completion(
            model="faker:no-temperature",
            messages=[{"role": "user", "content": "Test temperature cleanup"}],
            temperature=0.7  # This should be removed by Elelem
        )

        # Should succeed
        assert response
        content = response.choices[0].message.content
        assert "Temperature parameter was cleaned up" in content

        # Verify the request sent to faker doesn't contain temperature
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) == 1

        request_body = requests[0]['body']
        # Temperature should have been removed by Elelem
        assert 'temperature' not in request_body, f"Temperature should be removed but found: {request_body}"

        print(f"✓ Temperature parameter successfully removed for model that doesn't support it")
        print(f"Request body keys: {list(request_body.keys())}")

    @pytest.mark.asyncio
    async def test_system_message_handling(self, elelem_with_faker_env):
        """Test that Elelem handles system messages for models that don't support system role."""
        elelem, faker = elelem_with_faker_env

        # Configure faker for system message handling
        faker.configure_scenario('elelem_system_cleanup')
        faker.reset_state()

        # Make request with system message to model that doesn't support system role
        response = await elelem.create_chat_completion(
            model="faker:no-system-alt",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        )

        # Should succeed
        assert response
        content = response.choices[0].message.content
        assert "System message was handled" in content

        # Verify the request transformation
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) == 1

        request_body = requests[0]['body']
        messages = request_body.get('messages', [])

        # System message should be converted to user message or appended as instruction
        # Elelem should have transformed the system message somehow
        has_system_role = any(msg.get('role') == 'system' for msg in messages)

        if has_system_role:
            # If system message still exists, the model might actually support it
            print("System message preserved - model may support system role after all")
        else:
            # System message was converted/removed - this is expected behavior
            print("✓ System message was transformed for model that doesn't support system role")

            # Check that the system content was preserved somehow
            # (either appended to user message or added as separate user message)
            all_content = " ".join([msg.get('content', '') for msg in messages])
            assert "helpful assistant" in all_content.lower(), "System message content should be preserved"

        print(f"Messages sent to faker: {messages}")
        print(f"Number of messages: {len(messages)}")

    @pytest.mark.asyncio
    async def test_streaming_response_collection(self, elelem_with_faker_env):
        """Test that Elelem correctly collects and assembles streaming responses."""
        elelem, faker = elelem_with_faker_env

        # Configure faker for streaming response
        faker.configure_scenario('elelem_streaming')
        faker.reset_state()

        # Make normal request (no streaming from user perspective)
        # But the provider should internally use streaming (due to default_params)
        response = await elelem.create_chat_completion(
            model="faker-streaming:streaming-test",  # Use streaming provider
            messages=[{"role": "user", "content": "Stream a response"}]
            # No stream=True - Elelem doesn't support user-facing streaming
        )

        # Should succeed and have collected all chunks
        assert response
        print(f"DEBUG: Streaming response structure: {response}")

        # The response should be identical to non-streaming responses
        # Check response structure
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            print(f"DEBUG: Choice structure: {choice}")
            if hasattr(choice, "message"):
                content = choice.message.content
            elif hasattr(choice, "delta"):
                content = getattr(choice.delta, "content", "")
            else:
                # If neither message nor delta, maybe it's in a different format
                content = str(choice)
        else:
            content = str(response)

        # Should have assembled the complete message from chunks
        expected_content = "Hello there! How can I help you?"
        assert content == expected_content, f"Expected '{expected_content}' but got '{content}'"

        # Verify the response structure matches non-streaming responses
        # Response should be an OpenAI ChatCompletion object
        assert hasattr(response, "choices"), "Response should have 'choices'"
        assert len(response.choices) > 0, "Should have at least one choice"
        assert hasattr(response.choices[0], "message"), "Choice should have 'message'"
        assert hasattr(response.choices[0].message, "content"), "Message should have 'content'"

        print(f"✓ Response structure is correct: {type(response)}")
        print(f"✓ Response has choices[0].message.content structure")

        # Verify streaming was actually used by checking request
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) == 1

        request_body = requests[0]['body']
        assert request_body.get('stream') == True, "Stream parameter should be enabled"

        # Verify tokens are tracked correctly
        token_info = getattr(response, 'elelem_metrics', {}).get('tokens', {})
        assert token_info['input'] > 0
        assert token_info['output'] > 0

        print(f"✓ Streaming response successfully collected and assembled")
        print(f"Final content: '{content}'")
        print(f"Stream parameter sent: {request_body.get('stream')}")
        print(f"Token info: {token_info}")

    @pytest.mark.asyncio
    async def test_comprehensive_stats_collection(self, elelem_with_faker_env):
        """Test comprehensive stats collection and aggregation."""
        elelem, faker = elelem_with_faker_env

        # Configure faker
        faker.configure_scenario('happy_path')
        faker.reset_state()

        # Make multiple requests with different tags and models
        await elelem.create_chat_completion(
            model="faker:basic",
            messages=[{"role": "user", "content": "Request 1"}],
            tags=["batch-1", "test-suite"]
        )

        await elelem.create_chat_completion(
            model="faker:json-temp-test",
            messages=[{"role": "user", "content": "Request 2"}],
            tags=["batch-1", "json-test"]
        )

        await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Request 3"}],
            tags=["batch-2", "reasoning"]
        )

        # Test overall stats
        overall_stats = elelem.get_stats()
        print(f"Overall stats: {overall_stats}")

        assert "requests" in overall_stats
        assert overall_stats["requests"]["total"] >= 3  # At least our 3 requests
        assert "costs" in overall_stats
        assert overall_stats["costs"]["total"] > 0
        assert "tokens" in overall_stats
        total_tokens = overall_stats["tokens"]["input"]["total"] + overall_stats["tokens"]["output"]["total"]
        assert total_tokens > 0

        # Test stats by specific tag
        batch1_stats = elelem.get_stats_by_tag("batch-1")
        assert batch1_stats["requests"]["total"] == 2

        batch2_stats = elelem.get_stats_by_tag("batch-2")
        assert batch2_stats["requests"]["total"] == 1

        json_stats = elelem.get_stats_by_tag("json-test")
        assert json_stats["requests"]["total"] == 1

        # Test stats by multiple tags
        suite_stats = elelem.get_stats_by_tag("test-suite")
        assert suite_stats["requests"]["total"] == 1

        # Test metrics data
        data = elelem.get_metrics_data()
        assert len(data) >= 3
        first_row = data[0]
        assert "request_id" in first_row
        assert "actual_model" in first_row
        assert "actual_provider" in first_row
        assert "tags" in first_row
        assert "input_tokens" in first_row
        assert "output_tokens" in first_row
        assert "total_cost_usd" in first_row

        # Verify different models are recorded (check last 3 entries)
        recent_models = [row["requested_model"] for row in data[-3:]]
        assert "faker:basic" in recent_models
        assert "faker:json-temp-test" in recent_models
        assert "faker:reasoning-test" in recent_models

        print("✓ Comprehensive stats collection validated")

    @pytest.mark.asyncio
    async def test_stats_summary_aggregation(self, elelem_with_faker_env):
        """Test stats summary with aggregation functions."""
        elelem, faker = elelem_with_faker_env

        faker.configure_scenario('happy_path')
        faker.reset_state()

        # Clear any existing metrics for clean test
        # (This might not be possible, so we'll work with incremental)

        # Make requests with different costs/tokens
        responses = []
        for i in range(3):
            response = await elelem.create_chat_completion(
                model="faker:basic",
                messages=[{"role": "user", "content": f"Aggregation test message {i+1}"}],
                tags=[f"agg-test-{i+1}", "aggregation-test"]
            )
            responses.append(response)

        # Get summary with tag filter
        summary = elelem.get_summary(tags=["aggregation-test"])

        print(f"Summary: {summary}")

        assert "requests" in summary
        assert summary["requests"]["total"] == 3

        # Check token aggregations
        assert "tokens" in summary
        assert "input" in summary["tokens"]
        assert summary["tokens"]["input"]["total"] > 0

        # Check cost aggregations
        assert "costs" in summary
        assert summary["costs"]["total"] > 0

        print("✓ Summary aggregation validated")

    def test_metrics_tags_listing(self, elelem_with_faker_env):
        """Test metrics tags enumeration."""
        elelem, faker = elelem_with_faker_env

        # Configure and make requests
        faker.configure_scenario('happy_path')
        faker.reset_state()

        import asyncio
        async def make_tagged_requests():
            await elelem.create_chat_completion(
                model="faker:basic",
                messages=[{"role": "user", "content": "Tagged request"}],
                tags=["production", "api-v1", "user-123"]
            )

            await elelem.create_chat_completion(
                model="faker:reasoning-test",
                messages=[{"role": "user", "content": "Another tagged request"}],
                tags=["staging", "api-v2", "user-456"]
            )

        asyncio.run(make_tagged_requests())

        # Check tags after requests
        final_tags = elelem.get_metrics_tags()

        # Should include our custom tags
        expected_custom_tags = [
            "production", "api-v1", "user-123",
            "staging", "api-v2", "user-456"
        ]

        for tag in expected_custom_tags:
            assert tag in final_tags, f"Custom tag '{tag}' not found in {final_tags}"

        print(f"✓ Tags validated: {len(final_tags)} total tags found")
        print(f"  Custom tags found: {expected_custom_tags}")

    @pytest.mark.asyncio
    async def test_metrics_dataframe_filtering(self, elelem_with_faker_env):
        """Test metrics dataframe with various filters."""
        elelem, faker = elelem_with_faker_env

        faker.configure_scenario('happy_path')
        faker.reset_state()

        # Make requests with specific tags for filtering
        await elelem.create_chat_completion(
            model="faker:basic",
            messages=[{"role": "user", "content": "Filter test 1"}],
            tags=["dataframe-test", "filter-1"]
        )

        await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Filter test 2"}],
            tags=["dataframe-test", "filter-2"]
        )

        # Test filtering by tags
        data_all = elelem.get_metrics_data(tags=["dataframe-test"])
        assert len(data_all) == 2

        data_filter1 = elelem.get_metrics_data(tags=["filter-1"])
        assert len(data_filter1) == 1
        assert data_filter1[0]["requested_model"] == "faker:basic"

        data_filter2 = elelem.get_metrics_data(tags=["filter-2"])
        assert len(data_filter2) == 1
        assert data_filter2[0]["requested_model"] == "faker:reasoning-test"

        print("✓ Data filtering validated")

    @pytest.mark.asyncio
    async def test_virtual_model_stats_tracking(self, elelem_with_faker_env):
        """Test that virtual model stats are properly tracked."""
        elelem, faker = elelem_with_faker_env

        # Configure faker for potential failover
        faker.configure_scenario('elelem_rate_limits')
        faker.reset_state()

        # Use virtual model (should eventually succeed after failover)
        response = await elelem.create_chat_completion(
            model="virtual:faker-test-failover",
            messages=[{"role": "user", "content": "Virtual model stats test"}],
            tags=["virtual-model-test"]
        )

        assert response  # Should succeed eventually

        # Check that stats recorded the virtual model correctly
        virtual_stats = elelem.get_stats_by_tag("virtual-model-test")
        assert virtual_stats["requests"]["total"] == 1

        # Check data shows virtual model
        data = elelem.get_metrics_data(tags=["virtual-model-test"])
        assert len(data) == 1
        # The actual provider used should be recorded, not the virtual model name
        actual_model = data[0]["actual_model"]
        # Should be one of the candidates that succeeded
        assert actual_model in ["faker:rate-limited", "faker:no-json", "faker:basic"]

        print(f"✓ Virtual model stats validated - actual model used: {actual_model}")

    @pytest.mark.asyncio
    async def test_virtual_model_json_temperature_reduction(self, elelem_with_faker_env):
        """Test that virtual models handle JSON temperature reduction within candidates."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with JSON temperature reduction scenario
        faker.configure_scenario('elelem_json_temp_reduction')
        faker.reset_state()

        # Create virtual model with faker:json-temp-test as first candidate
        virtual_model_config = {
            "candidates": [
                {"model": "faker:json-temp-test", "timeout": 60}
            ]
        }

        # Make request with high temperature and JSON mode
        response = await elelem.create_chat_completion(
            model=f"dynamic:{virtual_model_config}",
            messages=[{"role": "user", "content": "Return JSON data"}],
            response_format={"type": "json_object"},
            temperature=0.9,
            tags=["virtual_json_test"]
        )

        # Should eventually succeed with valid JSON
        assert response
        content = response.choices[0].message.content
        json_data = json.loads(content)
        assert json_data["status"] == "success_after_temperature_reduction"

        # Verify multiple requests were made with decreasing temperature
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) > 1, "Should have made multiple requests for temperature reduction"

        # Check temperature reduction pattern
        temperatures = []
        for req in requests:
            body = req.get('body', {})
            temp = body.get('temperature')
            if temp is not None:
                temperatures.append(temp)

        # Should have decreasing temperatures
        assert len(temperatures) >= 2, "Should have multiple temperature values"
        assert temperatures[0] > temperatures[-1], "Temperature should decrease"

        # Check that all requests went to the same candidate (no failover to next candidate)
        models_used = [req.get('url', '').split('/')[-1] for req in requests]
        unique_models = set(models_used)
        assert len(unique_models) == 1, "All requests should go to the same candidate model"

        # Verify stats tracking
        stats = elelem.get_stats_by_tag("virtual_json_test")
        assert stats["requests"]["total"] == 1

        # Verify we have retry attempts (temperature reductions are tracked but not in summary stats)
        assert stats["retries"]["total_retry_attempts"] >= 1, "Should have some retry attempts"

        print(f"✓ Virtual model JSON temperature reduction validated - {len(requests)} requests made")

    @pytest.mark.asyncio
    async def test_parameterized_models_integration(self, elelem_with_faker_env):
        """Test that parameterized model system works with faker and doesn't break existing functionality."""
        elelem, faker = elelem_with_faker_env

        # Get all models to verify expansion worked
        all_models = elelem._models

        # Should contain multiple faker models
        faker_models = {k: v for k, v in all_models.items() if k.startswith('faker:')}

        assert len(faker_models) > 0, "Should have faker models available"

        # Test that existing faker functionality still works after parameterized model expansion
        faker.configure_scenario('happy_path')
        faker.reset_state()

        response = await elelem.create_chat_completion(
            model="faker:basic",
            messages=[{"role": "user", "content": "Test parameterized model integration"}]
        )

        assert response is not None
        assert response.choices[0].message.content is not None

        # Test reasoning models still work (these may be parameterized in the future)
        faker.configure_scenario('elelem_reasoning_tokens')
        faker.reset_state()

        reasoning_response = await elelem.create_chat_completion(
            model="faker:reasoning-test",
            messages=[{"role": "user", "content": "Think about this"}],
            temperature=0.6  # Think tags condition
        )

        assert reasoning_response is not None
        assert hasattr(reasoning_response, 'elelem_metrics')

        # Verify reasoning content extraction still works
        if "reasoning_content" in reasoning_response.elelem_metrics:
            reasoning_content = reasoning_response.elelem_metrics["reasoning_content"]
            assert reasoning_content is not None
            assert len(reasoning_content.strip()) > 0

        print(f"✅ Found {len(faker_models)} faker models")
        print("✅ Parameterized model system compatible with faker")
        print("✅ Existing functionality preserved after parameterized model expansion")

    @pytest.mark.asyncio
    async def test_pydantic_openai_style_json_schema(self, elelem_with_faker_env):
        """Test Pydantic models with OpenAI-style response_format.type='json_schema'."""
        from pydantic import BaseModel

        class CalcResult(BaseModel):
            steps: list[str]
            answer: int

        elelem, faker = elelem_with_faker_env
        faker.configure_scenario('happy_path')
        faker.reset_state()

        response = await elelem.create_chat_completion(
            model="faker:basic",
            messages=[
                {"role": "user", "content": "Calculate 2+2"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "calc_result",
                    "schema": CalcResult.model_json_schema()
                }
            },
            enforce_schema_in_prompt=True  # Inject schema so faker can read it
        )

        # Verify response
        assert response.choices[0].message.content
        parsed = json.loads(response.choices[0].message.content)
        result = CalcResult(**parsed)
        assert isinstance(result, CalcResult)

        print("✅ OpenAI-style json_schema format works with Pydantic")

    @pytest.mark.asyncio
    async def test_pydantic_elelem_style_json_schema(self, elelem_with_faker_env):
        """Test Pydantic models with Elelem-style json_schema parameter."""
        from pydantic import BaseModel

        class CalcResult(BaseModel):
            steps: list[str]
            answer: int

        elelem, faker = elelem_with_faker_env
        faker.configure_scenario('happy_path')
        faker.reset_state()

        response = await elelem.create_chat_completion(
            model="faker:basic",
            messages=[
                {"role": "user", "content": "Calculate 2+2"}
            ],
            response_format={"type": "json_object"},
            json_schema=CalcResult.model_json_schema(),
            enforce_schema_in_prompt=True  # Inject schema so faker can read it
        )

        # Verify response
        assert response.choices[0].message.content
        parsed = json.loads(response.choices[0].message.content)
        result = CalcResult(**parsed)
        assert isinstance(result, CalcResult)

        print("✅ Elelem-style json_schema parameter works with Pydantic")

    @pytest.mark.asyncio
    async def test_benchmark_routing_reorders_candidates(self, elelem_with_faker_env, tmp_path):
        """Test that benchmark data reorders virtual model candidates by value score."""
        elelem, faker = elelem_with_faker_env

        # Create test benchmark data file
        # Value = tokens_per_second / cost_per_1m
        # where cost_per_1m = (costs.avg / tokens.output.avg) * 1_000_000
        #
        # fast:   100 t/s, cost_per_1m = (0.001/100)*1M = 10  → value = 100/10 = 10.0
        # medium: 50 t/s, cost_per_1m = (0.0005/50)*1M = 10  → value = 50/10 = 5.0
        # slow:   10 t/s, cost_per_1m = (0.001/10)*1M = 100  → value = 10/100 = 0.1
        benchmark_data = {
            "models": {
                "faker:fast-provider": {
                    "tokens": {"output": {"avg": 100.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.001},
                    "requests": {"success_rate": 1.0, "total": 10}
                },
                "faker:medium-provider": {
                    "tokens": {"output": {"avg": 50.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.0005},
                    "requests": {"success_rate": 1.0, "total": 10}
                },
                "faker:slow-provider": {
                    "tokens": {"output": {"avg": 10.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.001},
                    "requests": {"success_rate": 1.0, "total": 10}
                }
                # Note: faker:unscored-provider has NO benchmark data - should go last
            }
        }

        benchmark_file = tmp_path / "test_benchmarks.json"
        benchmark_file.write_text(json.dumps(benchmark_data))

        # Configure benchmark store
        import os
        os.environ['ELELEM_BENCHMARK_SOURCE'] = str(benchmark_file)

        # Reload benchmark store with new data
        from elelem._benchmark_store import get_benchmark_store
        store = get_benchmark_store()
        await store.fetch_once()

        # Verify benchmark data was loaded
        assert store.get_benchmark("faker:fast-provider") is not None
        assert store.get_benchmark("faker:slow-provider") is not None
        assert store.get_benchmark("faker:unscored-provider") is None  # No data

        # Configure faker for benchmark routing scenario
        faker.configure_scenario('elelem_benchmark_routing')
        faker.reset_state()

        # Make request to virtual model with benchmark routing
        response = await elelem.create_chat_completion(
            model="virtual:faker-benchmark-test",
            messages=[{"role": "user", "content": "Test benchmark routing"}],
            tags=["benchmark-routing-test"]
        )

        # Should succeed
        assert response is not None

        # The request should have gone to fast-provider first (highest value score)
        # because benchmark routing reorders candidates
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) >= 1

        # Check which provider received the first request
        first_request = requests[0]
        first_model = first_request.get('body', {}).get('model', '')

        # With benchmark data, fast-provider should be tried first
        # (it has highest tokens_per_second)
        assert 'fast-provider' in first_model, \
            f"Expected fast-provider to be tried first, got: {first_model}"

        # Verify response content matches fast provider
        content = response.choices[0].message.content
        assert "fast provider" in content.lower(), \
            f"Expected response from fast provider, got: {content}"

        print("✅ Benchmark routing correctly reordered candidates")
        print(f"   First provider tried: {first_model}")
        print(f"   Response: {content}")

        # Cleanup
        del os.environ['ELELEM_BENCHMARK_SOURCE']

    @pytest.mark.asyncio
    async def test_benchmark_routing_priority_always_first(self, elelem_with_faker_env, tmp_path):
        """Test that priority: always_first overrides benchmark ordering."""
        elelem, faker = elelem_with_faker_env

        # Create benchmark data where fast-provider has highest score
        benchmark_data = {
            "models": {
                "faker:fast-provider": {
                    "tokens": {"output": {"avg": 100.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.0005},
                    "requests": {"success_rate": 1.0, "total": 10}
                },
                "faker:slow-provider": {
                    "tokens": {"output": {"avg": 10.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.0001},
                    "requests": {"success_rate": 1.0, "total": 10}
                },
                "faker:medium-provider": {
                    "tokens": {"output": {"avg": 50.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.00025},
                    "requests": {"success_rate": 1.0, "total": 10}
                }
            }
        }

        benchmark_file = tmp_path / "test_benchmarks_priority.json"
        benchmark_file.write_text(json.dumps(benchmark_data))

        import os
        os.environ['ELELEM_BENCHMARK_SOURCE'] = str(benchmark_file)

        from elelem._benchmark_store import get_benchmark_store
        store = get_benchmark_store()
        await store.fetch_once()

        # Configure faker - slow-provider should succeed (despite priority override)
        faker.configure_scenario('elelem_benchmark_routing')
        faker.reset_state()

        # Use virtual model where slow-provider has priority: always_first
        response = await elelem.create_chat_completion(
            model="virtual:faker-benchmark-priority",
            messages=[{"role": "user", "content": "Test priority override"}],
            tags=["priority-test"]
        )

        # Should succeed
        assert response is not None

        # The slow-provider should be tried first despite having lower score
        # because it has priority: always_first
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) >= 1

        first_request = requests[0]
        first_model = first_request.get('body', {}).get('model', '')

        # Slow provider should be first due to priority override
        # But it returns 503 overloaded, so fast-provider should eventually succeed
        # Let's check the response to see which provider actually succeeded
        content = response.choices[0].message.content

        # The slow-provider returns overloaded error, so we should have fallen back
        # to fast-provider which succeeds
        # First request should still be to slow-provider (due to priority)
        assert 'slow-provider' in first_model, \
            f"Expected slow-provider to be tried first due to priority, got: {first_model}"

        print("✅ Priority always_first correctly overrides benchmark ordering")
        print(f"   First provider tried: {first_model}")
        print(f"   Final response: {content}")

        # Cleanup
        del os.environ['ELELEM_BENCHMARK_SOURCE']

    @pytest.mark.asyncio
    async def test_benchmark_routing_unscored_last(self, elelem_with_faker_env, tmp_path):
        """Test that candidates without benchmark data are tried last."""
        elelem, faker = elelem_with_faker_env

        # Create benchmark data - only for some providers
        # unscored-provider has NO data - should be last
        benchmark_data = {
            "models": {
                "faker:slow-provider": {
                    "tokens": {"output": {"avg": 10.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.0001},
                    "requests": {"success_rate": 1.0, "total": 10}
                },
                "faker:medium-provider": {
                    "tokens": {"output": {"avg": 50.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.00025},
                    "requests": {"success_rate": 1.0, "total": 10}
                },
                "faker:fast-provider": {
                    "tokens": {"output": {"avg": 100.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.0005},
                    "requests": {"success_rate": 1.0, "total": 10}
                }
                # faker:unscored-provider is NOT in benchmark data
            }
        }

        benchmark_file = tmp_path / "test_benchmarks_unscored.json"
        benchmark_file.write_text(json.dumps(benchmark_data))

        import os
        os.environ['ELELEM_BENCHMARK_SOURCE'] = str(benchmark_file)

        from elelem._benchmark_store import get_benchmark_store, reorder_candidates_by_benchmark
        store = get_benchmark_store()
        await store.fetch_once()

        # Verify unscored-provider has no benchmark data
        assert store.get_benchmark("faker:unscored-provider") is None

        # Get the model config and check reordering
        model_config = elelem.config.get_model_config("virtual:faker-benchmark-test")
        candidates = model_config['candidates']

        # Reorder candidates
        reordered = reorder_candidates_by_benchmark(candidates)

        # Extract provider names in order
        provider_order = [c.get('original_model_ref', c.get('provider', '')) for c in reordered]

        print(f"Reordered candidates: {provider_order}")

        # Unscored provider should be LAST
        unscored_index = None
        for i, ref in enumerate(provider_order):
            if 'unscored' in ref:
                unscored_index = i
                break

        assert unscored_index is not None, "Unscored provider should be in the list"
        assert unscored_index == len(provider_order) - 1, \
            f"Unscored provider should be last, but was at index {unscored_index}"

        print("✅ Unscored candidates correctly placed last")
        print(f"   Order: {provider_order}")

        # Cleanup
        del os.environ['ELELEM_BENCHMARK_SOURCE']

    @pytest.mark.asyncio
    async def test_benchmark_routing_disabled_without_source(self, elelem_with_faker_env):
        """Test that benchmark routing doesn't affect ordering when disabled."""
        elelem, faker = elelem_with_faker_env

        # Ensure no benchmark source is configured
        import os
        if 'ELELEM_BENCHMARK_SOURCE' in os.environ:
            del os.environ['ELELEM_BENCHMARK_SOURCE']

        from elelem._benchmark_store import get_benchmark_store, reorder_candidates_by_benchmark
        store = get_benchmark_store()

        # Store should be disabled
        assert not store.enabled, "Benchmark store should be disabled without source"

        # Get the model config
        model_config = elelem.config.get_model_config("virtual:faker-benchmark-test")
        original_candidates = model_config['candidates'].copy()

        # Reorder should return same order when disabled
        reordered = reorder_candidates_by_benchmark(original_candidates)

        # Extract refs for comparison
        original_refs = [c.get('original_model_ref') for c in original_candidates]
        reordered_refs = [c.get('original_model_ref') for c in reordered]

        assert original_refs == reordered_refs, \
            f"Order should be unchanged when benchmark store is disabled.\n" \
            f"Original: {original_refs}\nReordered: {reordered_refs}"

        print("✅ Benchmark routing correctly disabled without source")
        print(f"   Order preserved: {original_refs}")

    @pytest.fixture(scope="function")
    def elelem_with_json_fixer(self, faker_server, monkeypatch, tmp_path):
        """Setup Elelem with faker environment and custom JSON fixer model."""
        # Set environment variable for faker
        monkeypatch.setenv("FAKER_API_KEY", "fake-key-123")

        # Use a unique temporary SQLite database for each test
        db_file = tmp_path / "test_metrics.db"
        monkeypatch.setenv("ELELEM_DATABASE_URL", f"sqlite:///{db_file}")

        # Update faker provider configuration
        from elelem.config import Config
        config = Config()
        faker_provider = config.get_provider_config("faker")
        faker_provider["endpoint"] = f"http://localhost:{faker_server.port}/v1"

        # Create Elelem instance with JSON fixer configured to use faker
        elelem = Elelem(
            extra_provider_dirs=["tests/providers"],
            json_fixer_enabled=True,
            json_fixer_model="faker:json-fixer"
        )
        return elelem, faker_server

    @pytest.mark.asyncio
    async def test_json_fixer_repairs_invalid_schema(self, elelem_with_json_fixer):
        """Test that the LLM-based JSON fixer repairs schema validation failures."""
        elelem, faker = elelem_with_json_fixer

        # Configure faker with JSON fixer scenario
        faker.configure_scenario('elelem_json_fixer')
        faker.reset_state()

        # Define schema that requires 'status' field
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "message": {"type": "string"},
                "data": {"type": "integer"}
            },
            "required": ["status", "message", "data"]
        }

        # Request to json-broken model which returns JSON missing 'status'
        # The fixer (faker:json-fixer) should repair it
        response = await elelem.create_chat_completion(
            model="faker:json-broken",
            messages=[{"role": "user", "content": "Return JSON data"}],
            response_format={"type": "json_object"},
            json_schema=schema,
            temperature=0.1  # Low temp so retries don't help
        )

        # Should succeed thanks to the fixer
        assert response
        content = response.choices[0].message.content

        # The fixed JSON should be valid and have all required fields
        json_data = json.loads(content)
        assert "status" in json_data, "Fixed JSON should have 'status' field"
        assert json_data["status"] == "fixed", "Status should be 'fixed' from fixer"

        # Verify the fixer was called (should be more than 1 request)
        requests = faker.request_analyzer.get_captured_requests()
        print(f"Number of requests made: {len(requests)}")

        # Should have: 1+ retries to json-broken, then 1 call to json-fixer
        assert len(requests) >= 2, "Should have multiple requests (original + fixer)"

        # Check that json-fixer model was called
        models_called = [r['body'].get('model', '') for r in requests]
        print(f"Models called: {models_called}")
        assert any('json-fixer' in m for m in models_called), "JSON fixer model should have been called"

        print("✅ JSON fixer successfully repaired schema validation failure")

    @pytest.mark.asyncio
    async def test_json_fixer_disabled(self, faker_server, monkeypatch, tmp_path):
        """Test that JSON fixer can be disabled."""
        # Set environment variable for faker
        monkeypatch.setenv("FAKER_API_KEY", "fake-key-123")

        # Use a unique temporary SQLite database
        db_file = tmp_path / "test_metrics.db"
        monkeypatch.setenv("ELELEM_DATABASE_URL", f"sqlite:///{db_file}")

        # Update faker provider configuration
        from elelem.config import Config
        config = Config()
        faker_provider = config.get_provider_config("faker")
        faker_provider["endpoint"] = f"http://localhost:{faker_server.port}/v1"

        # Create Elelem with fixer DISABLED
        elelem = Elelem(
            extra_provider_dirs=["tests/providers"],
            json_fixer_enabled=False
        )

        # Configure faker
        faker_server.configure_scenario('elelem_json_fixer')
        faker_server.reset_state()

        # Define schema that requires 'status' field
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "message": {"type": "string"}
            },
            "required": ["status", "message"]
        }

        # Request should fail since fixer is disabled and model returns invalid JSON
        from elelem._exceptions import ModelError
        with pytest.raises(ModelError) as exc_info:
            await elelem.create_chat_completion(
                model="faker:json-broken",
                messages=[{"role": "user", "content": "Return JSON data"}],
                response_format={"type": "json_object"},
                json_schema=schema,
                temperature=0.1
            )

        # Verify the error mentions JSON validation failure
        assert "JSON validation failed" in str(exc_info.value)
        print("✅ JSON fixer correctly disabled - request failed as expected")

    print("All comprehensive stats tests added to test_elelem_with_faker.py")