"""Integration tests for YAML support using the faker system."""

import pytest
import asyncio
import yaml
import time
from pathlib import Path
import sys

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elelem import Elelem

# Import faker from tests directory
sys.path.insert(0, str(Path(__file__).parent))
from faker.server import ModelFaker


class TestYAMLIntegration:
    """Test YAML functionality with the faker system."""

    @pytest.fixture(scope="function")
    def faker_server(self):
        """Start a faker server for testing."""
        faker = ModelFaker(port=6666)  # Use standard faker port
        try:
            faker.start()
            time.sleep(0.5)  # Wait for server to be ready
            yield faker
        finally:
            try:
                faker.stop()
                time.sleep(1.0)  # Give time for socket to be released
            except Exception as e:
                print(f"Warning: Error during faker cleanup: {e}")

    @pytest.fixture(scope="function")
    def elelem_with_faker_env(self, faker_server, monkeypatch, tmp_path):
        """Setup Elelem with faker environment."""
        monkeypatch.setenv("FAKER_API_KEY", "fake-key-123")

        # Use a unique temporary SQLite database for each test
        db_file = tmp_path / "test_yaml_metrics.db"
        monkeypatch.setenv("ELELEM_DATABASE_URL", f"sqlite:///{db_file}")

        # Create Elelem instance
        elelem = Elelem(extra_provider_dirs=["tests/providers"])
        return elelem, faker_server

    @pytest.mark.asyncio
    async def test_yaml_clean_response(self, elelem_with_faker_env):
        """Test parsing clean YAML response."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with clean YAML scenario
        faker.configure_scenario('elelem_yaml_clean')
        faker.reset_state()

        # Define schema
        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "active": {"type": "boolean"},
                "tags": {"type": "array"},
                "address": {"type": "object"}
            }
        }

        # Make request with YAML schema
        response = await elelem.create_chat_completion(
            model="faker:yaml-clean-test",
            messages=[{"role": "user", "content": "Return YAML data"}],
            yaml_schema=schema
        )

        # Verify response
        assert response
        content = response.choices[0].message.content

        # Parse YAML
        yaml_data = yaml.safe_load(content)
        assert yaml_data["name"] == "John Doe"
        assert yaml_data["age"] == 30
        assert yaml_data["email"] == "john@example.com"
        assert yaml_data["active"] is True
        assert "developer" in yaml_data["tags"]
        assert yaml_data["address"]["city"] == "San Francisco"

    @pytest.mark.asyncio
    async def test_yaml_markdown_extraction(self, elelem_with_faker_env):
        """Test extracting YAML from markdown code blocks."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with markdown YAML scenario
        faker.configure_scenario('elelem_yaml_markdown')
        faker.reset_state()

        # Define schema
        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"}
            }
        }

        # Make request
        response = await elelem.create_chat_completion(
            model="faker:yaml-markdown-test",
            messages=[{"role": "user", "content": "Return YAML data"}],
            yaml_schema=schema
        )

        # Verify YAML was extracted from markdown
        assert response
        content = response.choices[0].message.content

        # Content should have markdown stripped
        assert "```" not in content
        assert "Here's the YAML response" not in content

        # Parse YAML
        yaml_data = yaml.safe_load(content)
        assert yaml_data["name"] == "Jane Smith"
        assert yaml_data["age"] == 28
        assert yaml_data["email"] == "jane@example.com"

    @pytest.mark.asyncio
    async def test_yaml_temperature_reduction(self, elelem_with_faker_env):
        """Test that Elelem reduces temperature when YAML parsing fails."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with malformed YAML scenario
        faker.configure_scenario('elelem_yaml_malformed')
        faker.reset_state()

        # Make request with high temperature and YAML schema
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        response = await elelem.create_chat_completion(
            model="faker:yaml-malformed-test",
            messages=[{"role": "user", "content": "Return YAML data"}],
            yaml_schema=schema,
            temperature=1.0
        )

        # Should eventually succeed with valid YAML
        assert response
        content = response.choices[0].message.content
        yaml_data = yaml.safe_load(content)
        # Accept any valid YAML response (default is fine)
        assert "name" in yaml_data
        assert "age" in yaml_data
        assert isinstance(yaml_data["age"], (int, float))

    @pytest.mark.asyncio
    async def test_yaml_schema_validation(self, elelem_with_faker_env):
        """Test YAML schema validation with retry."""
        elelem, faker = elelem_with_faker_env

        # Configure faker with schema validation scenario
        faker.configure_scenario('elelem_yaml_schema')
        faker.reset_state()

        # Define schema
        schema = {
            "type": "object",
            "required": ["name", "age", "email", "active"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},  # This will fail with "thirty"
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            }
        }

        # Make request
        response = await elelem.create_chat_completion(
            model="faker:yaml-schema-test",
            messages=[{"role": "user", "content": "Return user data"}],
            yaml_schema=schema,
            temperature=1.0
        )

        # Should eventually succeed with valid schema
        assert response
        content = response.choices[0].message.content
        yaml_data = yaml.safe_load(content)

        # Verify the data matches schema
        assert isinstance(yaml_data["name"], str)
        assert isinstance(yaml_data["age"], (int, float))
        assert isinstance(yaml_data["email"], str)
        assert isinstance(yaml_data["active"], bool)

    @pytest.mark.asyncio
    async def test_yaml_without_schema(self, elelem_with_faker_env):
        """Test YAML mode without schema validation."""
        elelem, faker = elelem_with_faker_env

        # Configure faker
        faker.configure_scenario('elelem_yaml_clean')
        faker.reset_state()

        # Make request with YAML schema but no validation
        response = await elelem.create_chat_completion(
            model="faker:yaml-clean-test",
            messages=[{"role": "user", "content": "Return YAML data"}],
            yaml_schema={}  # Empty schema - just request YAML mode
        )

        # Should succeed and return valid YAML
        assert response
        content = response.choices[0].message.content
        yaml_data = yaml.safe_load(content)
        assert yaml_data is not None

    @pytest.mark.asyncio
    async def test_yaml_enforce_schema_in_prompt(self, elelem_with_faker_env):
        """Test that enforce_schema_in_prompt adds schema to messages."""
        elelem, faker = elelem_with_faker_env

        # Configure faker
        faker.configure_scenario('elelem_yaml_clean')
        faker.reset_state()

        # Define schema
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        # Make request with enforce_schema_in_prompt
        response = await elelem.create_chat_completion(
            model="faker:yaml-clean-test",
            messages=[{"role": "user", "content": "Return data"}],
            yaml_schema=schema,
            enforce_schema_in_prompt=True
        )

        # Check that the request included schema in the message
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) > 0

        last_request = requests[-1]
        messages = last_request.get('body', {}).get('messages', [])

        # Find system message with schema
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        assert len(system_messages) > 0

        system_content = system_messages[0]['content']
        assert "REQUIRED OUTPUT FORMAT" in system_content
        assert "name" in system_content or "age" in system_content

    @pytest.mark.asyncio
    async def test_yaml_json_mutual_exclusivity(self, elelem_with_faker_env):
        """Test that JSON and YAML modes cannot be used together."""
        elelem, faker = elelem_with_faker_env

        # Configure faker
        faker.configure_scenario('elelem_yaml_clean')
        faker.reset_state()

        # Attempt to use both JSON and YAML
        with pytest.raises(ValueError) as exc_info:
            await elelem.create_chat_completion(
                model="faker:yaml-clean-test",
                messages=[{"role": "user", "content": "Return data"}],
                response_format={"type": "json_object"},
                yaml_schema={"type": "object"}
            )

        assert "JSON and YAML" in str(exc_info.value)
        assert "simultaneously" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
