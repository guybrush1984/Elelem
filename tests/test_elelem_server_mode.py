"""
Run existing tests but with Elelem server mode - the test manages everything.
"""

import pytest
import asyncio
import threading
import time
import subprocess
import sys
import requests
import pandas as pd
from openai import OpenAI


class ElelemServerManager:
    """Manages Elelem server for testing."""

    def __init__(self, port=8000):
        self.port = port
        self.process = None

    def start(self):
        """Start Elelem server."""
        # Check if port is already in use
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', self.port))
            sock.close()

            if result == 0:  # Port is in use
                print(f"Port {self.port} is already in use!")
                import os
                result = os.popen(f"lsof -i :{self.port}").read()
                print("Processes using the port:")
                print(result)
                raise Exception(f"Please kill the process using port {self.port}")
        except ImportError:
            pass

        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.elelem.server.main:app",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--log-level", "error"
        ]

        # Start server process with test provider directory
        env = {
            "FAKER_API_KEY": "fake-key",
            "ELELEM_EXTRA_PROVIDER_DIRS": "tests/providers",
            **dict(__import__("os").environ)
        }
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        # Wait for server to start and verify it's running
        # Server needs time to probe all providers during startup
        max_wait = 60  # seconds (providers are probed during init)
        wait_interval = 0.5
        total_waited = 0

        while total_waited < max_wait:
            time.sleep(wait_interval)
            total_waited += wait_interval

            # Check if process crashed
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                print(f"Server process exited with code {self.process.returncode}")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                raise Exception(f"Server failed to start (exit code {self.process.returncode})")

            # Try to connect
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', self.port))
                sock.close()
                if result == 0:
                    print(f"Server started successfully on port {self.port}")
                    time.sleep(1)  # Give it a moment to fully initialize
                    return
            except Exception:
                pass

        raise Exception(f"Server did not start within {max_wait} seconds")

    def stop(self):
        """Stop Elelem server."""
        if self.process:
            self.process.terminate()
            self.process.wait()


@pytest.fixture
def elelem_with_faker_env_server(faker_port=6666):
    """Modified fixture that starts both faker AND Elelem server."""
    from tests.faker.server import ModelFaker

    # Start faker as usual
    faker = ModelFaker(port=faker_port)
    faker.start()

    # Start Elelem server
    server = ElelemServerManager(port=8000)
    server.start()

    try:
        # Create OpenAI client pointing to our Elelem server
        client = OpenAI(api_key="test-key", base_url="http://127.0.0.1:8000/v1")

        class ServerElelem:
            def __init__(self, base_url="http://127.0.0.1:8000"):
                self.client = client
                self.base_url = base_url

            async def create_chat_completion(self, **kwargs):
                response = self.client.chat.completions.create(**kwargs)
                return {
                    'id': response.id,
                    'object': response.object,
                    'created': response.created,
                    'model': response.model,
                    'choices': [{
                        'index': choice.index,
                        'message': {
                            'role': choice.message.role,
                            'content': choice.message.content
                        },
                        'finish_reason': choice.finish_reason
                    } for choice in response.choices],
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                }

            def get_stats(self):
                """Get overall stats via server API."""
                response = requests.get(f"{self.base_url}/v1/metrics/summary")
                return response.json()

            def get_stats_by_tag(self, tag):
                """Get stats by tag via server API."""
                response = requests.get(f"{self.base_url}/v1/metrics/summary?tags={tag}")
                return response.json()

            def get_metrics_dataframe(self, start_time=None, end_time=None, tags=None):
                """Get metrics dataframe via server API."""
                params = {}
                if start_time:
                    params['start_time'] = start_time.isoformat()
                if end_time:
                    params['end_time'] = end_time.isoformat()
                if tags:
                    params['tags'] = ','.join(tags) if isinstance(tags, list) else tags

                response = requests.get(f"{self.base_url}/v1/metrics/data", params=params)
                data = response.json()
                return pd.DataFrame(data)

            def get_summary(self, start_time=None, end_time=None, tags=None):
                """Get summary via server API."""
                params = {}
                if start_time:
                    params['start_time'] = start_time.isoformat()
                if end_time:
                    params['end_time'] = end_time.isoformat()
                if tags:
                    params['tags'] = ','.join(tags) if isinstance(tags, list) else tags

                response = requests.get(f"{self.base_url}/v1/metrics/summary", params=params)
                return response.json()

            def get_metrics_tags(self):
                """Get metrics tags via server API."""
                response = requests.get(f"{self.base_url}/v1/metrics/tags")
                return response.json()["tags"]

            def list_models(self):
                """List models via OpenAI SDK."""
                models = self.client.models.list()
                return {
                    'object': 'list',
                    'data': [
                        {'id': model.id, 'object': 'model', 'created': model.created}
                        for model in models.data
                    ]
                }

            @property
            def config(self):
                """Dummy config property - not needed for server mode tests."""
                class DummyConfig:
                    def get_provider_config(self, provider):
                        return {}
                return DummyConfig()

        elelem = ServerElelem()
        yield elelem, faker

    finally:
        server.stop()
        faker.stop()


@pytest.mark.asyncio
class TestElelemServerExtraBody:
    """Test Elelem-specific parameters via extra_body when using OpenAI SDK."""

    @pytest.mark.asyncio
    async def test_yaml_schema_via_extra_body(self, elelem_with_faker_env_server):
        """Test that yaml_schema works via extra_body through OpenAI SDK."""
        elelem, faker = elelem_with_faker_env_server

        # Configure faker with YAML scenario
        faker.configure_scenario('elelem_yaml_clean')
        faker.reset_state()

        # Define YAML schema
        yaml_schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            }
        }

        # Call through OpenAI SDK using extra_body
        response = elelem.client.chat.completions.create(
            model="faker:yaml-clean-test",
            messages=[{"role": "user", "content": "Return YAML user data"}],
            extra_body={
                "yaml_schema": yaml_schema
            }
        )

        # Verify response
        assert response
        assert response.choices[0].message.content

        # Parse YAML and verify structure
        import yaml
        yaml_data = yaml.safe_load(response.choices[0].message.content)
        assert "name" in yaml_data
        assert "age" in yaml_data
        assert "email" in yaml_data
        assert isinstance(yaml_data["age"], (int, float))

    @pytest.mark.asyncio
    async def test_json_schema_via_extra_body(self, elelem_with_faker_env_server):
        """Test that json_schema works via extra_body through OpenAI SDK."""
        elelem, faker = elelem_with_faker_env_server

        # Configure faker with JSON schema scenario
        faker.configure_scenario('elelem_json_schema')
        faker.reset_state()

        # Define JSON schema
        json_schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            }
        }

        # Call through OpenAI SDK using extra_body
        response = elelem.client.chat.completions.create(
            model="faker:json-schema-test",
            messages=[{"role": "user", "content": "Generate a user profile"}],
            response_format={"type": "json_object"},
            extra_body={
                "json_schema": json_schema
            },
            temperature=1.0
        )

        # Verify response
        assert response
        content = response.choices[0].message.content

        # Verify it's valid JSON matching schema
        import json
        parsed = json.loads(content)
        assert "name" in parsed
        assert "age" in parsed
        assert "email" in parsed
        assert isinstance(parsed["age"], (int, float))

    @pytest.mark.asyncio
    async def test_tags_via_extra_body(self, elelem_with_faker_env_server):
        """Test that tags work via extra_body through OpenAI SDK."""
        elelem, faker = elelem_with_faker_env_server

        # Configure faker
        faker.configure_scenario('happy_path')
        faker.reset_state()

        # Call with tags via extra_body
        response = elelem.client.chat.completions.create(
            model="faker:basic",
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={
                "tags": ["test-tag-extra-body", "server-mode"]
            }
        )

        # Verify response
        assert response
        assert response.choices[0].message.content

        # Verify tags were recorded (check via metrics API)
        time.sleep(0.5)  # Give metrics a moment to be written
        tags = elelem.get_metrics_tags()
        assert "test-tag-extra-body" in tags
        assert "server-mode" in tags

    @pytest.mark.asyncio
    async def test_enforce_schema_in_prompt_via_extra_body(self, elelem_with_faker_env_server):
        """Test that enforce_schema_in_prompt works via extra_body."""
        elelem, faker = elelem_with_faker_env_server

        # Configure faker
        faker.configure_scenario('elelem_json_schema')
        faker.reset_state()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        # Call with enforce_schema_in_prompt via extra_body
        response = elelem.client.chat.completions.create(
            model="faker:json-schema-test",
            messages=[{"role": "user", "content": "Generate data"}],
            response_format={"type": "json_object"},
            extra_body={
                "json_schema": schema,
                "enforce_schema_in_prompt": True
            },
            temperature=1.0
        )

        # Verify response
        assert response
        assert response.choices[0].message.content

        # Check that schema was in the messages sent to faker
        requests = faker.request_analyzer.get_captured_requests()
        assert len(requests) > 0

        # Check if schema instructions were added to messages
        last_request = requests[-1]
        messages = last_request.get('body', {}).get('messages', [])

        # Should have schema instructions in system message
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        assert len(system_messages) > 0
        assert "REQUIRED OUTPUT FORMAT" in system_messages[0]['content']

    @pytest.mark.asyncio
    async def test_cache_bypass_via_extra_body(self, elelem_with_faker_env_server):
        """Test that cache parameter works via extra_body."""
        elelem, faker = elelem_with_faker_env_server

        # Configure faker
        faker.configure_scenario('happy_path')
        faker.reset_state()

        # First call with cache enabled (default)
        response1 = elelem.client.chat.completions.create(
            model="faker:basic",
            messages=[{"role": "user", "content": "Test message"}]
        )

        # Second call with cache disabled via extra_body
        response2 = elelem.client.chat.completions.create(
            model="faker:basic",
            messages=[{"role": "user", "content": "Test message"}],
            extra_body={"cache": False}
        )

        # Both should succeed
        assert response1
        assert response2

    @pytest.mark.asyncio
    async def test_multiple_elelem_params_via_extra_body(self, elelem_with_faker_env_server):
        """Test using multiple Elelem parameters together via extra_body."""
        elelem, faker = elelem_with_faker_env_server

        # Configure faker
        faker.configure_scenario('elelem_yaml_clean')
        faker.reset_state()

        yaml_schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        # Call with multiple Elelem params
        response = elelem.client.chat.completions.create(
            model="faker:yaml-clean-test",
            messages=[{"role": "user", "content": "Generate YAML"}],
            extra_body={
                "yaml_schema": yaml_schema,
                "tags": ["multi-param-test"],
                "enforce_schema_in_prompt": True,
                "cache": False
            }
        )

        # Verify response
        assert response
        content = response.choices[0].message.content

        # Verify YAML is valid
        import yaml
        yaml_data = yaml.safe_load(content)
        assert "name" in yaml_data


if __name__ == "__main__":
    # Replace the fixture and run tests
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from test_elelem_with_faker import TestElelemWithFaker
    TestElelemWithFaker.elelem_with_faker_env = elelem_with_faker_env_server

    # Run all tests
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_elelem_with_faker.py",
        "-v"
    ])
    sys.exit(result.returncode)