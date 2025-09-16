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

        # Wait for server to start
        time.sleep(2)

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