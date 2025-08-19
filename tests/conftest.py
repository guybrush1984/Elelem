"""
Pytest configuration and fixtures for Elelem tests.
"""

import pytest
import os
from unittest.mock import Mock, AsyncMock
from elelem import Elelem


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-deepinfra-key")


@pytest.fixture
def elelem_instance(mock_env_vars):
    """Create an Elelem instance for testing."""
    return Elelem()


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = '{"test": "response"}'
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].index = 0
    
    # Mock usage statistics
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.reasoning_tokens = 0
    
    return mock_response


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a JSON response with test data."}
    ]