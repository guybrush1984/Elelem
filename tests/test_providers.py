"""
Test cases for provider implementations.
"""

import pytest
from unittest.mock import Mock, patch
from elelem.elelem import Elelem


class TestProviders:
    """Test cases for provider functionality."""
    
    def test_create_provider_client(self):
        """Test provider client creation."""
        with patch('elelem.elelem.openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            elelem = Elelem()
            client = elelem._create_provider_client(
                api_key="test-key",
                base_url="https://api.example.com/v1",
                timeout=120
            )
            
            # Verify OpenAI client was created with correct parameters
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.example.com/v1",
                timeout=120
            )
            
            assert client == mock_client