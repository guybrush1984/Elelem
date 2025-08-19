"""
Test cases for the main Elelem class.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from elelem import Elelem


class TestElelem:
    """Test cases for Elelem class functionality."""
    
    def test_elelem_initialization(self, elelem_instance):
        """Test that Elelem initializes correctly."""
        assert elelem_instance is not None
        assert hasattr(elelem_instance, '_statistics')
        assert hasattr(elelem_instance, '_tag_statistics')
        
    def test_parse_model_string_valid(self, elelem_instance):
        """Test model string parsing with valid input."""
        provider, model_name = elelem_instance._parse_model_string("openai:gpt-4.1-mini")
        assert provider == "openai"
        assert model_name == "gpt-4.1-mini"
        
    def test_parse_model_string_invalid_format(self, elelem_instance):
        """Test model string parsing with invalid format."""
        with pytest.raises(ValueError, match="Model must be in 'provider:model' format"):
            elelem_instance._parse_model_string("invalid-model")
            
    def test_parse_model_string_unknown_model(self, elelem_instance):
        """Test model string parsing with unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            elelem_instance._parse_model_string("openai:unknown-model")
            
    def test_remove_think_tags(self, elelem_instance):
        """Test removal of think tags from content."""
        content_with_tags = "Here is my response. <think>This is my thinking process</think> Final answer."
        cleaned_content = elelem_instance._remove_think_tags(content_with_tags)
        assert cleaned_content == "Here is my response.  Final answer."
        
    def test_remove_think_tags_multiline(self, elelem_instance):
        """Test removal of multiline think tags."""
        content_with_tags = """Response start.
<think>
Multi-line thinking
with multiple lines
</think>
Response end."""
        cleaned_content = elelem_instance._remove_think_tags(content_with_tags)
        assert "<think>" not in cleaned_content
        assert "Response start." in cleaned_content
        assert "Response end." in cleaned_content
        
    def test_extract_json_from_markdown(self, elelem_instance):
        """Test JSON extraction from markdown code blocks."""
        markdown_content = '''Here is the JSON:
```json
{"key": "value", "number": 42}
```
End of response.'''
        
        extracted = elelem_instance._extract_json_from_markdown(markdown_content)
        assert extracted == '{"key": "value", "number": 42}'
        
    def test_calculate_costs(self, elelem_instance):
        """Test cost calculation functionality."""
        costs = elelem_instance._calculate_costs("openai:gpt-4.1-mini", 1000, 2000, 0)
        
        assert "input_cost_usd" in costs
        assert "output_cost_usd" in costs
        assert "total_cost_usd" in costs
        assert costs["total_cost_usd"] == costs["input_cost_usd"] + costs["output_cost_usd"]
        
    def test_get_stats_empty(self, elelem_instance):
        """Test getting statistics when no calls have been made."""
        stats = elelem_instance.get_stats()
        
        expected_keys = [
            "total_input_tokens", "total_output_tokens", "total_tokens",
            "total_input_cost_usd", "total_output_cost_usd", "total_cost_usd",
            "total_calls", "total_duration_seconds", "avg_duration_seconds",
            "reasoning_tokens", "reasoning_cost_usd"
        ]
        
        for key in expected_keys:
            assert key in stats
            assert stats[key] == 0 or stats[key] == 0.0
            
    def test_get_stats_by_tag_empty(self, elelem_instance):
        """Test getting statistics by tag when no calls have been made."""
        stats = elelem_instance.get_stats_by_tag("test_tag")
        
        expected_keys = [
            "total_input_tokens", "total_output_tokens", "total_tokens",
            "total_input_cost_usd", "total_output_cost_usd", "total_cost_usd",
            "total_calls", "total_duration_seconds", "avg_duration_seconds",
            "reasoning_tokens", "reasoning_cost_usd"
        ]
        
        for key in expected_keys:
            assert key in stats
            assert stats[key] == 0 or stats[key] == 0.0


@pytest.mark.asyncio
class TestElelemAsync:
    """Test cases for async Elelem functionality."""
    
    @patch('elelem.elelem.openai.AsyncOpenAI')
    async def test_create_chat_completion_success(self, mock_openai_class, elelem_instance, sample_messages, mock_openai_response):
        """Test successful chat completion."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        # Override the provider in elelem instance
        elelem_instance._providers = {"openai": mock_client}
        
        # Make the API call
        response = await elelem_instance.create_chat_completion(
            messages=sample_messages,
            model="openai:gpt-4.1-mini",
            tags=["test"]
        )
        
        # Verify response structure
        assert hasattr(response, 'choices')
        assert len(response.choices) == 1
        assert "message" in response.choices[0]
        assert "content" in response.choices[0]["message"]
        
        # Verify statistics were updated
        stats = elelem_instance.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_input_tokens"] > 0
        assert stats["total_output_tokens"] > 0