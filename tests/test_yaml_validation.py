"""Unit tests for YAML validation functions."""

import pytest
import json
import yaml
from pathlib import Path
import sys

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elelem._yaml_validation import (
    validate_yaml_schema,
    add_yaml_instructions_to_messages,
    validate_yaml_response
)
from elelem._response_processing import extract_yaml_from_markdown
import logging


@pytest.fixture
def logger():
    """Create a logger for testing."""
    return logging.getLogger("test_yaml")


class TestYAMLSchemaValidation:
    """Test YAML schema validation functions."""

    def test_valid_yaml_schema(self):
        """Test validation succeeds for valid YAML."""
        yaml_obj = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "active": True
        }

        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string", "format": "email"},
                "active": {"type": "boolean"}
            }
        }

        is_valid, error = validate_yaml_schema(yaml_obj, schema)
        assert is_valid is True
        assert error is None

    def test_invalid_yaml_schema_missing_field(self):
        """Test validation fails when required field is missing."""
        yaml_obj = {
            "name": "John Doe",
            "email": "john@example.com"
        }

        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"}
            }
        }

        is_valid, error = validate_yaml_schema(yaml_obj, schema)
        assert is_valid is False
        assert error is not None
        assert "age" in error.lower() or "required" in error.lower()

    def test_invalid_yaml_schema_wrong_type(self):
        """Test validation fails when field has wrong type."""
        yaml_obj = {
            "name": "John Doe",
            "age": "thirty",  # Should be number
            "email": "john@example.com"
        }

        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"}
            }
        }

        is_valid, error = validate_yaml_schema(yaml_obj, schema)
        assert is_valid is False
        assert error is not None


class TestYAMLInstructions:
    """Test YAML instruction injection."""

    def test_add_yaml_instructions_to_system_message(self):
        """Test adding YAML instructions to system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return some data"}
        ]

        modified = add_yaml_instructions_to_messages(messages, supports_system=True)

        assert len(modified) == 2
        assert modified[0]["role"] == "system"
        assert "YAML" in modified[0]["content"]
        assert "clean YAML" in modified[0]["content"]
        assert "helpful assistant" in modified[0]["content"]

    def test_add_yaml_instructions_creates_system_message(self):
        """Test creating system message when none exists."""
        messages = [
            {"role": "user", "content": "Return some data"}
        ]

        modified = add_yaml_instructions_to_messages(messages, supports_system=True)

        assert len(modified) == 2
        assert modified[0]["role"] == "system"
        assert "YAML" in modified[0]["content"]

    def test_add_yaml_instructions_without_system_support(self):
        """Test adding YAML instructions as user message when system not supported."""
        messages = [
            {"role": "user", "content": "Return some data"}
        ]

        modified = add_yaml_instructions_to_messages(messages, supports_system=False)

        assert len(modified) == 2
        assert modified[1]["role"] == "user"
        assert "YAML" in modified[1]["content"]

    def test_add_yaml_instructions_with_schema(self):
        """Test including schema in YAML instructions."""
        messages = [
            {"role": "user", "content": "Return some data"}
        ]

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        modified = add_yaml_instructions_to_messages(
            messages,
            supports_system=True,
            yaml_schema=schema,
            enforce_schema_in_prompt=True
        )

        assert len(modified) == 2
        system_content = modified[0]["content"]
        assert "REQUIRED OUTPUT FORMAT" in system_content
        assert "name" in system_content
        assert "age" in system_content


class TestYAMLExtraction:
    """Test YAML extraction from markdown."""

    def test_extract_yaml_from_yaml_block(self, logger):
        """Test extraction from ```yaml block."""
        content = """Here's the data:

```yaml
name: John Doe
age: 30
email: john@example.com
```

That's all!"""

        extracted = extract_yaml_from_markdown(content, logger)
        assert "name: John Doe" in extracted
        assert "age: 30" in extracted
        assert "Here's the data" not in extracted
        assert "```" not in extracted

    def test_extract_yaml_from_yml_block(self, logger):
        """Test extraction from ```yml block."""
        content = """```yml
name: Jane Smith
age: 28
```"""

        extracted = extract_yaml_from_markdown(content, logger)
        assert "name: Jane Smith" in extracted
        assert "```" not in extracted

    def test_extract_yaml_no_markdown(self, logger):
        """Test that clean YAML is returned as-is."""
        content = """name: John Doe
age: 30
email: john@example.com"""

        extracted = extract_yaml_from_markdown(content, logger)
        assert extracted == content


class TestYAMLResponseValidation:
    """Test complete YAML response validation."""

    def test_validate_valid_yaml_response(self):
        """Test validating a valid YAML response."""
        content = """name: John Doe
age: 30
email: john@example.com
active: true"""

        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            }
        }

        # Should not raise
        validate_yaml_response(content, schema)

    def test_validate_invalid_yaml_syntax(self):
        """Test validating malformed YAML raises error."""
        content = """name: John Doe
age: [broken - missing bracket
email: john@example.com"""

        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        with pytest.raises(json.JSONDecodeError) as exc_info:
            validate_yaml_response(content, schema)

        assert "YAML parsing failed" in str(exc_info.value)

    def test_validate_yaml_schema_violation(self):
        """Test validating YAML that violates schema raises error."""
        content = """name: John Doe
age: thirty
email: john@example.com"""

        schema = {
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"}
            }
        }

        with pytest.raises(json.JSONDecodeError) as exc_info:
            validate_yaml_response(content, schema)

        assert "YAML schema validation failed" in str(exc_info.value)

    def test_validate_yaml_without_schema(self):
        """Test validating YAML without schema only checks syntax."""
        content = """name: John Doe
age: 30
tags:
  - developer
  - python"""

        # Should not raise - only syntax check
        validate_yaml_response(content, yaml_schema=None)

    def test_validate_with_api_error(self):
        """Test that API errors are propagated."""
        content = "some content"
        api_error = Exception("API validation failed")

        with pytest.raises(json.JSONDecodeError) as exc_info:
            validate_yaml_response(content, yaml_schema=None, api_error=api_error)

        assert "API YAML validation failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
