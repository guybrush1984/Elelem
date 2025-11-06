"""
YAML validation functions for Elelem.

YAML support is entirely client-side via prompt engineering and validation.
Uses JSON Schema for validation since YAML is a superset of JSON
and JSON Schema is the mature industry standard.
"""

import yaml
import json
from typing import List, Dict, Any, Optional, Tuple
from jsonschema import validate, ValidationError


def validate_yaml_schema(yaml_obj: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a YAML object against a JSON Schema.

    Args:
        yaml_obj: The parsed YAML object to validate
        schema: The JSON Schema to validate against

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        validate(instance=yaml_obj, schema=schema)
        return True, None
    except ValidationError as e:
        # Build detailed error message
        error_parts = [f"Schema validation failed: {e.message}"]

        if e.path:
            path_str = ".".join(str(p) for p in e.path)
            error_parts.append(f"at path: {path_str}")

        if e.schema_path:
            schema_path_str = ".".join(str(p) for p in e.schema_path)
            error_parts.append(f"schema path: {schema_path_str}")

        error_msg = " | ".join(error_parts)
        return False, error_msg


def add_yaml_instructions_to_messages(messages: List[Dict[str, str]], supports_system: bool, yaml_schema: Optional[Dict[str, Any]] = None, enforce_schema_in_prompt: bool = False) -> List[Dict[str, str]]:
    """Add YAML formatting instructions to messages when YAML mode is requested.

    Args:
        messages: Original message list
        supports_system: Whether the model supports system messages
        yaml_schema: Optional schema to include in instructions for structured outputs
        enforce_schema_in_prompt: If True, force including schema in prompt (default False)
                                   By default, assumes user already included schema to save tokens
    """
    modified_messages = messages.copy()

    # Build YAML instruction
    yaml_instruction = (
        "\n\nCRITICAL: You must respond with ONLY clean YAML - no markdown, no code blocks, no extra text. "
        "Do not wrap the YAML in ```yaml``` or ```yml``` blocks or any other formatting. "
        "Return raw, valid YAML that can be parsed directly. "
        "Use proper YAML syntax with correct indentation. "
        "Any non-YAML content will cause a parsing error."
    )

    # Only include schema in prompt if explicitly requested via enforce_schema_in_prompt
    # By default (False), we assume the user has already included schema in their prompt
    if yaml_schema and enforce_schema_in_prompt:
        # Convert schema to pretty-printed format for the prompt
        schema_str = yaml.dump(yaml_schema, default_flow_style=False, sort_keys=False)
        yaml_instruction += (
            "\n\n=== REQUIRED OUTPUT FORMAT ===\n"
            "Your response MUST conform to this exact schema:\n\n"
            f"{schema_str}\n"
            "Follow the schema precisely:\n"
            "- Include all required fields\n"
            "- Use correct data types (string, number, boolean, array/list, object/mapping)\n"
            "- Do not add extra fields unless allowed by the schema\n"
            "- Respect any constraints (enums, patterns, min/max values)\n"
            "- Use proper YAML syntax with correct indentation (2 spaces per level)\n"
            "=== END REQUIRED FORMAT ==="
        )

    if supports_system:
        # Find system message and append instruction
        system_found = False
        for msg in modified_messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + yaml_instruction
                system_found = True
                break

        # If no system message found, add one
        if not system_found:
            modified_messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant." + yaml_instruction
            })
    else:
        # Model doesn't support system messages, add as user message
        modified_messages.append({
            "role": "user",
            "content": yaml_instruction.strip()
        })

    return modified_messages


def validate_yaml_response(content: str, yaml_schema: Optional[Any], api_error: Optional[Exception] = None) -> None:
    """Validate YAML response content and schema.

    Args:
        content: The response content to validate
        yaml_schema: Optional JSON Schema to validate against
        api_error: Optional API error that occurred

    Raises:
        json.JSONDecodeError: If YAML parsing or schema validation fails
                              (reused for retry compatibility with existing logic)
    """
    if api_error:
        # Force YAML validation to fail so it triggers retry logic
        # Use JSONDecodeError for compatibility with existing retry logic
        raise json.JSONDecodeError(f"API YAML validation failed: {str(api_error)}", "", 0)

    # First parse the YAML
    try:
        parsed_yaml = yaml.safe_load(content)
    except yaml.YAMLError as e:
        # Convert YAML parse error to JSONDecodeError for retry compatibility
        raise json.JSONDecodeError(f"YAML parsing failed: {str(e)}", "", 0)

    # Then validate against schema if provided
    if yaml_schema:
        is_valid, schema_error = validate_yaml_schema(parsed_yaml, yaml_schema)
        if not is_valid:
            # Treat schema validation failure like parse failure for retry logic
            raise json.JSONDecodeError(f"YAML schema validation failed: {schema_error}", "", 0)
