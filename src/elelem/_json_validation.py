"""
JSON validation functions for Elelem.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from json_repair import repair_json
from jsonschema import validate, ValidationError

logger = logging.getLogger("elelem")


def validate_json_schema(json_obj: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a JSON object against a JSON Schema.

    Args:
        json_obj: The parsed JSON object to validate
        schema: The JSON Schema to validate against

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        validate(instance=json_obj, schema=schema)
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


def is_json_validation_api_error(error: Exception) -> bool:
    """Check if the error is a json_validate_failed API error."""
    error_str = str(error).lower()
    return "json_validate_failed" in error_str and "400" in error_str


def add_json_instructions_to_messages(messages: List[Dict[str, str]], supports_system: bool, json_schema: Optional[Dict[str, Any]] = None, enforce_schema_in_prompt: bool = False) -> List[Dict[str, str]]:
    """Add JSON formatting instructions to messages when response_format is JSON.

    Args:
        messages: Original message list
        supports_system: Whether the model supports system messages
        json_schema: Optional JSON schema to include in instructions for structured outputs
        enforce_schema_in_prompt: If True, force including schema in prompt (default False)
                                   By default, assumes user already included schema to save tokens
    """
    modified_messages = messages.copy()

    # Build JSON instruction
    json_instruction = (
        "\n\nCRITICAL: You must respond with ONLY a clean JSON object - no markdown, no code blocks, no extra text. "
        "Do not wrap the JSON in ```json``` blocks or any other formatting. "
        "Return raw, valid JSON that can be parsed directly. "
        "Start your response with { and end with }. "
        "Any non-JSON content will cause a parsing error."
    )

    # Only include schema in prompt if explicitly requested via enforce_schema_in_prompt
    # By default (False), we assume the user has already included schema in their prompt
    if json_schema and enforce_schema_in_prompt:
        import json as json_module
        schema_str = json_module.dumps(json_schema, indent=2)
        json_instruction += (
            "\n\n=== REQUIRED OUTPUT FORMAT ===\n"
            "Your response MUST conform to this exact JSON schema:\n\n"
            f"{schema_str}\n\n"
            "Follow the schema precisely:\n"
            "- Include all required fields\n"
            "- Use correct data types (string, number, boolean, array, object)\n"
            "- Do not add extra fields unless allowed by the schema\n"
            "- Respect any constraints (enums, patterns, min/max values)\n"
            "=== END REQUIRED FORMAT ==="
        )

    if supports_system:
        # Find system message and append instruction
        system_found = False
        for msg in modified_messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + json_instruction
                system_found = True
                break

        # If no system message found, add one
        if not system_found:
            modified_messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant." + json_instruction
            })
    else:
        # Model doesn't support system messages, add as user message
        modified_messages.append({
            "role": "user",
            "content": json_instruction.strip()
        })

    return modified_messages


def parse_json_with_repair(content: str, request_id: str = None) -> tuple[Any, str, bool]:
    """Parse JSON content, attempting repair if standard parse fails.

    Args:
        content: The JSON string to parse
        request_id: Optional request ID for logging

    Returns:
        Tuple of (parsed_json, content_string, was_repaired)
        - parsed_json: The parsed JSON object
        - content_string: The JSON string (original if valid, repaired if fixed)
        - was_repaired: True if repair was needed

    Raises:
        json.JSONDecodeError: If parsing fails even after repair attempt
    """
    try:
        return json.loads(content), content, False
    except json.JSONDecodeError as original_error:
        # Attempt repair
        try:
            repaired = repair_json(content, return_objects=True)

            # Validate repair result is meaningful (not empty string/list/dict for non-trivial input)
            if repaired == "" or repaired == [] or repaired == {}:
                # Repair returned empty result - not a meaningful fix
                logger.debug(f"JSON repair returned empty result, treating as failure")
                raise original_error

            repaired_str = json.dumps(repaired)
            req_id = f"[{request_id}] " if request_id else ""
            logger.warning(
                f"{req_id}JSON repair successful - original error: {str(original_error)[:100]}"
            )
            return repaired, repaired_str, True
        except json.JSONDecodeError:
            # Re-raise JSONDecodeError (including the one we raised above)
            raise
        except Exception as repair_error:
            # Other repair failures
            logger.debug(f"JSON repair failed: {repair_error}")
            raise original_error


def validate_json_response(content: str, json_schema: Optional[Any], api_error: Optional[Exception], request_id: str = None) -> str:
    """Validate JSON response content and schema.

    Args:
        content: The JSON string to validate
        json_schema: Optional schema to validate against
        api_error: Optional API error that triggered validation
        request_id: Optional request ID for logging

    Returns:
        The content string (possibly repaired if json-repair was needed)

    Raises:
        json.JSONDecodeError: If JSON parsing fails (unfixable - should failover)
        JsonSchemaError: If JSON parses but fails schema validation (fixable by LLM)
    """
    from ._exceptions import JsonSchemaError

    if api_error:
        # Force JSON validation to fail so it triggers retry logic
        raise json.JSONDecodeError(f"API JSON validation failed: {str(api_error)}", "", 0)
    else:
        # First parse the JSON (with repair if needed)
        # This raises json.JSONDecodeError if parsing fails completely
        parsed_json, repaired_content, was_repaired = parse_json_with_repair(content, request_id)

        # Then validate against schema if provided
        if json_schema:
            is_valid, schema_error = validate_json_schema(parsed_json, json_schema)
            if not is_valid:
                # JSON parsed OK but doesn't match schema - this is fixable
                raise JsonSchemaError(schema_error, content=repaired_content)

        # Return the (possibly repaired) content
        return repaired_content