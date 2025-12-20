"""
LLM-based JSON fixer for schema validation failures.
Uses a fast, reliable model to repair invalid JSON after all retries are exhausted.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("elelem")

# Default fixer model - direct provider reference to avoid recursion
DEFAULT_FIXER_MODEL = "cerebras:openai/gpt-oss-120b?reasoning=medium"


FIXER_SYSTEM_PROMPT = """You are a JSON fixer. Your task is to repair invalid JSON so it passes schema validation.

INSTRUCTIONS:
1. Read the validation error carefully - it tells you exactly what is wrong and where
2. Parse the error path to locate the exact position of the problem in the JSON
3. Fix ONLY what the error describes - make minimal changes
4. If a required field is missing, add it with a value that fits the context
5. If a key has wrong type, fix the type or remove the invalid key
6. Return the complete fixed JSON only - no explanations, no markdown

Return ONLY valid JSON starting with { and ending with }."""


def build_fixer_messages(invalid_json: str, error: str, context: str, schema: Dict[str, Any] = None) -> list:
    """Build the messages for the JSON fixer LLM.

    Args:
        invalid_json: The JSON string that failed validation
        error: The validation error message
        context: Context from the original request (first few messages)
        schema: Optional JSON schema the output must conform to

    Returns:
        List of message dicts with system and user roles
    """
    schema_section = ""
    if schema:
        schema_section = f"\n\nEXPECTED SCHEMA:\n{json.dumps(schema, indent=2)}"

    user_content = f"""Fix this JSON that failed schema validation.

VALIDATION ERROR:
{error}

CONTEXT:
{context}{schema_section}

INVALID JSON:
{invalid_json}"""

    return [
        {"role": "system", "content": FIXER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def extract_json_from_response(response_content: str) -> Optional[str]:
    """Extract JSON from fixer response, handling markdown if present.

    Args:
        response_content: The raw response from the fixer model

    Returns:
        The extracted JSON string, or None if extraction fails
    """
    fixed_str = response_content.strip()

    # Clean markdown code blocks if present
    if fixed_str.startswith('```'):
        lines = fixed_str.split('\n')
        # Remove first line (```json or ```) and last line if it's just ```)
        if lines[-1].strip() == '```':
            fixed_str = '\n'.join(lines[1:-1])
        else:
            fixed_str = '\n'.join(lines[1:])

    # Find JSON boundaries
    start = fixed_str.find('{')
    end = fixed_str.rfind('}') + 1

    if start >= 0 and end > start:
        return fixed_str[start:end]

    return None


def validate_fixed_json(json_str: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Any]]:
    """Validate the fixed JSON against the schema.

    Args:
        json_str: The JSON string to validate
        schema: The JSON schema to validate against

    Returns:
        Tuple of (is_valid, error_message, parsed_json)
    """
    from jsonschema import validate, ValidationError

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None

    try:
        validate(instance=parsed, schema=schema)
        return True, None, parsed
    except ValidationError as e:
        error_parts = [e.message]
        if e.path:
            error_parts.append(f"at path: {'.'.join(str(p) for p in e.path)}")
        return False, " | ".join(error_parts), parsed


async def call_json_fixer(
    elelem_instance: Any,
    invalid_json: str,
    error: str,
    messages: list,
    schema: Dict[str, Any],
    request_id: str,
    fixer_model: str = None
) -> Optional[str]:
    """Call the fixer LLM to repair invalid JSON.

    This is the main entry point for fixing JSON. It:
    1. Builds the fixer prompt with context and schema
    2. Calls the fixer model via the Elelem instance
    3. Extracts and validates the fixed JSON
    4. Returns the fixed JSON string if successful

    Args:
        elelem_instance: The Elelem instance to use for the API call
        invalid_json: The JSON string that failed validation
        error: The validation error message
        messages: Original request messages (for context)
        schema: The JSON schema the output must conform to
        request_id: The original request ID for logging
        fixer_model: Optional override for the fixer model

    Returns:
        The fixed JSON string if successful, None otherwise
    """
    if not schema:
        logger.debug(f"[{request_id}] JSON fixer skipped - no schema available")
        return None

    model = fixer_model or DEFAULT_FIXER_MODEL

    # Build context from first 2 messages (usually system + user)
    context = "\n".join(msg.get('content', '') for msg in messages[:2])

    # Build the fixer messages (system + user)
    fixer_messages = build_fixer_messages(invalid_json, error, context, schema)

    logger.info(f"[{request_id}] 🔧 Attempting JSON fix with {model}")

    try:
        # Call the fixer model - use low temperature for consistency
        response = await elelem_instance.create_chat_completion(
            model=model,
            messages=fixer_messages,
            temperature=0.2,
            # Don't use JSON mode - we want raw text
            # Don't pass schema validation to avoid infinite recursion
        )

        response_content = response.choices[0].message.content
        if not response_content:
            logger.warning(f"[{request_id}] JSON fixer returned empty response")
            return None

        # Extract JSON from response
        fixed_json = extract_json_from_response(response_content)
        if not fixed_json:
            logger.warning(f"[{request_id}] JSON fixer response did not contain valid JSON boundaries")
            return None

        # Validate the fixed JSON against the schema
        is_valid, validation_error, _ = validate_fixed_json(fixed_json, schema)

        if is_valid:
            logger.info(f"[{request_id}] ✅ JSON fixer successfully repaired the response")
            return fixed_json
        else:
            logger.warning(f"[{request_id}] JSON fixer output still invalid: {validation_error}")
            return None

    except Exception as e:
        logger.warning(f"[{request_id}] JSON fixer failed with error: {e}")
        return None
