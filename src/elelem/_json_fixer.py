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

OUTPUT FORMAT:
Return a JSON object with exactly two keys:
- "changes": Brief description of what you fixed (1 sentence max)
- "fixed": The complete fixed JSON

Example: {"changes": "Added missing 'id' field at path x.y", "fixed": {...}}"""


def build_fixer_messages(invalid_json: str, error: str, schema: Dict[str, Any] = None) -> list:
    """Build the messages for the JSON fixer LLM.

    Args:
        invalid_json: The JSON string that failed validation
        error: The validation error message
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
{schema_section}

INVALID JSON:
{invalid_json}"""

    return [
        {"role": "system", "content": FIXER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def extract_json_from_response(response_content: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract JSON and changes description from fixer response.

    Args:
        response_content: The raw response from the fixer model

    Returns:
        Tuple of (fixed_json_string, changes_description)
    """
    fixed_str = response_content.strip()

    # Clean markdown code blocks if present
    if fixed_str.startswith('```'):
        lines = fixed_str.split('\n')
        if lines[-1].strip() == '```':
            fixed_str = '\n'.join(lines[1:-1])
        else:
            fixed_str = '\n'.join(lines[1:])

    # Find JSON boundaries
    start = fixed_str.find('{')
    end = fixed_str.rfind('}') + 1

    if start < 0 or end <= start:
        return None, None

    json_str = fixed_str[start:end]

    # Try to parse as wrapper format {"changes": "...", "fixed": {...}}
    try:
        wrapper = json.loads(json_str)
        if isinstance(wrapper, dict) and "fixed" in wrapper:
            changes = wrapper.get("changes", "")
            fixed_json = json.dumps(wrapper["fixed"])
            return fixed_json, changes
    except json.JSONDecodeError:
        pass

    # Fallback: treat entire response as fixed JSON (old format)
    return json_str, None


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
    schema: Dict[str, Any],
    request_id: str,
    fixer_model: str = None,
    max_iterations: int = 2
) -> Optional[str]:
    """Call the fixer LLM to repair invalid JSON.

    This is the main entry point for fixing JSON. It:
    1. Builds the fixer prompt with error, schema, and invalid JSON only
    2. Calls the fixer model via the Elelem instance
    3. Extracts and validates the fixed JSON
    4. If still invalid, loops with new error (up to max_iterations)
    5. Returns the fixed JSON string if successful

    Args:
        elelem_instance: The Elelem instance to use for the API call
        invalid_json: The JSON string that failed validation
        error: The validation error message
        schema: The JSON schema the output must conform to
        request_id: The original request ID for logging
        fixer_model: Optional override for the fixer model
        max_iterations: Maximum fixer attempts (default 2)

    Returns:
        The fixed JSON string if successful, None otherwise
    """
    if not schema:
        logger.debug(f"[{request_id}] JSON fixer skipped - no schema available")
        return None

    model = fixer_model or DEFAULT_FIXER_MODEL
    current_json = invalid_json
    current_error = error

    for iteration in range(max_iterations):
        iter_label = f" (attempt {iteration + 1}/{max_iterations})" if max_iterations > 1 else ""
        logger.info(f"[{request_id}] 🔧 Attempting JSON fix with {model}{iter_label}")

        try:
            # Build the fixer messages - only error, schema, and invalid JSON (no original prompt)
            fixer_messages = build_fixer_messages(current_json, current_error, schema)

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

            # Extract JSON and changes description from response
            fixed_json, changes = extract_json_from_response(response_content)
            if not fixed_json:
                logger.warning(f"[{request_id}] JSON fixer response did not contain valid JSON boundaries")
                return None

            # Validate the fixed JSON against the schema
            is_valid, validation_error, _ = validate_fixed_json(fixed_json, schema)

            if is_valid:
                # Log what was fixed
                if changes:
                    logger.info(f"[{request_id}] ✅ JSON fixer: {changes}")
                else:
                    # Fallback: extract path from original error
                    error_path = ""
                    if "at path:" in current_error:
                        error_path = current_error.split("at path:")[1].split("|")[0].strip()
                    logger.info(f"[{request_id}] ✅ JSON fixer repaired: {error_path or 'schema structure'}")
                return fixed_json
            else:
                # Still invalid - prepare for next iteration
                if changes:
                    logger.info(f"[{request_id}] 🔧 JSON fixer partial: {changes}")
                logger.warning(f"[{request_id}] JSON fixer output still invalid: {validation_error}")
                current_json = fixed_json
                current_error = validation_error

        except Exception as e:
            logger.warning(f"[{request_id}] JSON fixer failed with error: {e}")
            return None

    logger.warning(f"[{request_id}] JSON fixer exhausted {max_iterations} iterations")
    return None
