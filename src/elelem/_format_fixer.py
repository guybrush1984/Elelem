"""Generic format fixer for Elelem.

Works with any OutputFormat implementation (JSON, YAML, CSV).
Replaces the JSON-specific _json_fixer.py.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from ._output_formats import OutputFormat

if TYPE_CHECKING:
    from .elelem import Elelem

logger = logging.getLogger("elelem")

DEFAULT_FIXER_MODEL = "cerebras:openai/gpt-oss-120b?reasoning=medium"


@dataclass
class FixerOutput:
    """Result from the format fixer.

    Attributes:
        content: Fixed content string, or None if fix failed
        is_fixable: True if content was fixable, False if too incomplete to repair.
                    When False, caller should failover to next provider.
    """

    content: Optional[str]
    is_fixable: bool


async def call_format_fixer(
    elelem_instance: "Elelem",
    format_handler: OutputFormat,
    invalid_content: str,
    error: str,
    schema: Dict[str, Any],
    request_id: str,
    fixer_model: str = None,
    max_iterations: int = 2,
) -> FixerOutput:
    """Call fixer LLM to repair invalid output.

    This is a generic fixer that works with any output format by delegating
    format-specific logic to the OutputFormat handler.

    Args:
        elelem_instance: Elelem instance for API calls
        format_handler: The OutputFormat instance (json, yaml, csv)
        invalid_content: Content that failed validation
        error: Validation error message
        schema: Schema to validate against
        request_id: For logging
        fixer_model: Override fixer model (default: cerebras gpt-oss-120b)
        max_iterations: Max fix attempts (default: 2)

    Returns:
        FixerOutput with:
        - content: Fixed content string if successful, None otherwise
        - is_fixable: False if content was too incomplete to repair (triggers failover)
    """
    if not schema:
        logger.debug(f"[{request_id}] Fixer skipped - no schema available")
        return FixerOutput(content=None, is_fixable=True)

    model = fixer_model or DEFAULT_FIXER_MODEL
    current_content = invalid_content
    current_error = error
    format_name = format_handler.name.upper()

    for iteration in range(max_iterations):
        iter_label = (
            f" (attempt {iteration + 1}/{max_iterations})" if max_iterations > 1 else ""
        )
        logger.info(
            f"[{request_id}] 🔧 Attempting {format_name} fix with {model}{iter_label}"
        )

        try:
            # Get format-specific fixer messages
            fixer_messages = format_handler.get_fixer_messages(
                current_content, current_error, schema
            )

            # Call fixer model (no schema validation to avoid recursion)
            response = await elelem_instance.create_chat_completion(
                model=model,
                messages=fixer_messages,
                temperature=0.2,
            )

            response_content = response.choices[0].message.content
            if not response_content:
                logger.warning(f"[{request_id}] {format_name} fixer returned empty response")
                return FixerOutput(content=None, is_fixable=True)

            # Extract fixed content using format-specific logic
            fixer_result = format_handler.extract_fixer_result(response_content)

            # Check if fixer reported content as unfixable (too incomplete)
            if not fixer_result.is_fixable:
                logger.warning(
                    f"[{request_id}] {format_name} fixer: unfixable - {fixer_result.changes}"
                )
                return FixerOutput(content=None, is_fixable=False)

            if not fixer_result.content:
                logger.warning(
                    f"[{request_id}] Could not extract fixed content from fixer response"
                )
                return FixerOutput(content=None, is_fixable=True)

            # Validate the fix
            parse_result = format_handler.parse(fixer_result.content)
            if not parse_result.success:
                logger.warning(
                    f"[{request_id}] Fixer output failed to parse: {parse_result.error}"
                )
                current_content = fixer_result.content
                current_error = f"Parse error: {parse_result.error}"
                continue

            validation = format_handler.validate_schema(parse_result.data, schema)
            if validation.is_valid:
                # Success!
                if fixer_result.changes:
                    logger.info(f"[{request_id}] ✅ {format_name} fixer: {fixer_result.changes}")
                else:
                    error_path = validation.error_path or "structure"
                    logger.info(
                        f"[{request_id}] ✅ {format_name} fixer repaired: {error_path}"
                    )
                return FixerOutput(content=parse_result.content, is_fixable=True)
            else:
                # Still invalid - prepare for next iteration
                if fixer_result.changes:
                    logger.info(
                        f"[{request_id}] 🔧 {format_name} fixer partial: {fixer_result.changes}"
                    )
                logger.warning(f"[{request_id}] Fix still invalid: {validation.error}")
                current_content = parse_result.content
                current_error = validation.error

        except Exception as e:
            logger.warning(f"[{request_id}] {format_name} fixer failed with error: {e}")
            return FixerOutput(content=None, is_fixable=True)

    logger.warning(
        f"[{request_id}] {format_name} fixer exhausted {max_iterations} iterations"
    )
    return FixerOutput(content=None, is_fixable=True)
