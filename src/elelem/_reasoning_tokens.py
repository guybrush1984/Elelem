"""
Reasoning token extraction and analysis for all LLM providers.

This module handles the complexity of extracting reasoning tokens across different
provider implementations, each with their own token reporting structure.
"""

import re
from typing import Any, Optional, Set, Tuple
import logging


def extract_token_counts(response: Any, logger: Optional[logging.Logger] = None) -> Tuple[int, int, int, int]:
    """
    Extract complete normalized token structure for all providers.

    Returns consistent token semantics across all providers:
    - input: prompt tokens (what you send)
    - output: completion tokens (what you get back, including reasoning)
    - reasoning: reasoning tokens (portion of output that's internal reasoning)
    - total: input + output tokens (standard total across all providers)

    This ensures reasoning_token_ratio = reasoning / output is always meaningful.

    Args:
        response: LLM response object with usage and choices
        logger: Optional logger for debug output

    Returns:
        Tuple[int, int, int, int]: (input_tokens, output_tokens, reasoning_tokens, total_tokens)
    """
    if not response or not hasattr(response, 'usage'):
        return 0, 0, 0, 0

    if logger is None:
        logger = logging.getLogger(__name__)

    usage = response.usage

    # Extract raw token values
    input_tokens = getattr(usage, 'prompt_tokens', 0)
    raw_output_tokens = getattr(usage, 'completion_tokens', 0)
    raw_total_tokens = getattr(usage, 'total_tokens', 0)

    # Extract reasoning tokens using existing sophisticated logic
    reasoning_tokens = extract_reasoning_tokens(response, logger)

    # Normalize token structure for consistency across providers
    if is_gemini_style_response(response):
        # Gemini: completion_tokens is pure output, reasoning is separate
        # Normalize: output should include reasoning to match other providers
        output_tokens = raw_output_tokens + reasoning_tokens
        total_tokens = input_tokens + output_tokens

        logger.debug(f"Normalized Gemini tokens: raw_output={raw_output_tokens}, "
                    f"reasoning={reasoning_tokens}, normalized_output={output_tokens}, total={total_tokens}")
    else:
        # Standard providers: completion_tokens already includes reasoning
        output_tokens = raw_output_tokens
        total_tokens = input_tokens + output_tokens

        logger.debug(f"Standard tokens: input={input_tokens}, output={output_tokens}, "
                    f"reasoning={reasoning_tokens}, total={total_tokens}")

    return input_tokens, output_tokens, reasoning_tokens, total_tokens


def extract_reasoning_tokens(response: Any, logger: Optional[logging.Logger] = None) -> int:
    """
    Extract reasoning tokens with correct priority order.

    Priority order:
    1. Explicit reasoning_tokens field (recursive search in usage object)
    2. Content analysis (<think> tags, reasoning fields)
    3. Mathematical fallback for Gemini-style responses
    4. No reasoning (return 0)

    Args:
        response: LLM response object with usage and choices
        logger: Optional logger for debug output

    Returns:
        int: Number of reasoning tokens found
    """
    if not response or not hasattr(response, 'usage'):
        return 0

    if logger is None:
        logger = logging.getLogger(__name__)

    # Priority 1: Explicit reasoning_tokens field
    explicit_reasoning = recursive_search_reasoning_tokens(response.usage)
    if explicit_reasoning > 0:
        logger.debug(f"Found explicit reasoning tokens: {explicit_reasoning}")
        return explicit_reasoning

    # Priority 2: Content analysis (<think> tags, reasoning fields)
    content_reasoning, _ = extract_reasoning_from_content(response, logger)
    if content_reasoning > 0:
        return content_reasoning

    # Priority 3: Mathematical fallback (Gemini case)
    if is_gemini_style_response(response):
        gemini_reasoning = calculate_gemini_reasoning(response, logger)
        if gemini_reasoning > 0:
            return gemini_reasoning

    # Priority 4: No reasoning
    return 0


def recursive_search_reasoning_tokens(obj: Any, target_field: str = "reasoning_tokens",
                                    visited: Optional[Set] = None, depth: int = 0) -> int:
    """
    Recursively search for explicit reasoning_tokens field in nested objects/dicts.

    Includes cycle detection and depth limiting for safety.
    """
    if visited is None:
        visited = set()

    # Prevent infinite recursion
    if depth > 10 or obj is None or id(obj) in visited:
        return 0

    visited.add(id(obj))

    # If it's a dict, check keys
    if isinstance(obj, dict):
        if target_field in obj:
            value = obj[target_field]
            return int(value) if isinstance(value, (int, float)) and value > 0 else 0
        # Recursively search nested dicts
        for value in obj.values():
            result = recursive_search_reasoning_tokens(value, target_field, visited, depth + 1)
            if result > 0:
                return result

    # If it's an object with attributes, check attributes
    elif hasattr(obj, '__dict__') or hasattr(obj, '__getattribute__'):
        # Direct attribute check
        if hasattr(obj, target_field):
            value = getattr(obj, target_field, 0)
            return int(value) if isinstance(value, (int, float)) and value > 0 else 0

        # Search nested attributes (only common usage-related attributes)
        common_attrs = ['completion_tokens_details', 'output_tokens_details',
                       'usage_details', 'token_details', 'details']
        for attr_name in common_attrs:
            if hasattr(obj, attr_name):
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):
                        result = recursive_search_reasoning_tokens(attr_value, target_field, visited, depth + 1)
                        if result > 0:
                            return result
                except (AttributeError, TypeError):
                    continue

    return 0


def extract_reasoning_from_content(response: Any, logger: Optional[logging.Logger] = None) -> Tuple[int, Optional[str]]:
    """
    Extract reasoning tokens and content from content analysis (for models with <think> tags or reasoning fields).

    Returns tuple of (reasoning_tokens, reasoning_content).

    Handles multiple edge cases:
    - Complete <think>content</think> tags
    - Missing opening <think> tag (malformed content)
    - Multiple reasoning field names in message object
    - Character-based token estimation using ratios
    """
    if not response or not hasattr(response, 'usage'):
        return 0, None

    if logger is None:
        logger = logging.getLogger(__name__)

    usage = response.usage

    # Check for <think> tags in response content (Parasail DeepSeek, etc.)
    if hasattr(response, 'choices') and response.choices:
        first_choice = response.choices[0]
        if hasattr(first_choice, 'message') and first_choice.message.content:
            content = first_choice.message.content
            logger.debug(f"Checking content for <think> tags, length: {len(content)}")

            # Extract reasoning content from <think> tags
            think_pattern = r'<think>(.*?)</think>'
            think_matches = re.findall(think_pattern, content, re.DOTALL)

            logger.debug(f"Found {len(think_matches)} <think> matches")

            if think_matches:
                reasoning_content = think_matches[0].strip()
                if reasoning_content:
                    # Estimate reasoning tokens using character count ratio
                    reasoning_chars = len(reasoning_content)

                    # Get actual response content (everything after </think>)
                    actual_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    actual_chars = len(actual_content)
                    total_chars = reasoning_chars + actual_chars

                    logger.debug(f"Reasoning chars: {reasoning_chars}, Actual chars: {actual_chars}")

                    if total_chars > 0:
                        reasoning_ratio = reasoning_chars / total_chars
                        total_completion_tokens = getattr(usage, 'completion_tokens', 0)
                        estimated_reasoning_tokens = int(total_completion_tokens * reasoning_ratio)

                        logger.debug(f"Extracted reasoning from <think> tags: {estimated_reasoning_tokens} tokens "
                                   f"({reasoning_ratio:.1%} of {total_completion_tokens} total)")
                        return estimated_reasoning_tokens, reasoning_content

            # Fallback for content that starts with thinking but no opening <think> tag
            elif '</think>' in content:
                logger.debug("Found </think> but no opening <think> tag, using fallback")
                parts = content.split('</think>', 1)
                if len(parts) > 1:
                    reasoning_content = parts[0].strip()
                    actual_content = parts[1].strip()

                    reasoning_chars = len(reasoning_content)
                    actual_chars = len(actual_content)
                    total_chars = reasoning_chars + actual_chars

                    logger.debug(f"Fallback - Reasoning chars: {reasoning_chars}, Actual chars: {actual_chars}")

                    if total_chars > 0 and reasoning_chars > 0:
                        reasoning_ratio = reasoning_chars / total_chars
                        total_completion_tokens = getattr(usage, 'completion_tokens', 0)
                        estimated_reasoning_tokens = int(total_completion_tokens * reasoning_ratio)

                        logger.debug(f"Extracted reasoning via fallback: {estimated_reasoning_tokens} tokens "
                                   f"({reasoning_ratio:.1%} of {total_completion_tokens} total)")
                        return estimated_reasoning_tokens, reasoning_content

    # Fallback: estimate from reasoning content fields (any provider)
    if response and hasattr(response, 'choices') and response.choices:
        first_choice = response.choices[0]
        if hasattr(first_choice, 'message'):
            message = first_choice.message
            actual_content = getattr(message, 'content', '')

            # Look for reasoning content in common field names
            reasoning_fields = ['reasoning_content', 'reasoning', 'chain_of_thought', 'thinking', 'scratchpad']
            reasoning_content = ''

            for field in reasoning_fields:
                if hasattr(message, field):
                    reasoning_content = getattr(message, field, '') or ''
                    if reasoning_content:
                        break

            if reasoning_content and actual_content:
                # Calculate proportional token distribution based on character count
                reasoning_chars = len(reasoning_content)
                content_chars = len(actual_content)
                total_chars = reasoning_chars + content_chars

                if total_chars > 0:
                    reasoning_ratio = reasoning_chars / total_chars
                    total_completion_tokens = getattr(usage, 'completion_tokens', 0)
                    estimated_reasoning_tokens = int(total_completion_tokens * reasoning_ratio)

                    logger.debug(f"Estimated reasoning tokens via char ratio: {estimated_reasoning_tokens} "
                               f"({reasoning_ratio:.1%} of {total_completion_tokens} total)")
                    return estimated_reasoning_tokens, reasoning_content

    return 0, None


def extract_reasoning_content(response: Any, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Extract just the reasoning content text from response.

    Returns the reasoning content string if found, None otherwise.
    Uses the same logic as extract_reasoning_from_content but only returns the content.
    """
    _, content = extract_reasoning_from_content(response, logger)
    return content


def is_gemini_style_response(response: Any) -> bool:
    """
    Detect if this is a Gemini-style response where completion + prompt != total.

    Gemini reports reasoning tokens separately from completion_tokens, while
    other providers include reasoning in completion_tokens.
    """
    if not response or not hasattr(response, 'usage'):
        return False

    usage = response.usage
    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
    completion_tokens = getattr(usage, 'completion_tokens', 0)
    total_tokens = getattr(usage, 'total_tokens', 0)

    # If total != prompt + completion, then there are hidden reasoning tokens
    return total_tokens != (prompt_tokens + completion_tokens)


def calculate_gemini_reasoning(response: Any, logger: Optional[logging.Logger] = None) -> int:
    """
    Calculate reasoning tokens for Gemini-style responses using mathematical fallback.

    For Gemini: total_tokens = prompt_tokens + completion_tokens + hidden_reasoning
    """
    if not response or not hasattr(response, 'usage'):
        return 0

    if logger is None:
        logger = logging.getLogger(__name__)

    usage = response.usage
    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
    completion_tokens = getattr(usage, 'completion_tokens', 0)
    total_tokens = getattr(usage, 'total_tokens', 0)

    if total_tokens > 0 and prompt_tokens >= 0 and completion_tokens >= 0:
        calculated_reasoning = max(0, total_tokens - prompt_tokens - completion_tokens)
        if calculated_reasoning > 0:
            logger.debug(f"Calculated reasoning tokens from total: {calculated_reasoning} "
                        f"(total: {total_tokens}, prompt: {prompt_tokens}, completion: {completion_tokens})")
            return calculated_reasoning

    return 0