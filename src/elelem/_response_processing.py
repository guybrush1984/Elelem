"""
Response processing functions for Elelem.
"""

import re
import logging
from typing import Any

from ._exceptions import ModelError


async def collect_streaming_response(stream, logger=None, request_id=None):
    """Collect streaming chunks and reconstruct a normal response object."""
    import time

    content_parts = []
    reasoning_content_parts = []
    final_chunk = None
    finish_reason = None

    # Trace variables
    chunk_count = 0
    first_chunk_time = None
    last_trace_time = None
    trace_interval = 30  # 30 seconds

    async for chunk in stream:
        chunk_count += 1
        current_time = time.time()

        # Log first chunk received
        if chunk_count == 1:
            first_chunk_time = current_time
            if logger:
                request_prefix = f"[{request_id}] " if request_id else ""
                logger.info(f"{request_prefix}🎬 First streaming chunk received")

        # Log periodic traces every 30s
        if logger and (last_trace_time is None or current_time - last_trace_time >= trace_interval):
            elapsed = current_time - first_chunk_time if first_chunk_time else 0
            request_prefix = f"[{request_id}] " if request_id else ""

            logger.info(f"{request_prefix}📊 Streaming progress: {chunk_count} chunks received ({elapsed:.1f}s elapsed)")
            last_trace_time = current_time

        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]

            # Extract content from delta
            if hasattr(choice, 'delta') and choice.delta.content is not None:
                content_parts.append(choice.delta.content)

            # Extract reasoning_content if available
            if hasattr(choice, 'delta') and hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content is not None:
                reasoning_content_parts.append(choice.delta.reasoning_content)

            # Capture finish_reason
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                finish_reason = choice.finish_reason

        # Keep the last chunk for metadata (id, usage, etc.)
        final_chunk = chunk

    if not final_chunk:
        raise ValueError("No chunks received from stream")

    # Reconstruct the complete content
    full_content = ''.join(content_parts) if content_parts else None
    full_reasoning_content = ''.join(reasoning_content_parts) if reasoning_content_parts else None

    # Create a normal ChatCompletion response (not streaming)
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion import ChatCompletion, Choice

    # Create the message object
    message = ChatCompletionMessage(
        role="assistant",
        content=full_content,
        reasoning_content=full_reasoning_content if full_reasoning_content else None
    )

    # Create the choice object
    choice = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason
    )

    # Create a proper ChatCompletion object (not ChatCompletionChunk)
    # Use the metadata from the final chunk but create a new ChatCompletion
    completion = ChatCompletion(
        id=final_chunk.id,
        object="chat.completion",  # Not "chat.completion.chunk"
        created=final_chunk.created,
        model=final_chunk.model,
        choices=[choice],
        usage=final_chunk.usage if hasattr(final_chunk, 'usage') and final_chunk.usage else None,
        system_fingerprint=final_chunk.system_fingerprint if hasattr(final_chunk, 'system_fingerprint') else None,
        service_tier=final_chunk.service_tier if hasattr(final_chunk, 'service_tier') else None
    )

    return completion


def remove_think_tags(content: str, logger: logging.Logger) -> str:
    """Remove <think>...</think> tags from content."""
    if not content:
        return ""

    # Pattern to match <think>...</think> including multiline content
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL).strip()

    # Fallback: if content starts with thinking content and contains </think>,
    # extract everything after </think>
    if cleaned == content and '</think>' in content:
        parts = content.split('</think>', 1)
        if len(parts) > 1:
            cleaned = parts[1].strip()
            logger.debug(f"Removed thinking content using </think> split: kept {len(cleaned)} chars")

    return cleaned


def extract_json_from_markdown(content: str, logger: logging.Logger) -> str:
    """Extract JSON from markdown code blocks."""
    # Pattern to match ```json\n{...}\n``` or ```\n{...}\n```
    patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(\{.*?\})\n```',
        r'```json\s*\n(.*?)```',
        r'```\s*\n(\{.*?\})```'
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.debug(f"Extracted JSON from markdown: {extracted}")
            return extracted

    # If no markdown wrapper found, return original content
    return content


def process_response_content(response: Any, json_mode_requested: bool, logger: logging.Logger) -> str:
    """Process and clean response content."""
    # Debug logging to understand response structure
    if hasattr(response, 'model_dump_json'):
        logger.debug(f"Full response JSON: {response.model_dump_json()[:500]}")

    if hasattr(response, 'choices') and response.choices:
        first_choice = response.choices[0]
        logger.debug(f"First choice type: {type(first_choice)}")
        if hasattr(first_choice, 'message'):
            logger.debug(f"Message type: {type(first_choice.message)}")
            logger.debug(f"Message content: {first_choice.message.content}")
            if hasattr(first_choice.message, 'model_dump_json'):
                logger.debug(f"Message JSON: {first_choice.message.model_dump_json()[:500]}")

    # Check finish_reason first - this should always be present
    finish_reason = getattr(response.choices[0], 'finish_reason', None)
    if finish_reason != 'stop':
        if finish_reason == 'length':
            raise ModelError(f"Response was truncated due to max_tokens limit (finish_reason: {finish_reason})")
        elif finish_reason == 'content_filter':
            raise ModelError(f"Response was filtered due to safety policies (finish_reason: {finish_reason})")
        elif finish_reason in ['function_call', 'tool_calls']:
            raise ModelError(f"Unexpected function/tool call in non-function context (finish_reason: {finish_reason})")
        else:
            raise ModelError(f"Unexpected finish_reason: {finish_reason}")

    content = response.choices[0].message.content

    # Handle None content (some providers may return None for empty responses)
    if content is None:
        logger.warning("Response content is None despite finish_reason being 'stop'")
        content = ""

    # Remove think tags if present
    content = remove_think_tags(content, logger)

    # Extract JSON from markdown if JSON mode was requested
    if json_mode_requested:
        content = extract_json_from_markdown(content, logger)

    return content