"""
Response processing functions for Elelem.
"""

import re
import logging
from typing import Any, Tuple, Optional, List

from ._exceptions import ModelError, InfrastructureError


def extract_text_from_content_parts(content: list, logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract reasoning and response text from structured content parts.

    Returns:
        (reasoning_text, response_text) tuple where either can be None
    """
    reasoning_parts = []
    response_parts = []

    for part in content:
        if isinstance(part, dict):
            part_type = part.get('type', '')

            # Extract text from the part based on its structure
            text_content = None
            if 'text' in part:
                text_content = part['text']
            elif 'thinking' in part:
                # Nested thinking/reasoning structure
                thinking_items = part.get('thinking', [])
                if isinstance(thinking_items, list):
                    texts = [item.get('text', '') for item in thinking_items if isinstance(item, dict)]
                    text_content = '\n'.join(texts)
                elif isinstance(thinking_items, str):
                    text_content = thinking_items

            # Categorize based on type field
            if text_content:
                if part_type in ['thinking', 'reasoning']:
                    reasoning_parts.append(text_content)
                else:
                    response_parts.append(text_content)
        elif isinstance(part, str):
            response_parts.append(part)

    reasoning_text = '\n'.join(reasoning_parts) if reasoning_parts else None
    response_text = '\n'.join(response_parts) if response_parts else None

    if logger and reasoning_text:
        logger.debug(f"Extracted {len(reasoning_parts)} reasoning part(s), {len(response_parts)} response part(s)")

    return reasoning_text, response_text


class ChunkTimeoutError(Exception):
    """Raised when no streaming chunk is received within the chunk timeout."""
    pass


async def collect_streaming_response(stream, logger=None, request_id=None, chunk_timeout=None):
    """Collect streaming chunks and reconstruct a normal response object.

    Args:
        stream: Async iterator of streaming chunks
        logger: Optional logger for debug output
        request_id: Optional request ID for log prefix
        chunk_timeout: Optional timeout (seconds) for receiving each chunk.
                       If set and no chunk arrives within this time, raises ChunkTimeoutError.
                       This helps detect serverless cold starts and stream stalls.
    """
    import time
    import json
    import asyncio

    content_parts = []
    reasoning_content_parts = []
    final_chunk = None
    finish_reason = None

    # Trace variables
    chunk_count = 0
    first_chunk_time = None
    last_chunk_time = None
    last_trace_time = None
    trace_interval = 30  # 30 seconds
    long_gap_threshold = 15  # Warn if gap between chunks exceeds this

    # Convert stream to async iterator for manual iteration with timeout
    stream_iter = stream.__aiter__()

    while True:
        try:
            if chunk_timeout and chunk_timeout > 0:
                chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=chunk_timeout)
            else:
                chunk = await stream_iter.__anext__()
        except asyncio.TimeoutError:
            timeout_msg = f"No chunk received within {chunk_timeout}s"
            if chunk_count == 0:
                timeout_msg += " (cold start?)"
            else:
                timeout_msg += f" after {chunk_count} chunks"
            raise ChunkTimeoutError(timeout_msg)
        except StopAsyncIteration:
            break  # Stream ended normally
        chunk_count += 1
        current_time = time.time()

        # Warn if large gap between chunks (potential provider stall)
        if last_chunk_time and logger:
            gap = current_time - last_chunk_time
            if gap >= long_gap_threshold:
                request_prefix = f"[{request_id}] " if request_id else ""
                logger.warning(f"{request_prefix}⏳ Long inter-chunk gap: {gap:.1f}s before chunk {chunk_count}")

        # Log first chunk received
        if chunk_count == 1:
            first_chunk_time = current_time
            if logger:
                request_prefix = f"[{request_id}] " if request_id else ""
                logger.info(f"{request_prefix}🎬 First streaming chunk received")

        # Log periodic traces every 30s (skip first interval - already logged "First chunk received")
        if logger and last_trace_time is not None and current_time - last_trace_time >= trace_interval:
            elapsed = current_time - first_chunk_time if first_chunk_time else 0
            request_prefix = f"[{request_id}] " if request_id else ""

            # Estimate token rate (~3.5 chars/token heuristic, avoids tiktoken CPU overhead)
            total_chars = sum(len(p) for p in content_parts) + sum(len(p) for p in reasoning_content_parts)
            estimated_tokens = total_chars / 3.5
            token_rate = estimated_tokens / elapsed if elapsed > 0 else 0

            logger.info(f"{request_prefix}📊 Streaming: {chunk_count} chunks, ~{estimated_tokens:.0f} tokens (~{token_rate:.0f} tok/s, {elapsed:.1f}s)")
            last_trace_time = current_time
        elif last_trace_time is None:
            # Initialize trace timer on first chunk (but don't log)
            last_trace_time = current_time

        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]

            # Debug: log first chunk structure to understand provider format
            if chunk_count == 1 and logger:
                request_prefix = f"[{request_id}] " if request_id else ""
                delta = getattr(choice, 'delta', None)
                if delta:
                    delta_fields = {k: type(v).__name__ for k, v in vars(delta).items() if not k.startswith('_') and v is not None}
                    logger.debug(f"{request_prefix}🔍 First chunk delta fields: {delta_fields}")

            # Extract content from delta
            if hasattr(choice, 'delta') and choice.delta.content is not None:
                content_parts.append(choice.delta.content)

            # Extract reasoning content if available (OpenRouter uses 'reasoning' field, not 'reasoning_content')
            if hasattr(choice, 'delta') and hasattr(choice.delta, 'reasoning') and choice.delta.reasoning is not None:
                reasoning_content_parts.append(choice.delta.reasoning)
            elif hasattr(choice, 'delta') and hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content is not None:
                reasoning_content_parts.append(choice.delta.reasoning_content)

            # Capture finish_reason
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                finish_reason = choice.finish_reason

        # Keep the last chunk for metadata (id, usage, etc.)
        final_chunk = chunk
        last_chunk_time = current_time

    if not final_chunk:
        raise ValueError("No chunks received from stream")

    # Reconstruct the complete content
    full_content = ''.join(content_parts) if content_parts else None
    full_reasoning_content = ''.join(reasoning_content_parts) if reasoning_content_parts else None

    # Debug: log what we captured
    if logger:
        request_prefix = f"[{request_id}] " if request_id else ""
        logger.debug(f"{request_prefix}📝 Streaming complete: content_parts={len(content_parts)}, reasoning_parts={len(reasoning_content_parts)}, content_len={len(full_content) if full_content else 0}, reasoning_len={len(full_reasoning_content) if full_reasoning_content else 0}")


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

    return completion, chunk_count


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


def extract_yaml_from_markdown(content: str, logger: logging.Logger) -> str:
    """Extract YAML from markdown code blocks."""
    # Pattern to match ```yaml\n...\n``` or ```yml\n...\n```
    patterns = [
        r'```yaml\s*\n(.*?)\n```',
        r'```yml\s*\n(.*?)\n```',
        r'```yaml\s*\n(.*?)```',
        r'```yml\s*\n(.*?)```'
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.debug(f"Extracted YAML from markdown: {extracted}")
            return extracted

    # If no markdown wrapper found, return original content
    return content


def process_response_content(response: Any, json_mode_requested: bool, yaml_mode_requested: bool, logger: logging.Logger) -> str:
    """Process and clean response content."""
    # Check finish_reason first - this should always be present
    finish_reason = getattr(response.choices[0], 'finish_reason', None)
    if finish_reason != 'stop':
        if finish_reason == 'length':
            # Truncation is an infra issue - another provider with higher max_tokens may succeed
            raise InfrastructureError(f"Response was truncated due to max_tokens limit (finish_reason: {finish_reason})")
        elif finish_reason == 'content_filter':
            raise ModelError(f"Response was filtered due to safety policies (finish_reason: {finish_reason})")
        elif finish_reason in ['function_call', 'tool_calls']:
            # Tool calls in non-function context - try another provider
            raise InfrastructureError(f"Unexpected function/tool call in non-function context (finish_reason: {finish_reason})")
        else:
            raise ModelError(f"Unexpected finish_reason: {finish_reason}")

    content = response.choices[0].message.content

    # Handle None content (some providers may return None for empty responses)
    if content is None:
        logger.warning("Response content is None despite finish_reason being 'stop'")
        content = ""

    # Handle content as list (structured content parts) or string
    if isinstance(content, list):
        logger.debug(f"Content is a list with {len(content)} parts")
        _, response_text = extract_text_from_content_parts(content, logger)
        content = response_text or ""

    # Remove think tags if present
    content = remove_think_tags(content, logger)

    # Extract JSON from markdown if JSON mode was requested
    if json_mode_requested:
        content = extract_json_from_markdown(content, logger)

    # Extract YAML from markdown if YAML mode was requested
    if yaml_mode_requested:
        content = extract_yaml_from_markdown(content, logger)

    return content