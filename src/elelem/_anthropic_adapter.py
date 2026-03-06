"""
Anthropic API adapter for Elelem.

Duck-type compatible with openai.AsyncOpenAI — translates between
Anthropic Messages API and OpenAI Chat Completions format so that
Elelem's request handlers and response processing work unchanged.
"""

import time
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("elelem.anthropic_adapter")

# ---------------------------------------------------------------------------
# Stop-reason mapping: Anthropic → OpenAI
# ---------------------------------------------------------------------------
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int) -> httpx.Response:
    """Create a minimal httpx.Response for OpenAI exception constructors."""
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


def _translate_messages(messages: List[Dict]) -> tuple:
    """Extract system messages and convert to Anthropic format.

    Returns:
        (system_text or None, anthropic_messages)
    """
    system_parts = []
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                # Handle structured content parts
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        system_parts.append(part["text"])
                    elif isinstance(part, str):
                        system_parts.append(part)
        else:
            anthropic_messages.append({"role": role, "content": content})

    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, anthropic_messages


def _translate_kwargs(kwargs: Dict) -> Dict:
    """Translate OpenAI-style kwargs to Anthropic parameters.

    Modifies kwargs in-place (pops consumed keys) and returns Anthropic kwargs.
    """
    anthropic_kwargs = {}

    # max_tokens is required by Anthropic
    anthropic_kwargs["max_tokens"] = kwargs.pop("max_tokens", 4096)

    # Pass through supported params
    if "temperature" in kwargs:
        anthropic_kwargs["temperature"] = kwargs.pop("temperature")

    if "stream" in kwargs:
        anthropic_kwargs["stream"] = kwargs.pop("stream")

    if "top_p" in kwargs:
        if "temperature" not in anthropic_kwargs:
            anthropic_kwargs["top_p"] = kwargs.pop("top_p")
        else:
            kwargs.pop("top_p")  # Anthropic forbids both temperature and top_p

    if "stop" in kwargs:
        anthropic_kwargs["stop_sequences"] = kwargs.pop("stop")

    # Handle extra_body — merge into top-level (for thinking config etc.)
    extra_body = kwargs.pop("extra_body", None)
    if extra_body and isinstance(extra_body, dict):
        anthropic_kwargs.update(extra_body)

    # Drop unsupported OpenAI-specific params silently
    for key in [
        "response_format", "reasoning_effort", "service_tier",
        "stream_options", "n", "frequency_penalty",
        "presence_penalty", "logprobs", "top_logprobs",
        "seed", "logit_bias", "user", "tools", "tool_choice",
    ]:
        kwargs.pop(key, None)

    # Warn about remaining unknown kwargs
    if kwargs:
        logger.debug(f"Dropping unknown kwargs for Anthropic: {list(kwargs.keys())}")

    return anthropic_kwargs


def _translate_response(anthropic_response) -> Any:
    """Translate Anthropic Message → OpenAI ChatCompletion."""
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.completion_usage import CompletionUsage

    # Extract text and thinking content from content blocks
    text_parts = []
    thinking_parts = []

    for block in anthropic_response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)

    content = "".join(text_parts) if text_parts else None
    reasoning_content = "".join(thinking_parts) if thinking_parts else None

    finish_reason = _STOP_REASON_MAP.get(anthropic_response.stop_reason, "stop")

    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        reasoning_content=reasoning_content,
    )

    choice = Choice(index=0, message=message, finish_reason=finish_reason)

    usage = CompletionUsage(
        prompt_tokens=anthropic_response.usage.input_tokens,
        completion_tokens=anthropic_response.usage.output_tokens,
        total_tokens=(
            anthropic_response.usage.input_tokens
            + anthropic_response.usage.output_tokens
        ),
    )

    return ChatCompletion(
        id=anthropic_response.id,
        object="chat.completion",
        created=int(time.time()),
        model=anthropic_response.model,
        choices=[choice],
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Streaming translator — async generator yielding ChatCompletionChunk
# ---------------------------------------------------------------------------

class _StreamAdapter:
    """Async iterable that yields OpenAI ChatCompletionChunk from Anthropic stream."""

    def __init__(self, anthropic_stream):
        self._stream = anthropic_stream
        self._message_id: Optional[str] = None
        self._model: Optional[str] = None
        self._input_tokens: int = 0

    def __aiter__(self):
        return self._translate()

    async def _translate(self):
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice as ChunkChoice,
            ChoiceDelta,
        )
        from openai.types.completion_usage import CompletionUsage

        async for event in self._stream:
            event_type = event.type

            if event_type == "message_start":
                self._message_id = event.message.id
                self._model = event.message.model
                self._input_tokens = event.message.usage.input_tokens
                # Yield initial empty chunk (sets up id/model for collector)
                yield ChatCompletionChunk(
                    id=self._message_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self._model,
                    choices=[ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(role="assistant"),
                        finish_reason=None,
                    )],
                )

            elif event_type == "content_block_delta":
                delta_type = event.delta.type

                if delta_type == "text_delta":
                    yield ChatCompletionChunk(
                        id=self._message_id or "",
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=self._model or "",
                        choices=[ChunkChoice(
                            index=0,
                            delta=ChoiceDelta(content=event.delta.text),
                            finish_reason=None,
                        )],
                    )

                elif delta_type == "thinking_delta":
                    # Map to reasoning field (same as OpenRouter convention)
                    delta = ChoiceDelta(content=None)
                    # Set reasoning via attribute since it may not be
                    # in the constructor for all openai SDK versions
                    delta.reasoning = event.delta.thinking
                    yield ChatCompletionChunk(
                        id=self._message_id or "",
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=self._model or "",
                        choices=[ChunkChoice(
                            index=0,
                            delta=delta,
                            finish_reason=None,
                        )],
                    )

                # Skip signature_delta, input_json_delta, etc.

            elif event_type == "message_delta":
                # Final chunk with stop_reason and usage
                finish_reason = _STOP_REASON_MAP.get(
                    event.delta.stop_reason, "stop"
                )
                output_tokens = event.usage.output_tokens

                usage = CompletionUsage(
                    prompt_tokens=self._input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=self._input_tokens + output_tokens,
                )

                yield ChatCompletionChunk(
                    id=self._message_id or "",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self._model or "",
                    choices=[ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(),
                        finish_reason=finish_reason,
                    )],
                    usage=usage,
                )

            # Skip: content_block_start, content_block_stop,
            #        message_stop, ping


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

def _translate_error(error: Exception) -> Exception:
    """Translate Anthropic SDK exception → OpenAI SDK exception."""
    from openai import (
        AuthenticationError as OAIAuthError,
        RateLimitError as OAIRateLimitError,
        BadRequestError as OAIBadRequestError,
        NotFoundError as OAINotFoundError,
        InternalServerError as OAIInternalServerError,
        APIConnectionError as OAIConnectionError,
    )

    # Lazy import to avoid top-level dependency
    import anthropic as anth

    status_code = getattr(error, "status_code", 500)
    msg = str(error)
    body = getattr(error, "body", None)
    resp = _mock_response(status_code)

    if isinstance(error, anth.AuthenticationError):
        return OAIAuthError(msg, response=resp, body=body)

    if isinstance(error, anth.RateLimitError):
        return OAIRateLimitError(msg, response=resp, body=body)

    if isinstance(error, anth.BadRequestError):
        return OAIBadRequestError(msg, response=resp, body=body)

    if isinstance(error, anth.NotFoundError):
        return OAINotFoundError(msg, response=resp, body=body)

    if isinstance(error, anth.InternalServerError):
        return OAIInternalServerError(msg, response=resp, body=body)

    if isinstance(error, anth.APIConnectionError):
        return OAIConnectionError(request=resp.request)

    if isinstance(error, anth.APIStatusError):
        return OAIInternalServerError(msg, response=resp, body=body)

    # Not an Anthropic SDK error — re-raise as-is
    return error


# ---------------------------------------------------------------------------
# Adapter classes — duck-type compatible with AsyncOpenAI
# ---------------------------------------------------------------------------

class _Completions:
    """Duck-type compatible with openai.AsyncOpenAI().chat.completions."""

    def __init__(self, adapter: "AnthropicAdapter"):
        self._adapter = adapter
        self._client = None

    def _get_client(self):
        """Lazy-init the Anthropic async client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for Anthropic providers. "
                    "Install it with: uv add anthropic"
                )
            self._client = anthropic.AsyncAnthropic(
                api_key=self._adapter.api_key,
            )

        # Propagate api_key changes (token refresh pattern from core.py:679)
        if self._client.api_key != self._adapter.api_key:
            self._client.api_key = self._adapter.api_key

        return self._client

    async def create(self, *, messages, model, **kwargs):
        """Translate and execute an Anthropic Messages API call.

        Accepts OpenAI-style parameters, translates to Anthropic format,
        and returns OpenAI-compatible response objects.
        """
        client = self._get_client()

        # Translate messages (extract system → top-level param)
        system_text, anthropic_messages = _translate_messages(messages)

        # Translate parameters
        is_streaming = kwargs.get("stream", False)
        anthropic_kwargs = _translate_kwargs(kwargs)
        anthropic_kwargs["model"] = model
        anthropic_kwargs["messages"] = anthropic_messages

        if system_text:
            anthropic_kwargs["system"] = system_text

        try:
            if is_streaming:
                stream = await client.messages.create(**anthropic_kwargs)
                return _StreamAdapter(stream)
            else:
                response = await client.messages.create(**anthropic_kwargs)
                return _translate_response(response)

        except Exception as e:
            # Translate Anthropic errors to OpenAI errors
            translated = _translate_error(e)
            if translated is not e:
                raise translated from e
            raise


class _Chat:
    """Duck-type compatible with openai.AsyncOpenAI().chat."""

    def __init__(self, completions: _Completions):
        self.completions = completions


class AnthropicAdapter:
    """Duck-type compatible with openai.AsyncOpenAI.

    Usage by Elelem internals:
        client = AnthropicAdapter(api_key="sk-...")
        client.chat.completions.create(messages=..., model=..., **kwargs)
        client.api_key = new_key  # token refresh
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        completions = _Completions(self)
        self.chat = _Chat(completions)
