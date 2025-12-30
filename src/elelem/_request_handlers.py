"""
Request state machine handlers for Elelem.

Each handler is a focused function that processes one state and returns
the next state transition. This makes the flow explicit and traceable.
"""

import asyncio
import logging
from typing import Any, Callable, Dict

from openai import RateLimitError

from ._request_state import (
    RequestState,
    RequestContext,
    StateTransition,
    TERMINAL_STATES,
)
from ._exceptions import InfrastructureError, ModelError
from ._reasoning_tokens import extract_token_counts, extract_reasoning_content
from ._response_processing import process_response_content, ChunkTimeoutError
from ._output_formats import FormatParseError, FormatSchemaError
from ._format_fixer import call_format_fixer


class RequestStateMachine:
    """Processes a request through explicit state transitions.

    The state machine makes the request flow visible:
    - Each state has a dedicated handler
    - Handlers return explicit transitions
    - Terminal states end processing with success/error

    Usage:
        ctx = RequestContext(...)
        machine = RequestStateMachine(elelem_instance)
        result = await machine.run(ctx)

        if result.next_state == RequestState.SUCCESS:
            return result.result
        else:
            raise result.error
    """

    def __init__(self, elelem_instance: Any):
        """Initialize with Elelem instance for access to config and helpers."""
        self.elelem = elelem_instance
        self.logger = elelem_instance.logger

        # Map states to their handlers
        self._handlers: Dict[RequestState, Callable] = {
            RequestState.CALL_API: self._handle_call_api,
            RequestState.EXTRACT_TOKENS: self._handle_extract_tokens,
            RequestState.VALIDATE_FORMAT: self._handle_validate_format,
            RequestState.TRY_FIXER: self._handle_try_fixer,
            RequestState.REDUCE_TEMPERATURE: self._handle_reduce_temperature,
            RequestState.WAIT_RATE_LIMIT: self._handle_wait_rate_limit,
        }

    async def run(self, ctx: RequestContext) -> StateTransition:
        """Run the state machine until a terminal state is reached.

        Args:
            ctx: Request context with all state

        Returns:
            StateTransition with terminal state, result/error, and state path
        """
        state = RequestState.CALL_API
        path = [state.name]

        while state not in TERMINAL_STATES:
            handler = self._handlers.get(state)
            if not handler:
                # Should never happen - programming error
                raise RuntimeError(f"No handler for state {state}")

            transition = await handler(ctx)
            state = transition.next_state
            path.append(state.name)

            # Terminal states carry their result/error
            if state in TERMINAL_STATES:
                transition.path = path
                return transition

        transition.path = path
        return transition

    # =========================================================================
    # State Handlers - Each handles one state and returns next transition
    # =========================================================================

    async def _handle_call_api(self, ctx: RequestContext) -> StateTransition:
        """Make API call to the provider.

        Transitions:
            - success → EXTRACT_TOKENS
            - timeout → NEXT_CANDIDATE (infrastructure error)
            - rate_limit → WAIT_RATE_LIMIT
            - infrastructure_error → NEXT_CANDIDATE
            - model_error → SKIP_MODEL
        """
        try:
            ctx.request_tracker.mark_llm_start()

            if ctx.api_kwargs.get("stream", False):
                # Streaming request
                stream = await asyncio.wait_for(
                    ctx.provider_client.chat.completions.create(
                        messages=ctx.messages,
                        model=ctx.model_name,
                        **ctx.api_kwargs
                    ),
                    timeout=ctx.timeout
                )
                ctx.response, ctx.chunk_count = await self.elelem._collect_streaming_response(
                    stream, ctx.request_id, chunk_timeout=ctx.chunk_timeout
                )
            else:
                # Non-streaming request
                ctx.response = await asyncio.wait_for(
                    ctx.provider_client.chat.completions.create(
                        messages=ctx.messages,
                        model=ctx.model_name,
                        **ctx.api_kwargs
                    ),
                    timeout=ctx.timeout
                )

            ctx.request_tracker.mark_llm_end()
            return StateTransition(RequestState.EXTRACT_TOKENS)

        except asyncio.TimeoutError:
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"Request timed out after {ctx.timeout}s",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        except ChunkTimeoutError as e:
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"Streaming chunk timeout: {e}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        except RateLimitError as e:
            ctx.error = e
            return StateTransition(RequestState.WAIT_RATE_LIMIT)

        except Exception as e:
            return self._classify_api_error(ctx, e)

    def _classify_api_error(self, ctx: RequestContext, error: Exception) -> StateTransition:
        """Classify an API error and return appropriate transition."""
        from openai import (
            AuthenticationError, PermissionDeniedError, BadRequestError,
            NotFoundError, InternalServerError, ConflictError, UnprocessableEntityError
        )

        # Check if it's an infrastructure error
        if self.elelem._is_infrastructure_error(error):
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"API infrastructure error: {error}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        # Check for JSON validation API error (special case for JSON format)
        if (self.elelem._is_json_validation_api_error(error) and
                ctx.format_handler and ctx.format_handler.name == "json"):
            # Will be handled in validation - continue with empty response
            ctx.response = None
            ctx.error = error
            return StateTransition(RequestState.VALIDATE_FORMAT)

        # Auth/permission errors -> infrastructure (try next provider)
        if isinstance(error, (AuthenticationError, PermissionDeniedError)):
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"Authentication/permission error: {error}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        # Server/request errors -> infrastructure (might work on another provider)
        if isinstance(error, (InternalServerError, BadRequestError, NotFoundError)):
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"Server/request error: {error}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        # Conflict/unprocessable -> model error (don't retry)
        if isinstance(error, (ConflictError, UnprocessableEntityError)):
            return StateTransition(
                RequestState.SKIP_MODEL,
                error=ModelError(
                    f"Request validation error: {error}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        # Check for rate limit in error string (fallback)
        error_str = str(error).lower()
        if "429" in error_str or "rate limit" in error_str:
            ctx.error = error
            return StateTransition(RequestState.WAIT_RATE_LIMIT)

        # Default: model error
        return StateTransition(
            RequestState.SKIP_MODEL,
            error=ModelError(
                f"Unexpected error: {error}",
                provider=ctx.provider_name,
                model=ctx.model_name
            )
        )

    async def _handle_extract_tokens(self, ctx: RequestContext) -> StateTransition:
        """Extract tokens and content from response.

        Transitions:
            - has format handler → VALIDATE_FORMAT
            - no format handler → SUCCESS
        """
        if ctx.response:
            # Extract token counts
            input_tokens, output_tokens, reasoning_tokens, _ = extract_token_counts(
                ctx.response, self.logger
            )
            ctx.total_input_tokens += input_tokens
            ctx.total_output_tokens += output_tokens
            ctx.total_reasoning_tokens += reasoning_tokens

            # Extract reasoning content
            ctx.reasoning_content = extract_reasoning_content(ctx.response, self.logger)

            # Process response content
            ctx.content = process_response_content(
                ctx.response, self.logger, ctx.format_handler
            )
        else:
            ctx.content = ""
            ctx.reasoning_content = None

        # Next state depends on whether we need format validation
        if ctx.format_handler:
            return StateTransition(RequestState.VALIDATE_FORMAT)
        else:
            return StateTransition(RequestState.SUCCESS, result=ctx.response)

    async def _handle_validate_format(self, ctx: RequestContext) -> StateTransition:
        """Validate format (JSON/YAML/CSV) and schema.

        Transitions:
            - valid → SUCCESS
            - parse_error → NEXT_CANDIDATE (infrastructure - response malformed)
            - schema_error → TRY_FIXER
        """
        try:
            # Parse content
            parse_result = ctx.format_handler.parse(ctx.content)

            if not parse_result.success:
                # Parse failed = infrastructure issue (response malformed)
                self._dump_debug(ctx, parse_result.error, "parse")
                return StateTransition(
                    RequestState.NEXT_CANDIDATE,
                    error=InfrastructureError(
                        f"{ctx.format_handler.name.upper()} parse failed: {str(parse_result.error)[:100]}",
                        provider=ctx.provider_name,
                        model=ctx.model_name
                    )
                )

            # Schema validation (if schema provided)
            if ctx.format_schema:
                validation = ctx.format_handler.validate_schema(
                    parse_result.data, ctx.format_schema
                )
                if not validation.is_valid:
                    error_msg = validation.error
                    if validation.error_path:
                        error_msg += f" at path: {validation.error_path}"

                    ctx.error = FormatSchemaError(
                        error_msg,
                        content=parse_result.content,
                        format_type=ctx.format_handler.name
                    )
                    ctx.error_content = parse_result.content
                    self._dump_debug(ctx, ctx.error, "schema")
                    return StateTransition(RequestState.TRY_FIXER)

            # Success - update content with possibly repaired version
            ctx.content = parse_result.content
            return StateTransition(RequestState.SUCCESS, result=ctx.response)

        except FormatParseError as e:
            self._dump_debug(ctx, e, "parse")
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"{ctx.format_handler.name.upper()} parse failed: {str(e)[:100]}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        except FormatSchemaError as e:
            ctx.error = e
            ctx.error_content = e.content or ctx.content
            self._dump_debug(ctx, e, "schema")
            return StateTransition(RequestState.TRY_FIXER)

    def _dump_debug(self, ctx: RequestContext, error: Exception, error_type: str) -> None:
        """Dump debug info for validation failures."""
        self.elelem._dump_validation_debug(
            ctx.request_id,
            ctx.messages,
            ctx.api_kwargs,
            ctx.error_content or ctx.content,
            error,
            ctx.format_handler.name if ctx.format_handler else "unknown",
            provider=ctx.provider_name,
            model_id=ctx.model_name,
            original_model=ctx.original_model,
            schema=ctx.format_schema
        )

    async def _handle_try_fixer(self, ctx: RequestContext) -> StateTransition:
        """Try to fix invalid content with LLM fixer.

        Transitions:
            - fixed → SUCCESS
            - unfixable (truncated/empty) → NEXT_CANDIDATE
            - fixable but failed → REDUCE_TEMPERATURE
        """
        if not self.elelem._json_fixer_enabled or not ctx.format_schema:
            # No fixer available - go to temperature reduction
            return StateTransition(RequestState.REDUCE_TEMPERATURE)

        fixer_output = await call_format_fixer(
            elelem_instance=self.elelem,
            format_handler=ctx.format_handler,
            invalid_content=ctx.error_content or ctx.content,
            error=str(ctx.error),
            schema=ctx.format_schema,
            request_id=ctx.request_id,
            fixer_model=self.elelem._json_fixer_model
        )

        # Check if content was unfixable (truncated/empty)
        if not fixer_output.is_fixable:
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"{ctx.format_handler.name.upper()} response unfixable (truncated/empty)",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        # Fixed successfully?
        if fixer_output.content:
            ctx.content = fixer_output.content
            ctx.request_tracker.record_retry("format_fixer")
            return StateTransition(RequestState.SUCCESS, result=ctx.response)

        # Fixer couldn't fix - try temperature reduction
        return StateTransition(RequestState.REDUCE_TEMPERATURE)

    async def _handle_reduce_temperature(self, ctx: RequestContext) -> StateTransition:
        """Reduce temperature and retry, or give up.

        Transitions:
            - can reduce → CALL_API
            - exhausted retries → SKIP_MODEL
        """
        if ctx.attempt >= ctx.max_retries:
            return StateTransition(
                RequestState.SKIP_MODEL,
                error=ModelError(
                    f"{ctx.format_handler.name.upper()} schema validation failed after all retries: {ctx.error}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        # Try temperature reduction
        if ctx.temperature_reductions:
            reduction_idx = min(ctx.attempt, len(ctx.temperature_reductions) - 1)
            reduction = ctx.temperature_reductions[reduction_idx]
            new_temp = max(ctx.min_temp, ctx.current_temperature - reduction)

            if new_temp < ctx.current_temperature:
                ctx.current_temperature = new_temp
                ctx.api_kwargs['temperature'] = new_temp
                ctx.request_tracker.record_retry("temperature_reductions")

                self.logger.warning(
                    f"[{ctx.request_id}] {ctx.format_handler.name.upper()} schema validation failed, "
                    f"reducing temperature to {new_temp}"
                )
                self.logger.warning(f"[{ctx.request_id}] Validation error: {ctx.error}")

                ctx.attempt += 1
                return StateTransition(RequestState.CALL_API)

        # Try removing response_format (JSON only)
        if (ctx.format_handler and ctx.format_handler.name == "json" and
                'response_format' in ctx.api_kwargs):
            ctx.api_kwargs.pop('response_format', None)
            ctx.api_kwargs['temperature'] = ctx.original_temperature
            ctx.current_temperature = ctx.original_temperature
            ctx.request_tracker.record_retry("response_format_removals")

            self.logger.warning(f"[{ctx.request_id}] Removing response_format and retrying")

            ctx.attempt += 1
            return StateTransition(RequestState.CALL_API)

        # All options exhausted
        return StateTransition(
            RequestState.SKIP_MODEL,
            error=ModelError(
                f"{ctx.format_handler.name.upper()} schema validation failed, temperature exhausted",
                provider=ctx.provider_name,
                model=ctx.model_name
            )
        )

    async def _handle_wait_rate_limit(self, ctx: RequestContext) -> StateTransition:
        """Handle rate limit with exponential backoff.

        Transitions:
            - can retry → CALL_API
            - exhausted → NEXT_CANDIDATE
        """
        if ctx.rate_limit_attempts >= ctx.max_rate_limit_retries:
            return StateTransition(
                RequestState.NEXT_CANDIDATE,
                error=InfrastructureError(
                    f"Rate limit exhausted after {ctx.rate_limit_attempts} retries: {ctx.error}",
                    provider=ctx.provider_name,
                    model=ctx.model_name
                )
            )

        # Calculate backoff time
        backoff_idx = min(ctx.rate_limit_attempts, len(ctx.rate_limit_backoff) - 1)
        wait_time = ctx.rate_limit_backoff[backoff_idx]

        ctx.rate_limit_attempts += 1
        ctx.request_tracker.record_retry("rate_limit_retries")

        self.logger.warning(
            f"[{ctx.request_id}] Rate limit hit, waiting {wait_time}s "
            f"(attempt {ctx.rate_limit_attempts}/{ctx.max_rate_limit_retries})"
        )

        await asyncio.sleep(wait_time)
        return StateTransition(RequestState.CALL_API)
