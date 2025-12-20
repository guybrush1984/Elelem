"""
Main Elelem class - Unified API wrapper for OpenAI, GROQ, and DeepInfra
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import openai
from openai import (
    BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError,
    ConflictError, UnprocessableEntityError, RateLimitError, InternalServerError,
)
import pandas as pd
from .config import Config
from .metrics import MetricsStore
from ._reasoning_tokens import extract_token_counts, extract_reasoning_content
from ._exceptions import InfrastructureError, ModelError
from ._cost_calculation import calculate_costs, extract_runtime_costs
from ._response_processing import collect_streaming_response, ChunkTimeoutError, remove_think_tags, extract_json_from_markdown, extract_yaml_from_markdown, process_response_content
from ._json_validation import validate_json_schema, is_json_validation_api_error, add_json_instructions_to_messages, validate_json_response
from ._yaml_validation import validate_yaml_schema, add_yaml_instructions_to_messages, validate_yaml_response
from ._provider_management import create_provider_client, initialize_providers, get_model_config
from ._retry_logic import update_retry_analytics, handle_json_retry, is_infrastructure_error
from ._request_execution import prepare_api_kwargs
from ._benchmark_store import reorder_candidates_by_benchmark
from ._json_fixer import call_json_fixer


class Elelem:
    """Unified API wrapper with cost tracking, JSON validation, and retry logic."""
    
    def __init__(self, metrics_persist_file: Optional[str] = None, extra_provider_dirs: Optional[List[str]] = None,
                 cache_enabled: bool = False, cache_ttl: int = 300, cache_max_size: int = 50000,
                 json_fixer_enabled: bool = True, json_fixer_model: Optional[str] = None):
        self.logger = logging.getLogger("elelem")
        self.config = Config(extra_provider_dirs)
        self._models = self._load_models()

        # Lazy provider initialization - start empty, probe on first use
        self._providers = {}
        self._probed_providers = set()  # Track which providers have been attempted
        self._failed_providers = {}  # provider_name -> (timestamp, retryable)
        self._provider_retry_interval = 300  # Retry failed providers after 5 minutes

        # JSON fixer configuration
        self._json_fixer_enabled = json_fixer_enabled
        self._json_fixer_model = json_fixer_model

        # Initialize metrics system (unified SQLAlchemy backend)
        self._metrics_store = MetricsStore()

        # Initialize cache if enabled (shares database with metrics)
        if cache_enabled:
            from .cache import PostgresCache
            self.cache = PostgresCache(
                engine=self._metrics_store.engine,
                ttl_seconds=cache_ttl,
                max_response_size=cache_max_size,
                logger=self.logger
            )
        else:
            self.cache = None
    
    def _load_models(self) -> Dict[str, Any]:
        """Load model definitions using the Config system."""
        return self.config.models
    
    def _create_provider_client(self, api_key: str, base_url: str, timeout: int = 120, provider_name: str = None, default_headers: Dict = None):
        """Create an OpenAI-compatible client for any provider."""
        return create_provider_client(api_key, base_url, timeout, provider_name, default_headers)

    def _ensure_provider_initialized(self, provider_name: str) -> bool:
        """
        Ensure a provider is probed and initialized on first use.
        Returns True if provider is ready, False if unavailable.
        Caches result so subsequent calls are instant.
        Retries failed providers after retry interval if failure was retryable.
        """
        import time
        import os

        # Already initialized successfully
        if provider_name in self._providers:
            return True

        # Check if we should retry a failed provider
        if provider_name in self._failed_providers:
            failed_time, retryable = self._failed_providers[provider_name]
            if not retryable:
                # Permanent failure (auth error) - never retry
                return False
            elapsed = time.time() - failed_time
            if elapsed < self._provider_retry_interval:
                # Not enough time passed - skip retry
                return False
            # Enough time passed - retry
            self.logger.info(f"[{provider_name}] Retrying after {elapsed:.0f}s...")
            del self._failed_providers[provider_name]
            self._probed_providers.discard(provider_name)

        # Already attempted this provider (and not retrying)
        if provider_name in self._probed_providers:
            return False

        # Mark as probed (whether successful or not)
        self._probed_providers.add(provider_name)

        # Get provider config
        provider_config = self.config.providers.get(provider_name)
        if not provider_config:
            self.logger.warning(f"[{provider_name}] Provider not found in configuration")
            return False

        # Import needed functions
        from ._provider_management import probe_endpoint, select_working_endpoint

        # Determine API key
        base_provider = provider_config.get("base_provider")
        if base_provider:
            env_var = f"{base_provider.upper()}_API_KEY"
        else:
            env_var = f"{provider_name.upper()}_API_KEY"

        api_key = os.getenv(env_var)

        if not api_key:
            self.logger.debug(f"[{provider_name}] No API key found (env var: {env_var})")
            # No API key is a permanent failure
            self._failed_providers[provider_name] = (time.time(), False)
            return False

        # Determine endpoint (probe if needed)
        endpoint = None
        retryable = True
        probe_timeout = provider_config.get("probe_timeout", 5.0)

        if "endpoints" in provider_config:
            # Multiple endpoints - probe and select
            self.logger.info(f"[{provider_name}] Probing {len(provider_config['endpoints'])} endpoint(s)...")
            endpoint, retryable = select_working_endpoint(
                provider_config["endpoints"],
                probe_timeout,
                provider_name,
                self.logger,
                api_key
            )
            if not endpoint:
                self.logger.warning(f"[{provider_name}] No working endpoints found (retryable={retryable})")
                self._failed_providers[provider_name] = (time.time(), retryable)
                return False

        elif "endpoint" in provider_config:
            # Single endpoint - probe it
            single_endpoint = provider_config["endpoint"]
            self.logger.info(f"[{provider_name}] Probing endpoint: {single_endpoint}")

            result = probe_endpoint(single_endpoint, probe_timeout, self.logger, api_key)
            if result.success:
                self.logger.info(f"[{provider_name}] ✅ Endpoint accessible")
                endpoint = single_endpoint
            else:
                self.logger.warning(f"[{provider_name}] ❌ Endpoint not accessible (retryable={result.retryable})")
                self._failed_providers[provider_name] = (time.time(), result.retryable)
                return False
        else:
            self.logger.error(f"[{provider_name}] No endpoint configured")
            self._failed_providers[provider_name] = (time.time(), False)
            return False

        # Create provider client
        custom_headers = provider_config.get("headers")
        self._providers[provider_name] = create_provider_client(
            api_key=api_key,
            base_url=endpoint,
            timeout=self.config.timeout_seconds,
            provider_name=provider_name,
            default_headers=custom_headers
        )
        self.logger.debug(f"[{provider_name}] Initialized successfully")
        return True

    def _reset_stats(self):
        """Reset all statistics."""
        self._metrics_store.reset()
        
    def _get_model_config(self, model: str) -> tuple[str, str]:
        """Get provider and model_id from model configuration (opaque key lookup)."""
        return get_model_config(model, self._models, self._providers)
        
        
    async def _collect_streaming_response(self, stream, request_id=None, chunk_timeout=None):
        """Collect streaming chunks and reconstruct a normal response object.

        Args:
            stream: Async stream of chunks
            request_id: Request ID for logging
            chunk_timeout: Optional timeout (seconds) for receiving each chunk.
                          If no chunk arrives within this time, raises ChunkTimeoutError.

        Returns:
            Tuple of (response, chunk_count)
        """
        return await collect_streaming_response(stream, logger=self.logger, request_id=request_id, chunk_timeout=chunk_timeout)
        
    def _remove_think_tags(self, content: str) -> str:
        """Remove <think>...</think> tags from content."""
        return remove_think_tags(content, self.logger)
        
    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks."""
        return extract_json_from_markdown(content, self.logger)
        
    def _validate_json_schema(self, json_obj: Any, schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a JSON object against a JSON Schema."""
        return validate_json_schema(json_obj, schema)
        
    def _is_json_validation_api_error(self, error: Exception) -> bool:
        """Check if the error is a json_validate_failed API error."""
        return is_json_validation_api_error(error)
        
    def _add_json_instructions_to_messages(self, messages: List[Dict[str, str]], capabilities: Dict, json_schema: Optional[Dict] = None, enforce_schema_in_prompt: bool = False) -> List[Dict[str, str]]:
        """Add JSON formatting instructions to messages when response_format is JSON."""
        supports_system = capabilities.get("supports_system", True)
        return add_json_instructions_to_messages(messages, supports_system, json_schema, enforce_schema_in_prompt)

    def _extract_yaml_from_markdown(self, content: str) -> str:
        """Extract YAML from markdown code blocks."""
        return extract_yaml_from_markdown(content, self.logger)

    def _validate_yaml_schema(self, yaml_obj: Any, schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a YAML object against a JSON Schema."""
        return validate_yaml_schema(yaml_obj, schema)

    def _add_yaml_instructions_to_messages(self, messages: List[Dict[str, str]], capabilities: Dict, yaml_schema: Optional[Dict] = None, enforce_schema_in_prompt: bool = False) -> List[Dict[str, str]]:
        """Add YAML formatting instructions to messages when YAML mode is requested."""
        supports_system = capabilities.get("supports_system", True)
        return add_yaml_instructions_to_messages(messages, supports_system, yaml_schema, enforce_schema_in_prompt)

    def _validate_yaml_response(self, content: str, yaml_schema: Any) -> None:
        """Validate YAML response content and schema."""
        validate_yaml_response(content, yaml_schema)

    def _calculate_costs(self, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, runtime_costs: Dict = None, candidate_cost_config: Dict = None) -> Dict[str, float]:
        """Calculate costs based on model pricing or runtime data from provider."""
        return calculate_costs(model, input_tokens, output_tokens, reasoning_tokens, runtime_costs, candidate_cost_config, self.logger)
    
    def _extract_runtime_costs(self, response, cost_config: str) -> Dict[str, Any]:
        """Extract runtime cost information from response when cost config is 'runtime'."""
        return extract_runtime_costs(response, cost_config, self.logger)
        
    
    
    def _preprocess_messages(self, messages: List[Dict[str, str]], model: str, json_mode_requested: bool, yaml_mode_requested: bool, capabilities: Dict, json_schema: Optional[Dict] = None, yaml_schema: Optional[Dict] = None, enforce_schema_in_prompt: bool = False) -> List[Dict[str, str]]:
        """Preprocess messages for the request."""
        if json_mode_requested:
            # Always add JSON instructions when JSON is requested
            # Include schema in instructions only if enforce_schema_in_prompt is True
            return self._add_json_instructions_to_messages(messages, capabilities, json_schema, enforce_schema_in_prompt)
        elif yaml_mode_requested:
            # Always add YAML instructions when YAML is requested (client-side only)
            # Include schema in instructions only if enforce_schema_in_prompt is True
            return self._add_yaml_instructions_to_messages(messages, capabilities, yaml_schema, enforce_schema_in_prompt)
        return messages
    
    def _cleanup_api_kwargs(self, api_kwargs: Dict, model: str, model_config: Dict) -> None:
        """Remove unsupported parameters from api_kwargs."""
        self.logger.debug(f"[DEBUG] _cleanup_api_kwargs called for model '{model}'")
        self.logger.debug(f"[DEBUG]   api_kwargs keys: {list(api_kwargs.keys())}")
        self.logger.debug(f"[DEBUG]   response_format in api_kwargs: {'response_format' in api_kwargs}")
        if 'response_format' in api_kwargs:
            self.logger.debug(f"[DEBUG]   response_format value: {api_kwargs['response_format']}")

        # Get capabilities from the passed model_config (which comes from candidate)
        capabilities = model_config.get("capabilities", {})
        supports_json_mode = capabilities.get("supports_json_mode", True)
        should_remove_rf = not supports_json_mode
        has_response_format = "response_format" in api_kwargs

        self.logger.debug(f"[DEBUG]   capabilities from model_config: {capabilities}")
        self.logger.debug(f"[DEBUG]   supports_json_mode: {supports_json_mode}")
        self.logger.debug(f"[DEBUG]   should_remove_rf: {should_remove_rf}")
        self.logger.debug(f"[DEBUG]   has response_format: {has_response_format}")

        if should_remove_rf and has_response_format:
            self.logger.debug(f"[DEBUG] REMOVING response_format for {model} (not supported)")
            api_kwargs.pop("response_format")
        else:
            if not should_remove_rf:
                self.logger.debug(f"[DEBUG] NOT removing response_format - model supports JSON mode")
            if not has_response_format:
                self.logger.debug(f"[DEBUG] NOT removing response_format - not present in kwargs")

        # Remove temperature if not supported
        if not capabilities.get("supports_temperature", True):
            if "temperature" in api_kwargs:
                self.logger.debug(f"Removing temperature for {model} (not supported)")
                api_kwargs.pop("temperature")

        # Remove Elelem-specific parameters that should not be passed to provider APIs
        if "enforce_schema_in_prompt" in api_kwargs:
            api_kwargs.pop("enforce_schema_in_prompt")
        if "yaml_schema" in api_kwargs:
            api_kwargs.pop("yaml_schema")

        self.logger.debug(f"[DEBUG] _cleanup_api_kwargs finished. Final response_format in api_kwargs: {'response_format' in api_kwargs}")

    def _process_response_content(self, response: Any, json_mode_requested: bool, yaml_mode_requested: bool) -> str:
        """Process and clean response content."""
        return process_response_content(response, json_mode_requested, yaml_mode_requested, self.logger)
    
    def _validate_json_response(self, content: str, json_schema: Any, api_error: Exception, request_id: str = None) -> str:
        """Validate JSON response content and schema. Returns possibly repaired content."""
        return validate_json_response(content, json_schema, api_error, request_id)

    def _dump_validation_debug(self, request_id: str, messages: List[Dict[str, str]],
                                api_kwargs: Dict[str, Any], content: str, error: Exception,
                                validation_type: str = "json",
                                provider: str = None, model_id: str = None,
                                original_model: str = None,
                                schema: Dict[str, Any] = None) -> Optional[str]:
        """Dump request/response for debugging validation failures.

        Only active when ELELEM_DEBUG_VALIDATION env var is set.
        Files are written to ELELEM_DEBUG_DIR (default: /tmp/elelem_debug).

        Args:
            request_id: Unique request identifier
            messages: The messages sent in the request
            api_kwargs: The API kwargs used for the request
            content: The response content that failed validation
            error: The validation error
            validation_type: "json" or "yaml"
            provider: Provider name (e.g., "deepinfra", "fireworks")
            model_id: Model ID at the provider
            original_model: Original model requested (e.g., "virtual:deepseek-v3.2-cheap")

        Returns:
            Path to debug file if created, None otherwise
        """
        if not os.environ.get('ELELEM_DEBUG_VALIDATION'):
            return None

        try:
            debug_dir = os.environ.get('ELELEM_DEBUG_DIR', '/tmp/elelem_debug')
            os.makedirs(debug_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/elelem_debug_{validation_type}_{request_id}_{timestamp}.json"

            # Build safe copy of api_kwargs (exclude non-serializable items)
            safe_kwargs = {}
            for k, v in api_kwargs.items():
                try:
                    json.dumps(v)  # Test serializability
                    safe_kwargs[k] = v
                except (TypeError, ValueError):
                    safe_kwargs[k] = str(v)

            debug_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "validation_type": validation_type,
                "error": str(error),
                "model_info": {
                    "original_model": original_model,
                    "provider": provider,
                    "model_id": model_id,
                },
                "request": {
                    "messages": messages,
                    "api_kwargs": safe_kwargs,
                },
                "response": {
                    "content": content,
                },
                "schema": schema,
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"[{request_id}] Debug dump written to: {filename}")
            return filename
        except Exception as dump_error:
            self.logger.warning(f"[{request_id}] Failed to write debug dump: {dump_error}")
            return None

    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tags: Union[str, List[str]] = [],
        cache: bool = True,
        **kwargs
    ) -> Dict:
        """
        Unified chat completion method matching OpenAI API signature exactly.
        Uses unified candidate-based iteration for both regular and virtual models.

        Args:
            messages: List of message dictionaries
            model: Model string in "provider:model" format (or virtual:model)
            tags: Tags for cost tracking (elelem-specific parameter)
            cache: Whether to use caching (elelem-specific parameter, default True)
            **kwargs: All OpenAI API parameters (response_format, temperature, etc.)

        Returns:
            OpenAI-compatible response dictionary
        """
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Normalize tags
        if isinstance(tags, str):
            tags = [tags]
        elif not tags:
            tags = []

        # Compute cache key ONCE with original unmodified values (before any preprocessing)
        # This key will be reused for both checking and saving to ensure consistency
        cache_key = None
        if self.cache and cache:
            cache_key = self.cache.get_cache_key(model, messages, **kwargs)
            cache_result = self.cache.get(cache_key)

            if cache_result:
                cached_response, cache_age = cache_result

                # Reconstruct response object from cached data
                from openai.types.chat import ChatCompletion
                response = ChatCompletion(**cached_response)

                # Mark as cached in elelem_metrics
                response.elelem_metrics['cached'] = True
                response.elelem_metrics['cache_age_seconds'] = cache_age

                # Cached responses are free - zero out all costs
                response.elelem_metrics['costs_usd']['input_cost_usd'] = 0.0
                response.elelem_metrics['costs_usd']['output_cost_usd'] = 0.0
                response.elelem_metrics['costs_usd']['reasoning_cost_usd'] = 0.0
                response.elelem_metrics['costs_usd']['total_cost_usd'] = 0.0

                response.elelem_metrics['total_duration_seconds'] = time.time() - start_time

                # Track cache hit in metrics (minimal record)
                cache_tracker = self._metrics_store.start_request(
                    request_id=request_id,
                    requested_model=model,
                    tags=tags,
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    stream=kwargs.get("stream", False)
                )
                cache_tracker.cache_hit = True
                cache_tracker.cache_age_seconds = cache_age
                cache_tracker.finalize_with_candidate(
                    self._metrics_store,
                    selected_candidate="cache",
                    actual_model=model,
                    actual_provider="cache",
                    status="success",
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    reasoning_tokens=getattr(response.usage, 'reasoning_tokens', 0),
                    total_cost_usd=0.0
                )

                self.logger.info(f"[{request_id}] ✅ Cache HIT (age: {cache_age:.1f}s)")
                return response

        # Cache miss - continue with normal request
        # Create RequestTracker for unified metrics
        request_tracker = self._metrics_store.start_request(
            request_id=request_id,
            requested_model=model,
            tags=tags,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            stream=kwargs.get("stream", False)
        )
        
        # Get model configuration and candidates
        try:
            model_config = self.config.get_model_config(model)
            candidates = model_config['candidates']

            # Filter out candidates whose providers are not available
            # Lazy initialization: probe providers on first use
            available_candidates = [
                c for c in candidates
                if self._ensure_provider_initialized(c.get('provider'))
            ]

            if len(available_candidates) < len(candidates):
                skipped_count = len(candidates) - len(available_candidates)
                skipped_providers = set(c.get('provider') for c in candidates if c.get('provider') not in self._providers)
                self.logger.info(
                    f"[{request_id}] Skipped {skipped_count} candidate(s) with unavailable provider(s): {', '.join(skipped_providers)}"
                )

            if not available_candidates:
                # All candidates filtered out - no providers available
                request_tracker.finalize_failure(self._metrics_store, "NoProvidersAvailable", "All candidate providers are unavailable")
                raise ValueError(f"No available providers for model {model} (all endpoints unreachable)")

            candidates = available_candidates

            # Apply benchmark-based reordering if routing config exists (virtual models only)
            routing = model_config.get('routing')
            if routing:
                speed_weight = routing.get('speed_weight', 1.0)
                min_tokens_per_sec = routing.get('min_tokens_per_sec', 0.0)
                candidates = reorder_candidates_by_benchmark(
                    candidates,
                    speed_weight=speed_weight,
                    min_tokens_per_sec=min_tokens_per_sec,
                    logger=self.logger
                )

        except ValueError as e:
            # Record failure for invalid model
            request_tracker.finalize_failure(self._metrics_store, "ModelNotFound", str(e))
            raise ValueError(f"Model configuration error: {e}")

        # Extract response_format and detect format type
        response_format = kwargs.get("response_format", {})
        response_format_type = response_format.get("type") if isinstance(response_format, dict) else None

        # Detect JSON mode request (both old and new formats)
        json_mode_requested = response_format_type in ["json_object", "json_schema"]

        # Extract json_schema from either source
        json_schema = None

        if response_format_type == "json_object":
            # Old JSON mode - schema provided separately (Elelem-specific)
            json_schema = kwargs.get("json_schema")

        elif response_format_type == "json_schema":
            # New structured outputs format (OpenAI standard)
            # Extract nested schema
            json_schema_obj = response_format.get("json_schema", {})
            json_schema = json_schema_obj.get("schema")

            # Validate exclusivity (can't use both formats)
            if "json_schema" in kwargs:
                raise ValueError(
                    "Cannot use both response_format with type='json_schema' "
                    "and the separate json_schema parameter. Use one or the other."
                )

            # CRITICAL: Downgrade to basic JSON mode for providers
            # No provider is guaranteed to support structured outputs format
            # We'll validate the schema client-side instead
            kwargs["response_format"] = {"type": "json_object"}
            self.logger.debug(f"[{request_id}] Converted structured outputs format to basic JSON mode (client-side validation)")

        # Detect YAML mode request (Elelem-specific, client-side only)
        yaml_schema = kwargs.get("yaml_schema")
        yaml_mode_requested = yaml_schema is not None

        # Validate mutual exclusivity between JSON and YAML
        if json_mode_requested and yaml_mode_requested:
            raise ValueError(
                "Cannot use both JSON and YAML modes simultaneously. "
                "Provide either response_format with json_object/json_schema OR yaml_schema, not both."
            )

        # Get original temperature
        original_temperature = kwargs.get("temperature", 1.0)

        # Remove Elelem-specific parameters (if present)
        if "json_schema" in kwargs:
            kwargs.pop("json_schema")
        if "yaml_schema" in kwargs:
            kwargs.pop("yaml_schema")

        # Warn if json_schema provided without JSON response format
        if json_schema and not json_mode_requested:
            self.logger.warning(
                f"[{request_id}] json_schema provided but response_format is not set to json_object or json_schema. "
                "Schema validation will be skipped."
            )
        
        # Build list of candidate providers for logging (with benchmark scores if available)
        def format_candidate(c):
            provider = c.get('provider')
            score = c.get('_benchmark_score')
            if score is not None:
                return f"{provider}({score:.1f})"
            return provider
        candidate_info = [format_candidate(c) for c in candidates]
        self.logger.info(f"[{request_id}] 🚀 Starting {model} with {len(candidates)} candidate(s): {', '.join(candidate_info)} (temp={original_temperature})")
        if json_mode_requested:
            self.logger.debug(f"[{request_id}] 📋 JSON mode requested")
        if yaml_mode_requested:
            self.logger.debug(f"[{request_id}] 📄 YAML mode requested")
        
        # Iterate through candidates
        # Track failed model_references to skip candidates with the same underlying model
        failed_model_refs = set()
        last_error = None

        for candidate_idx, candidate in enumerate(candidates):
            # Skip candidates whose model_reference has already failed with ModelError
            candidate_model_ref = candidate.get('model_reference')
            if candidate_model_ref and candidate_model_ref in failed_model_refs:
                self.logger.debug(f"[{request_id}] Skipping candidate {candidate_idx + 1} (model_reference '{candidate_model_ref}' already failed)")
                continue

            try:
                return await self._attempt_candidate(
                    candidate, candidate_idx + 1, len(candidates),
                    messages, model, model_config, request_id,
                    json_mode_requested, json_schema, yaml_mode_requested, yaml_schema,
                    original_temperature, tags, cache, cache_key, start_time, request_tracker, **kwargs
                )
            except InfrastructureError as e:
                self.logger.warning(f"[{request_id}] 🔄 Candidate {candidate_idx + 1} failed (infra): {e}")
                request_tracker.record_retry("candidate_iterations")
                last_error = e
                # Continue to next candidate (infrastructure errors don't blacklist the model)
                continue
            except ModelError as e:
                # Model errors: skip all remaining candidates with the same model_reference
                if candidate_model_ref:
                    failed_model_refs.add(candidate_model_ref)
                    self.logger.warning(f"[{request_id}] 🔄 Candidate {candidate_idx + 1} failed (model): {e} - skipping model_reference '{candidate_model_ref}'")
                    request_tracker.record_retry("candidate_iterations")
                    last_error = e
                    # Continue to find a candidate with a different model_reference
                    continue
                else:
                    # No model_reference - can't do model-level failover, raise immediately
                    request_tracker.finalize_failure(self._metrics_store, "ModelError", str(e))
                    raise e

        # All candidates exhausted (either tried or skipped)
        if last_error:
            request_tracker.finalize_failure(self._metrics_store, "AllCandidatesFailed", str(last_error))
            raise last_error

        # This should never be reached - if it is, there's a logic error
        raise RuntimeError(f"FATAL: Candidate loop completed without returning or raising - this is a bug in Elelem")
    
    
    async def _attempt_candidate(self, candidate, candidate_idx, total_candidates,
                                messages, original_model, model_config, request_id,
                                json_mode_requested, json_schema, yaml_mode_requested, yaml_schema,
                                original_temperature, tags, cache, cache_key, start_time, request_tracker, **kwargs):
        """Attempt to complete request with a specific candidate."""
        
        # Get timeout for this candidate
        timeout = self.config.get_candidate_timeout(candidate, model_config)
        
        # Setup provider and model for this candidate
        provider_name = candidate['provider']
        model_name = candidate['model_id']
        provider_client = self._providers[provider_name]
        capabilities = candidate.get('capabilities', {})
        
        # Use original model reference for statistics (cost lookup)
        # For virtual models: use the original model reference (e.g., "parasail:gpt-oss-120b")
        # For regular models: use the original requested model name (e.g., "fireworks:deepseek-v3p1")
        stats_model_name = candidate.get('original_model_ref', original_model)
        candidate_model_name = f"{provider_name}:{model_name}"
        
        self.logger.info(f"[{request_id}] 🎯 Candidate {candidate_idx}/{total_candidates}: {candidate_model_name} (timeout={timeout}s)")
        
        # Get provider configuration for defaults
        provider_config = self.config.get_provider_config(provider_name)

        # Prepare API parameters with proper precedence
        api_kwargs = prepare_api_kwargs(kwargs, original_temperature, provider_config,
                                      candidate, original_model, self.config, provider_name)
        
        # Clean up unsupported parameters for this candidate model
        self.logger.debug(f"[{request_id}] Full candidate dict: {candidate}")
        candidate_key = candidate.get('model', f"{provider_name}:{model_name}")  # Fallback to constructed name
        self.logger.debug(f"[{request_id}] Candidate key: {candidate_key}")
        self.logger.debug(f"[{request_id}] Capabilities passed to cleanup: {capabilities}")
        self.logger.debug(f"[{request_id}] Before cleanup - response_format in kwargs: {'response_format' in api_kwargs}")
        self._cleanup_api_kwargs(api_kwargs, candidate_key, {'capabilities': capabilities})
        self.logger.debug(f"[{request_id}] After cleanup - response_format in kwargs: {'response_format' in api_kwargs}")

        # Preprocess messages for JSON/YAML mode if needed
        # Only inject schema into prompt if enforce_schema_in_prompt=True (default False to save tokens)
        enforce_schema_in_prompt = kwargs.get('enforce_schema_in_prompt', False)
        modified_messages = self._preprocess_messages(messages, candidate_model_name, json_mode_requested, yaml_mode_requested, capabilities, json_schema, yaml_schema, enforce_schema_in_prompt)

        # Retry logic for this candidate (temperature reduction, JSON validation)
        max_retries = self.config.retry_settings["max_json_retries"]
        max_rate_limit_retries = self.config.retry_settings["max_rate_limit_retries"]
        temperature_reductions = self.config.retry_settings["temperature_reductions"]
        min_temp = self.config.retry_settings["min_temp"]

        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        rate_limit_attempts = 0  # Separate counter for rate limit retries

        # Use max of JSON retries and rate limit retries to ensure both can complete
        max_loop_iterations = max(max_retries, max_rate_limit_retries) + 1
        for attempt in range(max_loop_iterations):
            try:
                current_temp = api_kwargs.get('temperature', original_temperature)
                self.logger.debug(f"[{request_id}] 🔄 Attempt {attempt + 1}/{max_retries + 1} - temp={current_temp}")
                
                # Make API call with timeout
                try:
                    chunk_count = None  # Track streaming chunks
                    if api_kwargs.get("stream", False):
                        # Handle streaming response
                        # Use timeout for initial connection, then chunk_timeout for streaming
                        chunk_timeout = self.config.get_candidate_chunk_timeout(candidate, model_config)
                        stream = await asyncio.wait_for(
                            provider_client.chat.completions.create(
                                messages=modified_messages,
                                model=model_name,
                                **api_kwargs
                            ),
                            timeout=timeout
                        )
                        response, chunk_count = await self._collect_streaming_response(stream, request_id, chunk_timeout=chunk_timeout)
                    else:
                        # Handle non-streaming response
                        response = await asyncio.wait_for(
                            provider_client.chat.completions.create(
                                messages=modified_messages,
                                model=model_name,
                                **api_kwargs
                            ),
                            timeout=timeout
                        )
                    api_error = None
                except asyncio.TimeoutError as e:
                    # Timeout is an infrastructure error - try next candidate
                    raise InfrastructureError(f"Request timed out after {timeout}s", provider=provider_name, model=model_name)
                except ChunkTimeoutError as e:
                    # Chunk timeout (cold start or stream stall) - try next candidate
                    raise InfrastructureError(f"Streaming chunk timeout: {str(e)}", provider=provider_name, model=model_name)
                except RateLimitError:
                    # Let rate limit errors pass through to outer handler for proper retry logic
                    raise
                except Exception as api_e:
                    # Classify the error
                    if self._is_infrastructure_error(api_e):
                        raise InfrastructureError(f"API infrastructure error: {str(api_e)}", provider=provider_name, model=model_name)
                    elif self._is_json_validation_api_error(api_e) and json_mode_requested:
                        # API-level JSON validation errors will be handled in JSON validation section
                        api_error = api_e
                        response = None
                    else:
                        # Other API errors are typically model errors (don't iterate)
                        raise ModelError(f"API error: {str(api_e)}", provider=provider_name, model=model_name)
                
                # Extract tokens from response
                if response:
                    # Extract normalized token counts
                    input_tokens, output_tokens, reasoning_tokens, total_tokens = extract_token_counts(response, self.logger)

                    # Extract reasoning content
                    reasoning_content = extract_reasoning_content(response, self.logger)

                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_reasoning_tokens += reasoning_tokens

                    content = self._process_response_content(response, json_mode_requested, yaml_mode_requested)
                else:
                    content = ""
                    reasoning_content = None
                
                # JSON validation if requested
                if json_mode_requested:
                    try:
                        # Validation may repair malformed JSON, update content with repaired version
                        content = self._validate_json_response(content, json_schema, api_error, request_id)
                    except json.JSONDecodeError as e:
                        if attempt < max_retries:
                            # Try temperature reduction
                            if temperature_reductions and len(temperature_reductions) > 0:
                                reduction_idx = min(attempt, len(temperature_reductions) - 1)
                                reduction = temperature_reductions[reduction_idx]
                                new_temp = max(current_temp - reduction, min_temp)

                                if new_temp < current_temp:
                                    api_kwargs['temperature'] = new_temp
                                    request_tracker.record_retry("temperature_reductions")
                                    self.logger.warning(f"[{request_id}] JSON validation failed, reducing temperature to {new_temp}")
                                    self.logger.warning(f"[{request_id}] Validation error: {e}")
                                    self.logger.warning(f"[{request_id}] Failed content:\n{content}")
                                    self._dump_validation_debug(request_id, messages, api_kwargs, content, e, "json",
                                                                provider=provider_name, model_id=model_name, original_model=original_model,
                                                                schema=json_schema)
                                    continue

                            # Try removing response format
                            if 'response_format' in api_kwargs:
                                api_kwargs.pop('response_format', None)
                                api_kwargs['temperature'] = original_temperature
                                request_tracker.record_retry("response_format_removals")
                                self.logger.warning(f"[{request_id}] Removing response_format and retrying")
                                continue
                        
                        # All JSON retries exhausted - try the LLM fixer as last resort
                        self._dump_validation_debug(request_id, messages, api_kwargs, content, e, "json",
                                                    provider=provider_name, model_id=model_name, original_model=original_model,
                                                    schema=json_schema)

                        # Attempt JSON fix with LLM if enabled and schema is available
                        if self._json_fixer_enabled and json_schema:
                            fixed_content = await call_json_fixer(
                                elelem_instance=self,
                                invalid_json=content,
                                error=str(e),
                                messages=messages,
                                schema=json_schema,
                                request_id=request_id,
                                fixer_model=self._json_fixer_model
                            )
                            if fixed_content:
                                # Fixer succeeded - use the fixed content
                                content = fixed_content
                                request_tracker.record_retry("json_fixer")
                                # Skip raising error and continue to success path
                            else:
                                # Fixer failed - raise the original error
                                raise ModelError(f"JSON validation failed after all retries and fixer: {e}", provider=provider_name, model=model_name)
                        else:
                            # Fixer disabled or no schema available
                            raise ModelError(f"JSON validation failed after all retries: {e}", provider=provider_name, model=model_name)

                # YAML validation if requested
                if yaml_mode_requested:
                    try:
                        self._validate_yaml_response(content, yaml_schema)
                    except json.JSONDecodeError as e:
                        # YAML validation reuses JSONDecodeError for retry compatibility
                        if attempt < max_retries:
                            # Try temperature reduction
                            if temperature_reductions and len(temperature_reductions) > 0:
                                reduction_idx = min(attempt, len(temperature_reductions) - 1)
                                reduction = temperature_reductions[reduction_idx]
                                new_temp = max(current_temp - reduction, min_temp)

                                if new_temp < current_temp:
                                    api_kwargs['temperature'] = new_temp
                                    request_tracker.record_retry("temperature_reductions")
                                    self.logger.warning(f"[{request_id}] YAML validation failed, reducing temperature to {new_temp}")
                                    self.logger.warning(f"[{request_id}] Validation error: {e}")
                                    self.logger.warning(f"[{request_id}] Failed content:\n{content}")
                                    self._dump_validation_debug(request_id, messages, api_kwargs, content, e, "yaml",
                                                                provider=provider_name, model_id=model_name, original_model=original_model,
                                                                schema=yaml_schema)
                                    continue

                        # All YAML retries exhausted - this is a model error, don't iterate candidates
                        # Note: Don't finalize here - outer loop handles finalization to avoid duplicates
                        self._dump_validation_debug(request_id, messages, api_kwargs, content, e, "yaml",
                                                    provider=provider_name, model_id=model_name, original_model=original_model,
                                                    schema=yaml_schema)
                        raise ModelError(f"YAML validation failed after all retries: {e}", provider=provider_name, model=model_name)

                # Success! Calculate duration and return
                duration = time.time() - start_time

                # Get cost configuration for this candidate
                candidate_cost_config = candidate.get('cost', {})

                # Extract runtime costs if model is configured for runtime pricing
                runtime_costs = self._extract_runtime_costs(response, candidate_cost_config)
                costs = self._calculate_costs(stats_model_name, total_input_tokens, total_output_tokens,
                                            total_reasoning_tokens, runtime_costs, candidate_cost_config)

                # Log success with provider info, tokens, and cost
                provider_info = ""
                if hasattr(response, 'provider') and response.provider:
                    provider_info = f" via {response.provider}"

                # Build token/cost info string
                token_info = f"tokens: {total_input_tokens}→{total_output_tokens}"
                if total_reasoning_tokens > 0:
                    token_info += f" (reasoning: {total_reasoning_tokens})"

                cost_info = ""
                if costs and costs.get('total_cost_usd', 0) > 0:
                    cost_info = f", cost: ${costs['total_cost_usd']:.6f}"

                # Add chunk count for streaming responses
                chunk_info = ""
                if chunk_count is not None:
                    chunk_info = f", chunks: {chunk_count}"

                self.logger.info(f"[{request_id}] ✅ SUCCESS - {candidate_model_name}{provider_info} in {duration:.2f}s | {token_info}{cost_info}{chunk_info}")

                # Finalize request tracking with candidate details and store
                request_tracker.finalize_with_candidate(
                    self._metrics_store,
                    selected_candidate=f"{provider_name}:{candidate_model_name}",
                    actual_model=stats_model_name,
                    actual_provider=runtime_costs.get("actual_provider") if runtime_costs else provider_name,
                    status="success",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    reasoning_tokens=total_reasoning_tokens,
                    total_cost_usd=costs.get('total_cost_usd', 0.0)
                )

                # Update response content
                response.choices[0].message.content = content

                # Add Elelem-specific metrics to the response object
                response.elelem_metrics = {
                    "request_duration_seconds": duration,
                    "provider_used": provider_name,
                    "model_used": stats_model_name,
                    "tokens": {
                        "input": total_input_tokens,
                        "output": total_output_tokens,
                        "reasoning": total_reasoning_tokens,
                        "total": total_input_tokens + total_output_tokens
                    },
                    "costs_usd": costs,
                    "actual_provider": runtime_costs.get("actual_provider") if runtime_costs else None
                }

                # Add reasoning content if present
                if reasoning_content:
                    response.elelem_metrics["reasoning_content"] = reasoning_content
                    response.choices[0].message.reasoning = reasoning_content

                # Cache the successful response using the key computed at the start
                # (before any message preprocessing or kwargs modifications)
                if self.cache and cache and cache_key:
                    self.cache.set(cache_key, original_model, response)

                return response
                
            except (InfrastructureError, ModelError):
                # Re-raise classification errors as-is
                raise
            except RateLimitError as e:
                # Handle OpenAI rate limit errors with dedicated retry logic
                if rate_limit_attempts < max_rate_limit_retries:
                    backoff_times = self.config.retry_settings["rate_limit_backoff"]
                    wait_time = backoff_times[min(rate_limit_attempts, len(backoff_times) - 1)]
                    rate_limit_attempts += 1
                    request_tracker.record_retry("rate_limit_retries")
                    self.logger.warning(f"[{request_id}] Rate limit hit, waiting {wait_time}s (attempt {rate_limit_attempts}/{max_rate_limit_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Rate limit exhaustion is infrastructure issue - try next candidate
                    raise InfrastructureError(f"Rate limit exhausted after {rate_limit_attempts} retries: {e}", provider=provider_name, model=model_name)
            except (AuthenticationError, PermissionDeniedError) as e:
                # Authentication/permission errors are infrastructure issues - try next candidate
                raise InfrastructureError(f"Authentication/permission error: {e}", provider=provider_name, model=model_name)
            except (InternalServerError, BadRequestError, NotFoundError) as e:
                # Server errors, bad requests, and model not found are infrastructure issues
                # (e.g., model might exist on another provider) - try next candidate
                raise InfrastructureError(f"Server/request error: {e}", provider=provider_name, model=model_name)
            except (ConflictError, UnprocessableEntityError) as e:
                # These are request validation issues - don't retry candidate
                raise ModelError(f"Request validation error: {e}", provider=provider_name, model=model_name)
            except Exception as e:
                # Fallback for any other unexpected errors
                # Check if it might be a rate limit that wasn't caught as RateLimitError
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if rate_limit_attempts < max_rate_limit_retries:
                        backoff_times = self.config.retry_settings["rate_limit_backoff"]
                        wait_time = backoff_times[min(rate_limit_attempts, len(backoff_times) - 1)]
                        rate_limit_attempts += 1
                        request_tracker.record_retry("rate_limit_retries")
                        self.logger.warning(f"[{request_id}] Rate limit (fallback), waiting {wait_time}s (attempt {rate_limit_attempts}/{max_rate_limit_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise InfrastructureError(f"Rate limit exhausted after {rate_limit_attempts} retries: {e}", provider=provider_name, model=model_name)

                # Other unexpected errors
                raise ModelError(f"Unexpected error: {e}", provider=provider_name, model=model_name)

        # Should not reach here - if we do, something unexpected happened
        self.logger.error(f"[{request_id}] ⚠️ Unexpected: loop exhausted without resolution (attempt={attempt}, rate_limit_attempts={rate_limit_attempts})")
        raise InfrastructureError("Exhausted all retry attempts for candidate (unexpected state)", provider=provider_name, model=model_name)
    
    def _is_infrastructure_error(self, error) -> bool:
        """Determine if an error is infrastructure-related (should try next candidate)."""
        return is_infrastructure_error(error)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return self._metrics_store.get_stats()

    def get_stats_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get statistics for a specific tag."""
        return self._metrics_store.get_stats(tags=[tag])

    def get_summary(self,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive summary statistics for a time range.

        Args:
            start_time: Filter calls after this time (inclusive). None = no lower bound
            end_time: Filter calls before this time (inclusive). None = no upper bound
            tags: Filter by specific tags. None = all tags

        Returns:
            Dict with aggregated metrics for tokens, costs, duration, and retry analytics
        """
        return self._metrics_store.get_stats(start_time, end_time, tags)

    def get_metrics_dataframe(self,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              tags: Optional[List[str]] = None) -> pd.DataFrame:
        """Get filtered DataFrame for custom metrics analysis.

        Args:
            start_time: Filter calls after this time (inclusive). None = no lower bound
            end_time: Filter calls before this time (inclusive). None = no upper bound
            tags: Filter by specific tags. None = all tags

        Returns:
            Filtered pandas DataFrame with all metrics data
        """
        return self._metrics_store.get_dataframe(start_time, end_time, tags)

    def get_metrics_tags(self) -> List[str]:
        """Get all unique tags from metrics data.

        Returns:
            Sorted list of unique tags including automatic tags like model:* and provider:*
        """
        return self._metrics_store.get_available_tags()

    def list_models(self) -> Dict[str, Any]:
        """List all available models in OpenAI-compatible format.
        
        Returns a response matching OpenAI's GET /v1/models endpoint but with
        an additional 'available' field indicating if the provider's API key is present.
        """
        models_list = []
        
        for model_key, model_config in self._models.items():
            provider = model_config.get("provider", "unknown")

            # Get model metadata from display_metadata
            metadata = model_config.get("display_metadata", {})

            # Handle infrastructure providers for availability check
            provider_config = self.config.providers.get(provider, {})
            base_provider = provider_config.get("base_provider")
            if base_provider:
                env_var = f"{base_provider.upper()}_API_KEY"
            else:
                env_var = f"{provider.upper()}_API_KEY"

            is_available = provider in self._providers

            # Determine if this is a virtual model
            is_virtual = 'candidates' in model_config

            model_entry = {
                "id": model_key,
                "object": "model",
                "created": 1677610602,  # Fixed timestamp like OpenAI
                "owned_by": "elelem" if is_virtual else metadata.get("model_owner", provider),
                "provider": provider,  # Elelem-specific: service provider
                "available": is_available,  # Elelem-specific field
                "model_type": "virtual" if is_virtual else "regular"
            }

            # Add additional metadata if available
            if metadata:
                if "model_nickname" in metadata:
                    model_entry["nickname"] = metadata["model_nickname"]
                if "license" in metadata:
                    model_entry["license"] = metadata["license"]
                if "model_configuration" in metadata:
                    model_entry["model_configuration"] = metadata["model_configuration"]
                else:
                    model_entry["model_configuration"] = "none"
                if "model_page" in metadata:
                    model_entry["model_page"] = metadata["model_page"]

            # Add cost information
            cost_config = model_config.get("cost", {})
            if isinstance(cost_config, dict) and cost_config:
                model_entry["cost"] = {
                    "input_cost_per_1m": cost_config.get("input_cost_per_1m", 0),
                    "output_cost_per_1m": cost_config.get("output_cost_per_1m", 0),
                    "currency": cost_config.get("currency", "USD")
                }

            # Add candidate information for virtual models
            if is_virtual:
                candidates_info = []
                for candidate in model_config['candidates']:
                    if 'model' in candidate:
                        # Reference to another model
                        ref_model_name = candidate['model']
                        ref_config = self._models.get(ref_model_name, {})
                        candidates_info.append({
                            "model": ref_model_name,
                            "provider": ref_config.get("provider", "unknown"),
                            "timeout": candidate.get("timeout")
                        })

                model_entry["candidates"] = candidates_info

            models_list.append(model_entry)
        
        return {
            "object": "list",
            "data": sorted(models_list, key=lambda x: x["id"])
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Elelem and its subsystems.

        Returns:
            Dictionary with health status including metrics backends
        """
        return self._metrics_store.get_health_status()


    def close(self):
        """Clean up resources and close connections."""
        if hasattr(self._metrics_store, 'postgres_engine') and self._metrics_store.postgres_engine:
            try:
                self._metrics_store.postgres_engine.dispose()
            except Exception as e:
                self.logger.error(f"Error closing PostgreSQL connections: {e}")