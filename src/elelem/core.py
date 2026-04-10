"""
Main Elelem class - Unified API wrapper for OpenAI, GROQ, and DeepInfra
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import openai
from openai import (
    BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError,
    ConflictError, UnprocessableEntityError, RateLimitError, InternalServerError,
)
from .config import Config
from .metrics import MetricsStore
from ._reasoning_tokens import extract_token_counts, extract_reasoning_content
from ._exceptions import InfrastructureError, ModelError, JsonSchemaError, TooSlowError
from ._cost_calculation import calculate_costs, extract_runtime_costs
from ._response_processing import collect_streaming_response, ChunkTimeoutError, StreamingAbortError, process_response_content
from ._json_validation import is_json_validation_api_error  # Still needed for API error detection
from ._provider_management import create_provider_client, get_model_config
from ._retry_logic import update_retry_analytics, handle_json_retry, is_infrastructure_error
from ._request_execution import prepare_api_kwargs
from ._benchmark_store import reorder_candidates_by_benchmark
from ._dynamic_routing import DynamicRoutingStore
from ._output_formats import FormatRegistry, FormatParseError, FormatSchemaError, OutputFormat
from ._format_fixer import call_format_fixer
from ._request_id import generate_request_id
from ._request_state import RequestState, RequestContext
from ._request_handlers import RequestStateMachine


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
        self._token_providers = {}  # provider_name -> TokenProvider (for cloud auth)

        # JSON fixer configuration
        self._json_fixer_enabled = json_fixer_enabled
        self._json_fixer_model = json_fixer_model

        # Initialize metrics system (unified SQLAlchemy backend)
        self._metrics_store = MetricsStore()

        # Initialize dynamic routing store (for performance-based provider selection)
        self._dynamic_routing_store = DynamicRoutingStore(
            metrics_store=self._metrics_store
        )

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
        from ._token_providers import create_token_provider

        # Create token provider (handles both static keys and cloud auth)
        token_provider = create_token_provider(provider_name, provider_config, self.logger)
        if not token_provider:
            self._failed_providers[provider_name] = (time.time(), False)
            return False

        self._token_providers[provider_name] = token_provider
        api_key = token_provider.get_token()

        # Determine endpoint (token provider may provide one, e.g., cloud providers)
        endpoint = token_provider.get_endpoint()
        retryable = True
        probe_timeout = provider_config.get("probe_timeout", 5.0)

        if endpoint:
            # Token provider specified endpoint - use its probe method
            self.logger.info(f"[{provider_name}] Probing endpoint: {endpoint[:60]}...")
            result = token_provider.probe(endpoint, probe_timeout, self.logger)
            if not result.success:
                self.logger.warning(f"[{provider_name}] ❌ Endpoint not accessible (retryable={result.retryable})")
                self._failed_providers[provider_name] = (time.time(), result.retryable)
                return False
            self.logger.info(f"[{provider_name}] ✅ Endpoint accessible")

        elif "endpoints" in provider_config:
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
        client_type = provider_config.get("client_type")
        self._providers[provider_name] = create_provider_client(
            api_key=api_key,
            base_url=endpoint,
            timeout=self.config.timeout_seconds,
            provider_name=provider_name,
            default_headers=custom_headers,
            client_type=client_type
        )
        self.logger.debug(f"[{provider_name}] Initialized successfully")
        return True

    def _reset_stats(self):
        """Reset all statistics."""
        self._metrics_store.reset()
        
    def _get_model_config(self, model: str) -> tuple[str, str]:
        """Get provider and model_id from model configuration (opaque key lookup)."""
        return get_model_config(model, self._models, self._providers)
        
        
    async def _collect_streaming_response(self, stream, request_id=None, chunk_timeout=None, format_name=None, min_tps=None, min_tps_eval_window=10):
        """Collect streaming chunks and reconstruct a normal response object.

        Args:
            stream: Async stream of chunks
            request_id: Request ID for logging
            chunk_timeout: Optional timeout (seconds) for receiving each chunk.
                          If no chunk arrives within this time, raises ChunkTimeoutError.
            format_name: Optional expected format ('json', 'yaml', 'csv') for early abort.
            min_tps: Optional minimum tokens/sec — abort if too slow after eval window.
            min_tps_eval_window: Seconds before evaluating tps (default: 10).

        Returns:
            Tuple of (response, chunk_count)
        """
        return await collect_streaming_response(stream, logger=self.logger, request_id=request_id, chunk_timeout=chunk_timeout, format_name=format_name, min_tps=min_tps, min_tps_eval_window=min_tps_eval_window)
        
    def _is_json_validation_api_error(self, error: Exception) -> bool:
        """Check if the error is a json_validate_failed API error."""
        return is_json_validation_api_error(error)

    def _calculate_costs(self, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, runtime_costs: Dict = None, candidate_cost_config: Dict = None) -> Dict[str, float]:
        """Calculate costs based on model pricing or runtime data from provider."""
        return calculate_costs(model, input_tokens, output_tokens, reasoning_tokens, runtime_costs, candidate_cost_config, self.logger)

    def _extract_runtime_costs(self, response, cost_config: str) -> Dict[str, Any]:
        """Extract runtime cost information from response when cost config is 'runtime'."""
        return extract_runtime_costs(response, cost_config, self.logger)
    
    def _cleanup_api_kwargs(self, api_kwargs: Dict, model: str, model_config: Dict) -> None:
        """Remove unsupported parameters from api_kwargs based on model capabilities."""
        capabilities = model_config.get("capabilities", {})

        # Remove response_format if model doesn't support JSON mode
        if not capabilities.get("supports_json_mode", True) and "response_format" in api_kwargs:
            self.logger.debug(f"Removing response_format for {model} (not supported)")
            api_kwargs.pop("response_format")

        # Remove temperature if not supported
        if not capabilities.get("supports_temperature", True) and "temperature" in api_kwargs:
            self.logger.debug(f"Removing temperature for {model} (not supported)")
            api_kwargs.pop("temperature")

        # Remove Elelem-specific parameters that should not be passed to provider APIs
        for param in ["enforce_schema_in_prompt", "yaml_schema", "csv_schema"]:
            api_kwargs.pop(param, None)

    def _process_response_content(self, response: Any, format_handler: OutputFormat = None) -> str:
        """Process and clean response content."""
        return process_response_content(response, self.logger, format_handler)

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
        # Generate human-readable request ID for tracking
        request_id = generate_request_id()
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
                    selected_candidate=None,  # Don't pollute routing stats with cache hits
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
        
        # Extract Elelem-specific parameters early (needed for routing decisions)
        min_tps = kwargs.pop("min_tps", None)
        min_tps_eval_window = kwargs.pop("min_tps_eval_window", 10)

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
                # Use the higher of routing config threshold and per-request min_tps
                config_min_tps = routing.get('min_tokens_per_sec', 0.0)
                effective_min_tokens = max(config_min_tps, min_tps or 0.0)

                # Get dynamic stats for routing decisions
                dynamic_stats = self._dynamic_routing_store.get_dynamic_stats()

                # Get failed candidates in cooldown (excluded from routing)
                failed_candidates = self._dynamic_routing_store.get_failed_candidates()

                candidates = reorder_candidates_by_benchmark(
                    candidates,
                    speed_weight=speed_weight,
                    min_tokens_per_sec=effective_min_tokens,
                    dynamic_stats=dynamic_stats,
                    failed_candidates=failed_candidates,
                    logger=self.logger,
                    request_id=request_id
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

        # Detect CSV mode request (Elelem-specific, client-side only)
        csv_schema = kwargs.get("csv_schema")
        csv_mode_requested = csv_schema is not None

        # Validate mutual exclusivity between JSON, YAML, and CSV
        active_formats = sum([json_mode_requested, yaml_mode_requested, csv_mode_requested])
        if active_formats > 1:
            raise ValueError(
                "Cannot use multiple output formats simultaneously. "
                "Provide only one of: response_format (json_object/json_schema), yaml_schema, or csv_schema."
            )

        # Resolve format handler for new abstraction layer
        format_handler: Optional[OutputFormat] = None
        format_schema = None
        if json_mode_requested:
            format_handler = FormatRegistry.get("json")
            format_schema = json_schema
        elif yaml_mode_requested:
            format_handler = FormatRegistry.get("yaml")
            format_schema = yaml_schema
        elif csv_mode_requested:
            format_handler = FormatRegistry.get("csv")
            format_schema = csv_schema
            self.logger.debug(f"[{request_id}] CSV format requested (client-side validation)")

        # Get original temperature
        original_temperature = kwargs.get("temperature", 1.0)

        # Remove Elelem-specific parameters (if present)
        if "json_schema" in kwargs:
            kwargs.pop("json_schema")
        if "yaml_schema" in kwargs:
            kwargs.pop("yaml_schema")
        if "csv_schema" in kwargs:
            kwargs.pop("csv_schema")

        # Warn if json_schema provided without JSON response format
        if json_schema and not json_mode_requested:
            self.logger.warning(
                f"[{request_id}] json_schema provided but response_format is not set to json_object or json_schema. "
                "Schema validation will be skipped."
            )
        
        # Build list of candidate providers for logging (with routing info)
        # Format: provider(Nx, XXXt/s, YYYv) where N = samples, v = value score (speed^weight/cost)
        def format_candidate(c):
            provider = c.get('provider')
            tps = c.get('_tps')
            value_score = c.get('_value_score')
            sample_count = c.get('_sample_count', 0)

            if tps is not None and value_score is not None and value_score > 0:
                # Show both speed and value score (value = speed^weight / cost)
                return f"{provider}({sample_count}x, {tps:.0f}t/s, {value_score:.0f}v)"
            elif tps is not None and tps > 0:
                return f"{provider}({sample_count}x, {tps:.0f}t/s)"
            elif sample_count > 0:
                return f"{provider}({sample_count}x)"
            return provider

        candidate_info = [format_candidate(c) for c in candidates[:5]]
        if len(candidates) > 5:
            candidate_info.append(f"+{len(candidates)-5}")
        self.logger.info(f"[{request_id}] 🚀 {model} → [{', '.join(candidate_info)}] (temp={original_temperature})")
        if json_mode_requested:
            self.logger.debug(f"[{request_id}] 📋 JSON mode requested")
        if yaml_mode_requested:
            self.logger.debug(f"[{request_id}] 📄 YAML mode requested")
        if csv_mode_requested:
            self.logger.debug(f"[{request_id}] 📊 CSV mode requested")
        
        # Iterate through candidates
        # Track failed model_references to skip candidates with the same underlying model
        failed_model_refs = set()
        last_error = None
        cumulative_path = []  # Track full journey across all candidates

        for candidate_idx, candidate in enumerate(candidates):
            # Skip candidates whose model_reference has already failed with ModelError
            candidate_model_ref = candidate.get('model_reference')
            if candidate_model_ref and candidate_model_ref in failed_model_refs:
                self.logger.debug(f"[{request_id}] Skipping candidate {candidate_idx + 1} (model_reference '{candidate_model_ref}' already failed)")
                continue

            try:
                # Skip min_tps for: single candidate (no fallback) or exploration picks (measuring speed)
                if len(candidates) <= 1 or candidate.get('_skip_min_tps'):
                    effective_min_tps = None
                else:
                    effective_min_tps = min_tps
                return await self._attempt_candidate(
                    candidate, candidate_idx + 1, len(candidates),
                    messages, model, model_config, request_id,
                    format_handler, format_schema,
                    original_temperature, tags, cache, cache_key, start_time, request_tracker,
                    cumulative_path=cumulative_path, min_tps=effective_min_tps, min_tps_eval_window=min_tps_eval_window, **kwargs
                )
            except TooSlowError as e:
                self.logger.warning(f"[{request_id}] 🐢 Candidate {candidate_idx + 1} too slow ({e.observed_tps:.1f} tps): {e}")
                cumulative_path.append("NEXT_CANDIDATE")
                request_tracker.record_retry("candidate_iterations")
                last_error = e
                # No cooldown — provider works fine, just too slow for this request
                # Record partial tps observation so dynamic routing learns the speed
                candidate_ref = candidate.get('original_model_ref')
                if candidate_ref and self._metrics_store:
                    self._record_partial_tps(
                        request_id, candidate_ref, candidate.get('provider', ''),
                        candidate.get('model_id', ''), model,
                        e.observed_tps, e.elapsed, tags
                    )
                continue
            except InfrastructureError as e:
                self.logger.warning(f"[{request_id}] 🔄 Candidate {candidate_idx + 1} failed (infra): {e}")
                cumulative_path.append("NEXT_CANDIDATE")
                request_tracker.record_retry("candidate_iterations")
                last_error = e
                # Mark candidate as failed for cooldown (only for virtual models with routing)
                if candidate.get('original_model_ref'):
                    self._dynamic_routing_store.mark_failed(candidate['original_model_ref'])
                # Continue to next candidate (infrastructure errors don't blacklist the model)
                continue
            except ModelError as e:
                # Model errors: skip all remaining candidates with the same model_reference
                if candidate_model_ref:
                    failed_model_refs.add(candidate_model_ref)
                    self.logger.warning(f"[{request_id}] 🔄 Candidate {candidate_idx + 1} failed (model): {e} - skipping model_reference '{candidate_model_ref}'")
                    cumulative_path.append("SKIP_MODEL")
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
                                format_handler: Optional[OutputFormat], format_schema: Optional[Dict],
                                original_temperature, tags, cache, cache_key, start_time, request_tracker,
                                cumulative_path: list = None, min_tps: Optional[float] = None, min_tps_eval_window: int = 10, **kwargs):
        """Attempt to complete request with a specific candidate.

        Uses the RequestStateMachine to process the request through explicit states:
            CALL_API → EXTRACT_TOKENS → VALIDATE_FORMAT → TRY_FIXER → REDUCE_TEMPERATURE

        Terminal states:
            SUCCESS → return response
            NEXT_CANDIDATE → raise InfrastructureError (try next provider)
            SKIP_MODEL → raise ModelError (skip same model_reference)

        Args:
            format_handler: OutputFormat handler (json, yaml, csv) or None if no format requested
            format_schema: Schema for validation, or None
            cumulative_path: List to accumulate state path across all candidate attempts
        """
        if cumulative_path is None:
            cumulative_path = []
        # === Setup phase (unchanged) ===

        # Get timeout for this candidate
        timeout = self.config.get_candidate_timeout(candidate, model_config)
        chunk_timeout = self.config.get_candidate_chunk_timeout(candidate, model_config)

        # Setup provider and model for this candidate
        provider_name = candidate['provider']
        model_name = candidate['model_id']
        provider_client = self._providers[provider_name]
        capabilities = candidate.get('capabilities', {})

        # Refresh token if needed (for cloud providers with expiring tokens)
        if provider_name in self._token_providers:
            token = self._token_providers[provider_name].get_token()
            provider_client.api_key = token

        # Use original model reference for statistics (cost lookup)
        stats_model_name = candidate.get('original_model_ref', original_model)
        candidate_model_name = f"{provider_name}:{model_name}"

        self.logger.info(f"[{request_id}] 🎯 Candidate {candidate_idx}/{total_candidates}: {candidate_model_name} (timeout={timeout}s)")

        # Get provider configuration for defaults
        provider_config = self.config.get_provider_config(provider_name)

        # Prepare API parameters with proper precedence
        api_kwargs = prepare_api_kwargs(kwargs, original_temperature, provider_config,
                                      candidate, stats_model_name, self.config, provider_name)

        # Clean up unsupported parameters for this candidate model
        candidate_key = candidate.get('model', f"{provider_name}:{model_name}")
        self._cleanup_api_kwargs(api_kwargs, candidate_key, {'capabilities': capabilities})

        # Preprocess messages for structured output formats
        enforce_schema_in_prompt = kwargs.get('enforce_schema_in_prompt', False)
        if format_handler:
            supports_system = capabilities.get("supports_system", True)
            schema_for_prompt = format_schema if enforce_schema_in_prompt else None
            modified_messages = format_handler.add_instructions_to_messages(
                messages, schema_for_prompt, supports_system
            )
        else:
            modified_messages = messages

        # === Build request context for state machine ===
        ctx = RequestContext(
            # Request info
            request_id=request_id,
            messages=modified_messages,
            original_model=original_model,
            format_handler=format_handler,
            format_schema=format_schema,
            original_temperature=original_temperature,

            # Candidate info
            provider_name=provider_name,
            model_name=model_name,
            candidate=candidate,
            timeout=timeout,
            chunk_timeout=chunk_timeout,
            min_tps=min_tps,
            min_tps_eval_window=min_tps_eval_window,
            capabilities=capabilities,
            api_kwargs=api_kwargs,
            provider_client=provider_client,
            stats_model_name=stats_model_name,

            # Mutable state
            current_temperature=api_kwargs.get('temperature', original_temperature),

            # Configuration
            max_retries=self.config.retry_settings["max_json_retries"],
            max_rate_limit_retries=self.config.retry_settings["max_rate_limit_retries"],
            temperature_reductions=self.config.retry_settings["temperature_reductions"],
            min_temp=self.config.retry_settings["min_temp"],
            rate_limit_backoff=self.config.retry_settings["rate_limit_backoff"],

            # Tracking
            request_tracker=request_tracker,
        )

        # === Run state machine ===
        state_machine = RequestStateMachine(self)
        result = await state_machine.run(ctx)

        # Extend cumulative path with this candidate's journey (excluding terminal state for non-success)
        if result.path:
            # For SUCCESS, include full path; for errors, exclude terminal state (will be added by caller)
            if result.next_state == RequestState.SUCCESS:
                cumulative_path.extend(result.path)
            else:
                cumulative_path.extend(result.path[:-1])  # Exclude NEXT_CANDIDATE/SKIP_MODEL

        # === Handle terminal states ===
        if result.next_state == RequestState.SUCCESS:
            return self._build_success_response(
                ctx, start_time, cache, cache_key, original_model, candidate_model_name,
                state_path=cumulative_path
            )
        elif result.next_state == RequestState.NEXT_CANDIDATE:
            raise result.error
        elif result.next_state == RequestState.SKIP_MODEL:
            raise result.error
        else:
            # Should never happen
            raise RuntimeError(f"Unexpected terminal state: {result.next_state}")

    def _build_success_response(self, ctx: RequestContext, start_time: float,
                                cache: bool, cache_key: str, original_model: str,
                                candidate_model_name: str, state_path: list = None):
        """Build the success response from context after state machine completes.

        This handles:
        - Cost calculation
        - Logging
        - Metrics finalization
        - Response augmentation
        - Caching
        """
        duration = time.time() - start_time
        response = ctx.response

        # Get cost configuration for this candidate
        candidate_cost_config = ctx.candidate.get('cost', {})

        # Extract runtime costs if model is configured for runtime pricing
        runtime_costs = self._extract_runtime_costs(response, candidate_cost_config)
        costs = self._calculate_costs(
            ctx.stats_model_name,
            ctx.total_input_tokens,
            ctx.total_output_tokens,
            ctx.total_reasoning_tokens,
            runtime_costs,
            candidate_cost_config
        )

        # Log success with provider info, tokens, and cost
        provider_info = ""
        if hasattr(response, 'provider') and response.provider:
            provider_info = f" via {response.provider}"

        token_info = f"tokens: {ctx.total_input_tokens}→{ctx.total_output_tokens}"
        if ctx.total_reasoning_tokens > 0:
            token_info += f" (reasoning: {ctx.total_reasoning_tokens})"

        cost_info = ""
        if costs and costs.get('total_cost_usd', 0) > 0:
            cost_info = f", cost: ${costs['total_cost_usd']:.6f}"

        chunk_info = ""
        if ctx.chunk_count is not None:
            chunk_info = f", chunks: {ctx.chunk_count}"

        # Format state path (compact: only show non-happy-path or abbreviated)
        path_info = ""
        if state_path:
            path_str = "→".join(state_path)
            path_info = f" [{path_str}]"

        self.logger.info(
            f"[{ctx.request_id}] ✅ SUCCESS{path_info} - {candidate_model_name}{provider_info} "
            f"in {duration:.2f}s | {token_info}{cost_info}{chunk_info}"
        )

        # Finalize request tracking
        ctx.request_tracker.finalize_with_candidate(
            self._metrics_store,
            selected_candidate=ctx.stats_model_name,
            actual_model=ctx.stats_model_name,
            actual_provider=runtime_costs.get("actual_provider") if runtime_costs else ctx.provider_name,
            status="success",
            input_tokens=ctx.total_input_tokens,
            output_tokens=ctx.total_output_tokens,
            reasoning_tokens=ctx.total_reasoning_tokens,
            total_cost_usd=costs.get('total_cost_usd', 0.0)
        )

        # Update response content
        response.choices[0].message.content = ctx.content

        # Add Elelem-specific metrics to the response object
        response.elelem_metrics = {
            "request_duration_seconds": duration,
            "provider_used": ctx.provider_name,
            "model_used": ctx.stats_model_name,
            "tokens": {
                "input": ctx.total_input_tokens,
                "output": ctx.total_output_tokens,
                "reasoning": ctx.total_reasoning_tokens,
                "total": ctx.total_input_tokens + ctx.total_output_tokens
            },
            "costs_usd": costs,
            "actual_provider": runtime_costs.get("actual_provider") if runtime_costs else None
        }

        # Add reasoning content if present
        if ctx.reasoning_content:
            response.elelem_metrics["reasoning_content"] = ctx.reasoning_content
            response.choices[0].message.reasoning = ctx.reasoning_content

        # Cache the successful response
        if self.cache and cache and cache_key:
            self.cache.set(cache_key, original_model, response)

        return response
    
    def _record_partial_tps(self, request_id, candidate_ref, provider, model_id,
                            requested_model, observed_tps, elapsed, tags):
        """Record a partial tps observation from an aborted-too-slow stream.

        Writes a 'success' record with the observed tps so the dynamic routing
        store picks up the speed for future candidate ranking.
        """
        try:
            from .metrics import RequestTracker
            partial = RequestTracker(request_id=f"{request_id}-tps-{provider}")
            partial.requested_model = requested_model
            partial.selected_candidate = candidate_ref
            partial.actual_provider = provider
            partial.actual_model = model_id
            partial.output_tokens = int(observed_tps * elapsed)
            partial.llm_start_time = partial.start_time
            partial.llm_end_time = partial.start_time + elapsed
            partial.tags = tags or []
            partial.finalize(status="success")
            self._metrics_store.finalize_request(partial)
            self.logger.debug(f"[{request_id}] 📊 Recorded partial tps: {candidate_ref} @ {observed_tps:.1f} t/s")
        except Exception as e:
            self.logger.warning(f"[{request_id}] Failed to record partial tps: {e}")

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

    def get_metrics_data(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get filtered metrics data for custom analysis.

        Args:
            start_time: Filter calls after this time (inclusive). None = no lower bound
            end_time: Filter calls before this time (inclusive). None = no upper bound
            tags: Filter by specific tags. None = all tags

        Returns:
            Filtered list of dicts with all metrics data
        """
        return self._metrics_store.get_data(start_time, end_time, tags)

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
            Dictionary with health status including metrics backends and dynamic routing
        """
        health = self._metrics_store.get_health_status()

        # Add dynamic routing status
        health["dynamic_routing"] = {
            "enabled": self._dynamic_routing_store.enabled,
            "cache_ttl": self._dynamic_routing_store._cache_ttl,
            "cached_models": len(self._dynamic_routing_store._cache),
        }

        # Add routing statistics
        from ._benchmark_store import get_routing_stats
        health["routing_stats"] = get_routing_stats()

        return health


    def close(self):
        """Clean up resources and close connections."""
        if hasattr(self._metrics_store, 'postgres_engine') and self._metrics_store.postgres_engine:
            try:
                self._metrics_store.postgres_engine.dispose()
            except Exception as e:
                self.logger.error(f"Error closing PostgreSQL connections: {e}")