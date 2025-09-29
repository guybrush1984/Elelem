"""
Main Elelem class - Unified API wrapper for OpenAI, GROQ, and DeepInfra
"""

import asyncio
import json
import logging
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
from ._response_processing import collect_streaming_response, remove_think_tags, extract_json_from_markdown, process_response_content
from ._json_validation import validate_json_schema, is_json_validation_api_error, add_json_instructions_to_messages, validate_json_response
from ._provider_management import create_provider_client, initialize_providers, get_model_config
from ._retry_logic import update_retry_analytics, handle_json_retry, is_infrastructure_error
from ._request_execution import prepare_api_kwargs


class Elelem:
    """Unified API wrapper with cost tracking, JSON validation, and retry logic."""
    
    def __init__(self, metrics_persist_file: Optional[str] = None, extra_provider_dirs: Optional[List[str]] = None):
        self.logger = logging.getLogger("elelem")
        self.config = Config(extra_provider_dirs)
        self._models = self._load_models()
        self._providers = self._initialize_providers()

        # Initialize metrics system (unified SQLAlchemy backend)
        self._metrics_store = MetricsStore()
    
    def _load_models(self) -> Dict[str, Any]:
        """Load model definitions using the Config system."""
        return self.config.models
    
    def _create_provider_client(self, api_key: str, base_url: str, timeout: int = 120, provider_name: str = None, default_headers: Dict = None):
        """Create an OpenAI-compatible client for any provider."""
        return create_provider_client(api_key, base_url, timeout, provider_name, default_headers)
        
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize provider clients."""
        return initialize_providers(self.config.providers, self.config.timeout_seconds, self.logger)
        
    def _reset_stats(self):
        """Reset all statistics."""
        self._metrics_store.reset()
        
    def _get_model_config(self, model: str) -> tuple[str, str]:
        """Get provider and model_id from model configuration (opaque key lookup)."""
        return get_model_config(model, self._models, self._providers)
        
        
    async def _collect_streaming_response(self, stream, request_id=None):
        """Collect streaming chunks and reconstruct a normal response object."""
        return await collect_streaming_response(stream, logger=self.logger, request_id=request_id)
        
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
        
    def _add_json_instructions_to_messages(self, messages: List[Dict[str, str]], capabilities: Dict) -> List[Dict[str, str]]:
        """Add JSON formatting instructions to messages when response_format is JSON."""
        supports_system = capabilities.get("supports_system", True)
        return add_json_instructions_to_messages(messages, supports_system)
        
    def _calculate_costs(self, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, runtime_costs: Dict = None, candidate_cost_config: Dict = None) -> Dict[str, float]:
        """Calculate costs based on model pricing or runtime data from provider."""
        return calculate_costs(model, input_tokens, output_tokens, reasoning_tokens, runtime_costs, candidate_cost_config, self.logger)
    
    def _extract_runtime_costs(self, response, cost_config: str) -> Dict[str, Any]:
        """Extract runtime cost information from response when cost config is 'runtime'."""
        return extract_runtime_costs(response, cost_config, self.logger)
        
    
    
    def _preprocess_messages(self, messages: List[Dict[str, str]], model: str, json_mode_requested: bool, capabilities: Dict) -> List[Dict[str, str]]:
        """Preprocess messages for the request."""
        if json_mode_requested:
            # Always add JSON instructions when JSON is requested
            return self._add_json_instructions_to_messages(messages, capabilities)
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

        self.logger.debug(f"[DEBUG] _cleanup_api_kwargs finished. Final response_format in api_kwargs: {'response_format' in api_kwargs}")

    def _process_response_content(self, response: Any, json_mode_requested: bool) -> str:
        """Process and clean response content."""
        return process_response_content(response, json_mode_requested, self.logger)
    
    def _validate_json_response(self, content: str, json_schema: Any, api_error: Exception) -> None:
        """Validate JSON response content and schema."""
        validate_json_response(content, json_schema, api_error)
    
    
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tags: Union[str, List[str]] = [],
        **kwargs
    ) -> Dict:
        """
        Unified chat completion method matching OpenAI API signature exactly.
        Uses unified candidate-based iteration for both regular and virtual models.
        
        Args:
            messages: List of message dictionaries
            model: Model string in "provider:model" format (or virtual:model)
            tags: Tags for cost tracking (elelem-specific parameter)
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
        except ValueError as e:
            # Record failure for invalid model
            request_tracker.finalize_failure(self._metrics_store, "ModelNotFound", str(e))
            raise ValueError(f"Model configuration error: {e}")
        
        # Extract common parameters
        json_mode_requested = kwargs.get("response_format", {}).get("type") == "json_object"
        json_schema = kwargs.get("json_schema")
        original_temperature = kwargs.get("temperature", 1.0)
        
        # Remove json_schema from kwargs - it's not part of the OpenAI API
        if "json_schema" in kwargs:
            kwargs.pop("json_schema")
        
        # Warn if json_schema provided without JSON response format
        if json_schema and not json_mode_requested:
            self.logger.warning(f"[{request_id}] json_schema provided but response_format is not set to json_object. Schema validation will be skipped.")
        
        self.logger.info(f"[{request_id}] 🚀 Starting {model} with {len(candidates)} candidate(s)")
        if json_mode_requested:
            self.logger.debug(f"[{request_id}] 📋 JSON mode requested")
        
        # Iterate through candidates
        last_error = None
        for candidate_idx, candidate in enumerate(candidates):
            try:
                return await self._attempt_candidate(
                    candidate, candidate_idx + 1, len(candidates),
                    messages, model, model_config, request_id,
                    json_mode_requested, json_schema, original_temperature,
                    tags, start_time, request_tracker, **kwargs
                )
            except InfrastructureError as e:
                self.logger.warning(f"[{request_id}] 🔄 Candidate {candidate_idx + 1} failed: {e}")
                request_tracker.record_retry("candidate_iterations")
                last_error = e

                if candidate_idx < len(candidates) - 1:
                    continue  # Try next candidate
                else:
                    # All candidates exhausted - record failure
                    request_tracker.finalize_failure(self._metrics_store, "AllCandidatesFailed", str(e))
                    raise e
            except ModelError as e:
                # Model/request errors don't trigger candidate iteration
                request_tracker.finalize_failure(self._metrics_store, "ModelError", str(e))
                raise e

        # This should never be reached - if it is, there's a logic error
        raise RuntimeError(f"FATAL: Candidate loop completed without returning or raising - this is a bug in Elelem")
    
    
    async def _attempt_candidate(self, candidate, candidate_idx, total_candidates,
                                messages, original_model, model_config, request_id,
                                json_mode_requested, json_schema, original_temperature,
                                tags, start_time, request_tracker, **kwargs):
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
        
        # Preprocess messages for JSON mode if needed
        modified_messages = self._preprocess_messages(messages, candidate_model_name, json_mode_requested, capabilities)

        # Retry logic for this candidate (temperature reduction, JSON validation)
        max_retries = self.config.retry_settings["max_json_retries"]
        max_rate_limit_retries = self.config.retry_settings["max_rate_limit_retries"]
        temperature_reductions = self.config.retry_settings["temperature_reductions"]
        min_temp = self.config.retry_settings["min_temp"]
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        
        for attempt in range(max_retries + 1):
            try:
                current_temp = api_kwargs.get('temperature', original_temperature)
                self.logger.debug(f"[{request_id}] 🔄 Attempt {attempt + 1}/{max_retries + 1} - temp={current_temp}")
                
                # Make API call with timeout
                try:
                    if api_kwargs.get("stream", False):
                        # Handle streaming response
                        stream = await asyncio.wait_for(
                            provider_client.chat.completions.create(
                                messages=modified_messages,
                                model=model_name,
                                **api_kwargs
                            ),
                            timeout=timeout
                        )
                        response = await self._collect_streaming_response(stream, request_id)
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
                    raise InfrastructureError(f"Request timed out after {timeout}s")
                except RateLimitError:
                    # Let rate limit errors pass through to outer handler for proper retry logic
                    raise
                except Exception as api_e:
                    # Classify the error
                    if self._is_infrastructure_error(api_e):
                        raise InfrastructureError(f"API infrastructure error: {str(api_e)}")
                    elif self._is_json_validation_api_error(api_e) and json_mode_requested:
                        # API-level JSON validation errors will be handled in JSON validation section
                        api_error = api_e
                        response = None
                    else:
                        # Other API errors are typically model errors (don't iterate)
                        raise ModelError(f"API error: {str(api_e)}")
                
                # Extract tokens from response
                if response:
                    # Extract normalized token counts
                    input_tokens, output_tokens, reasoning_tokens, total_tokens = extract_token_counts(response, self.logger)

                    # Extract reasoning content
                    reasoning_content = extract_reasoning_content(response, self.logger)

                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_reasoning_tokens += reasoning_tokens

                    content = self._process_response_content(response, json_mode_requested)
                else:
                    content = ""
                    reasoning_content = None
                
                # JSON validation if requested
                if json_mode_requested:
                    try:
                        self._validate_json_response(content, json_schema, api_error)
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
                                    error_snippet = str(e)[:300]
                                    self.logger.warning(f"[{request_id}] JSON validation failed, reducing temperature to {new_temp} (error: {error_snippet})")
                                    continue
                            
                            # Try removing response format
                            if 'response_format' in api_kwargs:
                                api_kwargs.pop('response_format', None)
                                api_kwargs['temperature'] = original_temperature
                                request_tracker.record_retry("response_format_removals")
                                self.logger.warning(f"[{request_id}] Removing response_format and retrying")
                                continue
                        
                        # All JSON retries exhausted - this is a model error, don't iterate candidates
                        request_tracker.finalize_failure(self._metrics_store, "JSONValidationFailed", str(e))
                        raise ModelError(f"JSON validation failed after all retries: {e}")
                
                # Success! Calculate duration and return
                duration = time.time() - start_time
                
                # Log success with provider info
                provider_info = ""
                if hasattr(response, 'provider') and response.provider:
                    provider_info = f" via {response.provider}"
                
                self.logger.info(f"[{request_id}] ✅ SUCCESS - {candidate_model_name}{provider_info} in {duration:.2f}s")

                # Get cost configuration for this candidate
                candidate_cost_config = candidate.get('cost', {})

                # Extract runtime costs if model is configured for runtime pricing
                runtime_costs = self._extract_runtime_costs(response, candidate_cost_config)
                costs = self._calculate_costs(stats_model_name, total_input_tokens, total_output_tokens,
                                            total_reasoning_tokens, runtime_costs, candidate_cost_config)

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

                return response
                
            except (InfrastructureError, ModelError):
                # Re-raise classification errors as-is
                raise
            except RateLimitError as e:
                # Handle OpenAI rate limit errors with dedicated retry logic
                if attempt < max_rate_limit_retries:
                    backoff_times = self.config.retry_settings["rate_limit_backoff"]
                    wait_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                    request_tracker.record_retry("rate_limit_retries")
                    self.logger.warning(f"[{request_id}] Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_rate_limit_retries + 1})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Rate limit exhaustion is infrastructure issue - try next candidate
                    raise InfrastructureError(f"Rate limit exhausted after {max_rate_limit_retries + 1} attempts: {e}")
            except (AuthenticationError, PermissionDeniedError) as e:
                # Authentication/permission errors are infrastructure issues - try next candidate
                raise InfrastructureError(f"Authentication/permission error: {e}")
            except (InternalServerError, BadRequestError) as e:
                # Server errors and bad requests are infrastructure issues - try next candidate
                raise InfrastructureError(f"Server/request error: {e}")
            except (NotFoundError, ConflictError, UnprocessableEntityError) as e:
                # These are typically model/request issues - don't retry candidate
                raise ModelError(f"Model/request error: {e}")
            except Exception as e:
                # Fallback for any other unexpected errors
                # Check if it might be a rate limit that wasn't caught as RateLimitError
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < max_rate_limit_retries:
                        backoff_times = self.config.retry_settings["rate_limit_backoff"]
                        wait_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                        request_tracker.record_retry("rate_limit_retries")
                        self.logger.warning(f"[{request_id}] Rate limit (fallback), waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise InfrastructureError(f"Rate limit exhausted: {e}")

                # Other unexpected errors
                raise ModelError(f"Unexpected error: {e}")
        
        # Should not reach here
        raise ModelError("Exhausted all attempts for candidate")
    
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
        return self._metrics_store.get_unique_tags()

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