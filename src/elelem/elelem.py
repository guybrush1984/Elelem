"""
Main Elelem class - Unified API wrapper for OpenAI, GROQ, and DeepInfra
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, List, Optional, Union, Any
from .models import MODELS
from .config import Config
from .providers import create_provider_client


class Elelem:
    """Unified API wrapper with cost tracking, JSON validation, and retry logic."""
    
    def __init__(self):
        self.logger = logging.getLogger("elelem")
        self.config = Config()
        self._statistics = {}
        self._tag_statistics = {}
        self._providers = self._initialize_providers()
        
        # Initialize statistics tracking
        self._reset_stats()
        
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize provider clients."""
        providers = {}
        
        # Initialize all configured providers
        for provider_name, provider_config in self.config.providers.items():
            env_var = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(env_var)
            
            if api_key:
                providers[provider_name] = create_provider_client(
                    api_key=api_key,
                    base_url=provider_config["endpoint"],
                    timeout=self.config.timeout_seconds
                )
                self.logger.debug(f"Initialized {provider_name} provider")
            else:
                self.logger.warning(f"No API key found for {provider_name} (env var: {env_var})")
                
        return providers
        
    def _reset_stats(self):
        """Reset all statistics."""
        self._statistics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_input_cost_usd": 0.0,
            "total_output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_duration_seconds": 0.0,
            "avg_duration_seconds": 0.0,
            "reasoning_tokens": 0,
            "reasoning_cost_usd": 0.0
        }
        self._tag_statistics = {}
        
    def _parse_model_string(self, model: str) -> tuple[str, str]:
        """Parse provider:model string."""
        if ":" not in model:
            raise ValueError(f"Model must be in 'provider:model' format, got: {model}")
            
        provider, model_name = model.split(":", 1)
        
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")
            
        if provider not in self._providers:
            available_providers = list(self._providers.keys())
            raise ValueError(f"Provider '{provider}' not available. Available: {available_providers}")
            
        return provider, model_name
        
    def _should_remove_response_format(self, model: str) -> bool:
        """Check if response_format should be removed for this model."""
        model_config = MODELS.get(model, {})
        return not model_config.get("capabilities", {}).get("supports_json_mode", True)
        
    def _remove_think_tags(self, content: str) -> str:
        """Remove <think>...</think> tags from content."""
        # Pattern to match <think>...</think> including multiline content
        pattern = r'<think>.*?</think>'
        return re.sub(pattern, '', content, flags=re.DOTALL).strip()
        
    def _extract_json_from_markdown(self, content: str) -> str:
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
                self.logger.debug(f"Extracted JSON from markdown: {extracted}")
                return extracted
                
        # If no markdown wrapper found, return original content
        return content
        
    def _is_json_validation_api_error(self, error: Exception) -> bool:
        """Check if the error is a json_validate_failed API error."""
        error_str = str(error).lower()
        return "json_validate_failed" in error_str and "400" in error_str
        
    def _add_json_instructions_to_messages(self, messages: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
        """Add JSON formatting instructions to messages when response_format is JSON."""
        model_config = MODELS.get(model, {})
        supports_system = model_config.get("capabilities", {}).get("supports_system", True)
        
        modified_messages = messages.copy()
        
        # Add JSON instruction to system message or create one
        json_instruction = (
            "\n\nCRITICAL: You must respond with ONLY a clean JSON object - no markdown, no code blocks, no extra text. "
            "Do not wrap the JSON in ```json``` blocks or any other formatting. "
            "Return raw, valid JSON that can be parsed directly. "
            "Start your response with { and end with }. "
            "Any non-JSON content will cause a parsing error."
        )
        
        if supports_system:
            # Find system message and append instruction
            system_found = False
            for msg in modified_messages:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + json_instruction
                    system_found = True
                    break
                    
            # If no system message found, add one
            if not system_found:
                modified_messages.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant." + json_instruction
                })
        else:
            # Model doesn't support system messages, add as user message
            modified_messages.append({
                "role": "user",
                "content": json_instruction.strip()
            })
            
        return modified_messages
        
    def _calculate_costs(self, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> Dict[str, float]:
        """Calculate costs based on model pricing."""
        model_config = MODELS.get(model, {})
        cost_config = model_config.get("cost", {})
        
        input_cost_per_1m = cost_config.get("input_cost_per_1m", 0.0)
        output_cost_per_1m = cost_config.get("output_cost_per_1m", 0.0)
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        # Reasoning tokens count as output cost
        total_output_tokens = output_tokens + reasoning_tokens
        output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
        
        return {
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "reasoning_cost_usd": (reasoning_tokens / 1_000_000) * output_cost_per_1m,
            "total_cost_usd": input_cost + output_cost
        }
        
    def _update_statistics(self, model: str, input_tokens: int, output_tokens: int, 
                          reasoning_tokens: int, duration: float, tags: List[str]):
        """Update statistics tracking."""
        costs = self._calculate_costs(model, input_tokens, output_tokens, reasoning_tokens)
        
        # Update overall statistics
        self._statistics["total_input_tokens"] += input_tokens
        self._statistics["total_output_tokens"] += output_tokens
        self._statistics["total_tokens"] += input_tokens + output_tokens
        self._statistics["total_input_cost_usd"] += costs["input_cost_usd"]
        self._statistics["total_output_cost_usd"] += costs["output_cost_usd"]
        self._statistics["total_cost_usd"] += costs["total_cost_usd"]
        self._statistics["total_calls"] += 1
        self._statistics["total_duration_seconds"] += duration
        self._statistics["avg_duration_seconds"] = (
            self._statistics["total_duration_seconds"] / self._statistics["total_calls"]
        )
        
        if reasoning_tokens > 0:
            self._statistics["reasoning_tokens"] += reasoning_tokens
            self._statistics["reasoning_cost_usd"] += costs["reasoning_cost_usd"]
            
        # Update tag-specific statistics
        for tag in tags:
            if tag not in self._tag_statistics:
                self._tag_statistics[tag] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "total_input_cost_usd": 0.0,
                    "total_output_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                    "total_calls": 0,
                    "total_duration_seconds": 0.0,
                    "avg_duration_seconds": 0.0,
                    "reasoning_tokens": 0,
                    "reasoning_cost_usd": 0.0
                }
                
            tag_stats = self._tag_statistics[tag]
            tag_stats["total_input_tokens"] += input_tokens
            tag_stats["total_output_tokens"] += output_tokens
            tag_stats["total_tokens"] += input_tokens + output_tokens
            tag_stats["total_input_cost_usd"] += costs["input_cost_usd"]
            tag_stats["total_output_cost_usd"] += costs["output_cost_usd"]
            tag_stats["total_cost_usd"] += costs["total_cost_usd"]
            tag_stats["total_calls"] += 1
            tag_stats["total_duration_seconds"] += duration
            tag_stats["avg_duration_seconds"] = (
                tag_stats["total_duration_seconds"] / tag_stats["total_calls"]
            )
            
            if reasoning_tokens > 0:
                tag_stats["reasoning_tokens"] += reasoning_tokens
                tag_stats["reasoning_cost_usd"] += costs["reasoning_cost_usd"]
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tags: Union[str, List[str]] = [],
        **kwargs
    ) -> Dict:
        """
        Unified chat completion method matching OpenAI API signature exactly.
        
        Args:
            messages: List of message dictionaries
            model: Model string in "provider:model" format  
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
            
        # Parse model string
        provider_name, model_name = self._parse_model_string(model)
        provider_client = self._providers[provider_name]
        
        # Get model and provider configurations
        model_config = MODELS[model]
        provider_config = self.config.get_provider_config(provider_name)
        
        # Make a copy of kwargs to avoid modifying the original
        api_kwargs = kwargs.copy()
        
        # Add provider-specific default parameters
        provider_defaults = provider_config.get("default_params", {})
        for key, value in provider_defaults.items():
            if key not in api_kwargs:
                api_kwargs[key] = value
        
        # Store original values for retry logic
        response_format = api_kwargs.get("response_format")
        original_temperature = api_kwargs.get("temperature", 0.7)
        
        # Handle JSON mode - add instructions whenever response_format is JSON
        modified_messages = messages
        json_mode_requested = response_format and response_format.get("type") == "json_object"
        
        if json_mode_requested:
            # Always add JSON instructions when JSON is requested
            modified_messages = self._add_json_instructions_to_messages(messages, model)
            self.logger.debug(f"[{request_id}] 🔧 Injected JSON enforcement instructions for {model} (response_format=json_object)")
        
        # Remove response_format if not supported by the model
        if self._should_remove_response_format(model) and "response_format" in api_kwargs:
            self.logger.debug(f"[{request_id}] Removing response_format for {model} (not supported)")
            api_kwargs.pop("response_format")
            
        # Remove temperature if not supported  
        if not model_config["capabilities"].get("supports_temperature", True):
            if "temperature" in api_kwargs:
                self.logger.debug(f"[{request_id}] Removing temperature for {model} (not supported)")
                api_kwargs.pop("temperature")
                
        # Implement JSON validation and retry logic with fallback
        max_retries = self.config.retry_settings["max_json_retries"]
        temperature_reductions = self.config.retry_settings["temperature_reductions"]  # [0.1, 0.3]
        min_temp = self.config.retry_settings["min_temp"]  # 0.3
        
        # Track all attempts for token accumulation
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        
        current_model = model
        current_model_name = model_name
        current_provider_client = provider_client
        
        # Log initial request details
        self.logger.debug(f"[{request_id}] 🚀 Making request to {model} with temperature={api_kwargs.get('temperature', 'default')}")
        if json_mode_requested:
            self.logger.debug(f"[{request_id}] 📋 JSON mode requested - response_format=json_object")
        
        for attempt in range(max_retries + 2):  # +2 for fallback attempt
            try:
                # Log attempt start with detailed info
                current_temp = api_kwargs.get('temperature', 'default')
                max_attempts = max_retries + 2
                self.logger.info(f"[{request_id}] 🔄 REQUEST START - {current_model} (attempt {attempt + 1}/{max_attempts}) - temp={current_temp}")
                
                # Make the API call
                try:
                    response = await current_provider_client.chat.completions.create(
                        messages=modified_messages,
                        model=current_model_name,
                        **api_kwargs
                    )
                    api_error = None
                except Exception as api_e:
                    # Check if this is a JSON validation API error
                    if self._is_json_validation_api_error(api_e) and json_mode_requested:
                        self.logger.warning(f"[{request_id}] 🎯 API JSON VALIDATION FAILED - {str(api_e)[:30]}... Will trigger retry logic")
                        # Set a flag to trigger JSON validation retry logic
                        api_error = api_e
                        # Create a dummy response to continue the flow
                        response = None
                    else:
                        # Re-raise non-JSON validation errors
                        raise api_e
                
                # Extract usage statistics and accumulate (only if we got a real response)
                if response:
                    usage = response.usage
                    input_tokens = getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                    reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_reasoning_tokens += reasoning_tokens
                    
                    # Extract content
                    content = response.choices[0].message.content
                    
                    # Remove think tags if present
                    content = self._remove_think_tags(content)
                    
                    # Extract JSON from markdown if JSON mode was requested
                    if json_mode_requested:
                        content = self._extract_json_from_markdown(content)
                else:
                    # API error case - no content to process
                    content = ""
                
                # Validate JSON if response_format was requested
                if response_format and response_format.get("type") == "json_object":
                    try:
                        if api_error:
                            # Force JSON validation to fail so it triggers retry logic
                            raise json.JSONDecodeError(f"API JSON validation failed: {str(api_error)}", "", 0)
                        else:
                            json.loads(content)
                    except json.JSONDecodeError as e:
                        if attempt < max_retries:
                            # Calculate new temperature for retry
                            if attempt < len(temperature_reductions):
                                new_temp = max(min_temp, original_temperature - temperature_reductions[attempt])
                            else:
                                new_temp = min_temp
                                
                            api_kwargs["temperature"] = new_temp
                            self.logger.warning(
                                f"[{request_id}] JSON validation failed (attempt {attempt + 1}/{max_retries + 1}). "
                                f"Retrying with temperature {new_temp}: {str(e)[:50]}..."
                            )
                            # This continue will go back to the start of the retry loop with the same request_id
                            continue
                        elif attempt == max_retries:
                            # Try removing response_format first if still present
                            if "response_format" in api_kwargs:
                                self.logger.warning(
                                    f"[{request_id}] JSON validation failed after {max_retries + 1} attempts with {current_model}. "
                                    f"Removing response_format and retrying with text parsing."
                                )
                                api_kwargs.pop("response_format")
                                api_kwargs["temperature"] = original_temperature  # Reset temperature
                                self.logger.info(f"[{request_id}] 🔄 RETRY (format removal) - {current_model} (attempt {attempt + 2}/{max_attempts}) - temp={original_temperature}")
                                continue
                            else:
                                # Try fallback model as last resort
                                fallback_model = self.config.get_fallback_model(provider_name)
                                if fallback_model and fallback_model != current_model:
                                    self.logger.warning(
                                        f"[{request_id}] JSON validation failed after format removal. "
                                        f"Trying fallback model: {fallback_model}"
                                    )
                                    
                                    # Parse fallback model
                                    fallback_provider, fallback_model_name = self._parse_model_string(fallback_model)
                                    current_model = fallback_model
                                    current_model_name = fallback_model_name
                                    current_provider_client = self._providers[fallback_provider]
                                    
                                    # Reset temperature for fallback
                                    api_kwargs["temperature"] = original_temperature
                                    self.logger.info(f"[{request_id}] 🔄 RETRY (fallback model) - {current_model} (attempt {attempt + 2}/{max_attempts}) - temp={original_temperature}")
                                    continue
                        else:
                            raise ValueError(f"Failed to generate valid JSON after all retries including fallback: {e}")
                
                # Calculate final duration
                duration = time.time() - start_time
                
                # Log successful response
                self.logger.info(f"[{request_id}] ✅ REQUEST SUCCESS - {current_model} in {duration:.2f}s")
                self.logger.debug(f"[{request_id}] 📊 Token usage - Input: {total_input_tokens}, Output: {total_output_tokens} tokens")
                if total_reasoning_tokens > 0:
                    self.logger.debug(f"[{request_id}] 🧠 Reasoning tokens used: {total_reasoning_tokens}")
                
                # Update statistics with accumulated tokens
                self._update_statistics(current_model, total_input_tokens, total_output_tokens, 
                                      total_reasoning_tokens, duration, tags)
                
                # Update response with cleaned content
                # This modifies the original response object to have the cleaned content
                response.choices[0].message.content = content
                
                # TODO: Override usage stats with accumulated tokens from all retry attempts
                # Currently response.usage only shows tokens from the final successful call
                # but we should show total_input_tokens, total_output_tokens, total_reasoning_tokens
                # from ALL attempts (including failed retries)
                
                # Return the original OpenAI-style response object
                # Users access it like: response.choices[0].message.content
                return response
                
            except Exception as e:
                # Extract tokens even from failed calls if available
                if hasattr(e, 'response') and hasattr(e.response, 'usage'):
                    usage = e.response.usage
                    total_input_tokens += getattr(usage, 'prompt_tokens', 0)
                    total_output_tokens += getattr(usage, 'completion_tokens', 0)
                    total_reasoning_tokens += getattr(usage, 'reasoning_tokens', 0)
                
                # Handle rate limits with exponential backoff
                if "429" in str(e) or "rate limit" in str(e).lower():
                    backoff_times = self.config.retry_settings["rate_limit_backoff"]
                    max_rate_retries = self.config.retry_settings["max_rate_limit_retries"]
                    
                    if attempt < max_rate_retries:
                        wait_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                        self.logger.warning(
                            f"[{request_id}] Rate limit hit (attempt {attempt + 1}/{max_rate_retries + 1}). "
                            f"Waiting {wait_time}s before retry."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                
                # Re-raise if not a retryable error or max attempts reached
                raise
                
        raise Exception("Unexpected error in retry loop")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return dict(self._statistics)
        
    def get_stats_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get statistics for a specific tag."""
        if tag not in self._tag_statistics:
            # Return empty stats structure if tag not found
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_input_cost_usd": 0.0,
                "total_output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "total_calls": 0,
                "total_duration_seconds": 0.0,
                "avg_duration_seconds": 0.0,
                "reasoning_tokens": 0,
                "reasoning_cost_usd": 0.0
            }
        
        return dict(self._tag_statistics[tag])
        
    def list_models(self) -> Dict[str, Any]:
        """List all available models in OpenAI-compatible format.
        
        Returns a response matching OpenAI's GET /v1/models endpoint but with
        an additional 'available' field indicating if the provider's API key is present.
        """
        models_list = []
        
        for model_key, model_config in MODELS.items():
            provider = model_config.get("provider", "unknown")
            # Check if API key is available for this provider
            env_var = f"{provider.upper()}_API_KEY"
            is_available = provider in self._providers
            
            models_list.append({
                "id": model_key,
                "object": "model", 
                "created": 1677610602,  # Fixed timestamp like OpenAI
                "owned_by": provider,
                "available": is_available  # Elelem-specific field
            })
        
        return {
            "object": "list",
            "data": sorted(models_list, key=lambda x: x["id"])
        }