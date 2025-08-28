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
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import openai
from jsonschema import validate, ValidationError
from .config import Config


class Elelem:
    """Unified API wrapper with cost tracking, JSON validation, and retry logic."""
    
    def __init__(self):
        self.logger = logging.getLogger("elelem")
        self.config = Config()
        self._statistics = {}
        self._tag_statistics = {}
        self._models = self._load_models()
        self._providers = self._initialize_providers()
        
        # Initialize statistics tracking
        self._reset_stats()
    
    def _load_models(self) -> Dict[str, Any]:
        """Load model definitions from YAML file."""
        models_path = Path(__file__).parent / "models.yaml"
        
        try:
            with open(models_path, 'r') as f:
                data = yaml.safe_load(f)
            return data.get("models", {})
        except (FileNotFoundError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load Elelem models: {e}")
    
    def _create_provider_client(self, api_key: str, base_url: str, timeout: int = 120, provider_name: str = None, default_headers: Dict = None):
        """Create an OpenAI-compatible client for any provider."""
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout
        }
        
        # Add custom headers for specific providers
        if default_headers:
            client_kwargs["default_headers"] = default_headers
            
        return openai.AsyncOpenAI(**client_kwargs)
        
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize provider clients."""
        providers = {}
        
        # Initialize all configured providers
        for provider_name, provider_config in self.config.providers.items():
            # Handle infrastructure providers (e.g., cerebras@openrouter)
            base_provider = provider_config.get("base_provider")
            if base_provider:
                # Use base provider for API key: cerebras@openrouter -> OPENROUTER_API_KEY
                env_var = f"{base_provider.upper()}_API_KEY"
            else:
                # Direct provider: openrouter -> OPENROUTER_API_KEY
                env_var = f"{provider_name.upper()}_API_KEY"
            
            api_key = os.getenv(env_var)
            
            if api_key:
                # Get custom headers from provider config
                custom_headers = provider_config.get("headers")
                
                providers[provider_name] = self._create_provider_client(
                    api_key=api_key,
                    base_url=provider_config["endpoint"],
                    timeout=self.config.timeout_seconds,
                    provider_name=provider_name,
                    default_headers=custom_headers
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
            "reasoning_cost_usd": 0.0,
            "providers": {},  # Track which actual providers were used (e.g., "Novita": {"count": 5, "cost": 0.001})
            "retry_analytics": {
                "json_parse_retries": 0,
                "json_schema_retries": 0,
                "api_json_validation_retries": 0,
                "rate_limit_retries": 0,
                "total_retries": 0,
                "temperature_reductions": 0,
                "final_failures": 0,
                "fallback_model_usage": 0,
                "response_format_removals": 0
            }
        }
        self._tag_statistics = {}
        
    def _parse_model_string(self, model: str) -> tuple[str, str]:
        """Parse provider:model string and get actual model_id from config."""
        if ":" not in model:
            raise ValueError(f"Model must be in 'provider:model' format, got: {model}")
            
        provider, _ = model.split(":", 1)
        
        if model not in self._models:
            raise ValueError(f"Unknown model: {model}")
            
        if provider not in self._providers:
            available_providers = list(self._providers.keys())
            raise ValueError(f"Provider '{provider}' not available. Available: {available_providers}")
        
        # Get the actual model_id from configuration
        model_config = self._models.get(model, {})
        model_id = model_config.get("model_id")
        if not model_id:
            # Fallback to the part after colon if no model_id is specified
            _, model_id = model.split(":", 1)
            
        return provider, model_id
        
    def _should_remove_response_format(self, model: str) -> bool:
        """Check if response_format should be removed for this model."""
        model_config = self._models.get(model, {})
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
        
    def _validate_json_schema(self, json_obj: Any, schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a JSON object against a JSON Schema.
        
        Args:
            json_obj: The parsed JSON object to validate
            schema: The JSON Schema to validate against
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            validate(instance=json_obj, schema=schema)
            return True, None
        except ValidationError as e:
            # Build detailed error message
            error_parts = [f"Schema validation failed: {e.message}"]
            
            if e.path:
                path_str = ".".join(str(p) for p in e.path)
                error_parts.append(f"at path: {path_str}")
                
            if e.schema_path:
                schema_path_str = ".".join(str(p) for p in e.schema_path)
                error_parts.append(f"schema path: {schema_path_str}")
                
            error_msg = " | ".join(error_parts)
            return False, error_msg
        
    def _is_json_validation_api_error(self, error: Exception) -> bool:
        """Check if the error is a json_validate_failed API error."""
        error_str = str(error).lower()
        return "json_validate_failed" in error_str and "400" in error_str
        
    def _add_json_instructions_to_messages(self, messages: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
        """Add JSON formatting instructions to messages when response_format is JSON."""
        model_config = self._models.get(model, {})
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
        
    def _calculate_costs(self, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, runtime_costs: Dict = None) -> Dict[str, float]:
        """Calculate costs based on model pricing or runtime data from provider."""
        model_config = self._models.get(model, {})
        cost_config = model_config.get("cost", {})
        
        # Handle runtime cost calculation (e.g., from OpenRouter headers)
        if runtime_costs:
            return {
                "input_cost_usd": runtime_costs.get("input_cost_usd", 0.0),
                "output_cost_usd": runtime_costs.get("output_cost_usd", 0.0),
                "reasoning_cost_usd": runtime_costs.get("reasoning_cost_usd", 0.0),
                "total_cost_usd": runtime_costs.get("total_cost_usd", 0.0)
            }
        
        # Handle runtime-priced models (cost: "runtime")
        if cost_config == "runtime":
            self.logger.warning(f"Runtime pricing model {model} used without runtime cost data - costs set to 0")
            return {
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "reasoning_cost_usd": 0.0,
                "total_cost_usd": 0.0
            }
        
        # Standard static pricing calculation
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
    
    def _extract_openrouter_costs(self, response, model: str) -> Dict[str, Any]:
        """Extract OpenRouter cost information from response usage field."""
        runtime_costs = None
        
        try:
            # Check if this is an OpenRouter model (direct or infrastructure provider)
            if not (model.startswith("openrouter:") or "@openrouter:" in model):
                return runtime_costs
            
            # OpenRouter includes detailed usage information when requested
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                
                # Extract cost information (OpenRouter returns cost in USD)
                if hasattr(usage, 'cost'):
                    total_cost_usd = usage.cost
                    # OpenRouter cost field is already in USD, no conversion needed
                    
                    # For now, we'll distribute the cost proportionally between input/output
                    # based on token counts (this is an approximation)
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    total_tokens = prompt_tokens + completion_tokens
                    
                    if total_tokens > 0:
                        input_cost_usd = total_cost_usd * (prompt_tokens / total_tokens)
                        output_cost_usd = total_cost_usd * (completion_tokens / total_tokens)
                    else:
                        input_cost_usd = total_cost_usd / 2
                        output_cost_usd = total_cost_usd / 2
                    
                    # Extract reasoning tokens if available
                    reasoning_tokens = 0
                    if hasattr(usage, 'completion_tokens_details'):
                        reasoning_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
                    
                    # Extract provider information (which actual provider was used)
                    actual_provider = None
                    if hasattr(response, 'provider'):
                        actual_provider = response.provider
                    else:
                        # Try model_dump fallback
                        try:
                            response_dict = response.model_dump()
                            actual_provider = response_dict.get('provider')
                        except Exception:
                            pass
                    
                    runtime_costs = {
                        "input_cost_usd": input_cost_usd,
                        "output_cost_usd": output_cost_usd,
                        "reasoning_cost_usd": 0.0,  # OpenRouter includes this in output cost
                        "total_cost_usd": total_cost_usd,
                        "openrouter_cost_usd": total_cost_usd,  # OpenRouter cost in USD
                        "actual_provider": actual_provider  # Which provider OpenRouter routed to
                    }
                    
                    self.logger.debug(f"OpenRouter runtime costs: ${total_cost_usd:.8f} USD")
                    
        except Exception as e:
            self.logger.debug(f"Could not extract OpenRouter costs: {e}")
        
        return runtime_costs
        
    def _update_statistics(self, model: str, input_tokens: int, output_tokens: int, 
                          reasoning_tokens: int, duration: float, tags: List[str], runtime_costs: Dict = None):
        """Update statistics tracking."""
        costs = self._calculate_costs(model, input_tokens, output_tokens, reasoning_tokens, runtime_costs)
        
        # Track actual provider used (for OpenRouter)
        actual_provider = runtime_costs.get("actual_provider") if runtime_costs else None
        
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
        
        # Track provider usage (for OpenRouter actual providers)
        if actual_provider:
            if actual_provider not in self._statistics["providers"]:
                self._statistics["providers"][actual_provider] = {
                    "count": 0,
                    "total_cost_usd": 0.0,
                    "total_tokens": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0
                }
            
            provider_stats = self._statistics["providers"][actual_provider]
            provider_stats["count"] += 1
            provider_stats["total_cost_usd"] += costs["total_cost_usd"]
            provider_stats["total_tokens"] += input_tokens + output_tokens
            provider_stats["total_input_tokens"] += input_tokens
            provider_stats["total_output_tokens"] += output_tokens
            
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
                    "reasoning_cost_usd": 0.0,
                    "retry_analytics": {
                        "json_parse_retries": 0,
                        "json_schema_retries": 0,
                        "api_json_validation_retries": 0,
                        "rate_limit_retries": 0,
                        "total_retries": 0,
                        "temperature_reductions": 0,
                        "final_failures": 0,
                        "fallback_model_usage": 0,
                        "response_format_removals": 0
                    }
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
    
    def _update_retry_analytics(self, retry_type: str, tags: List[str], count: int = 1):
        """Update retry analytics for both overall and tag-specific statistics.
        
        Args:
            retry_type: Type of retry ('json_parse_retries', 'json_schema_retries', etc.)
            tags: List of tags to update
            count: Number to increment by (default 1)
        """
        # Update overall statistics
        if retry_type in self._statistics["retry_analytics"]:
            self._statistics["retry_analytics"][retry_type] += count
            self._statistics["retry_analytics"]["total_retries"] += count
        
        # Update tag-specific statistics
        for tag in tags:
            if tag in self._tag_statistics:
                if retry_type in self._tag_statistics[tag]["retry_analytics"]:
                    self._tag_statistics[tag]["retry_analytics"][retry_type] += count
                    self._tag_statistics[tag]["retry_analytics"]["total_retries"] += count
    
    def _setup_request(self, model: str, request_id: str, **kwargs) -> tuple[str, str, Any, Dict, Dict, str, bool, Any]:
        """Setup and validate request parameters.
        
        Returns:
            Tuple of (provider_name, model_name, provider_client, model_config, 
                     api_kwargs, original_temperature, json_mode_requested, json_schema)
        """
        # Parse model string
        provider_name, model_name = self._parse_model_string(model)
        provider_client = self._providers[provider_name]
        
        # Get model and provider configurations
        model_config = self._models[model]
        provider_config = self.config.get_provider_config(provider_name)
        
        # Make a copy of kwargs to avoid modifying the original
        api_kwargs = kwargs.copy()
        
        # Add provider-specific default parameters
        provider_defaults = provider_config.get("default_params", {})
        for key, value in provider_defaults.items():
            if key not in api_kwargs:
                api_kwargs[key] = value
        
        # Add provider-specific extra_body parameters (for newer OpenAI library)
        extra_body = provider_config.get("extra_body", {})
        if extra_body:
            api_kwargs["extra_body"] = {**extra_body, **api_kwargs.get("extra_body", {})}
        
        # Store original values for retry logic
        response_format = api_kwargs.get("response_format")
        original_temperature = api_kwargs.get("temperature", 0.7)
        json_schema = api_kwargs.get("json_schema")
        
        # Warn if json_schema provided without JSON response format
        if json_schema and (not response_format or response_format.get("type") != "json_object"):
            self.logger.warning(f"[{request_id}] json_schema provided but response_format is not set to json_object. Schema validation will be skipped.")
        
        # Remove json_schema from api_kwargs - it's not part of the OpenAI API
        if "json_schema" in api_kwargs:
            api_kwargs.pop("json_schema")
        
        # Handle JSON mode - add instructions whenever response_format is JSON
        json_mode_requested = response_format and response_format.get("type") == "json_object"
        
        return (provider_name, model_name, provider_client, model_config, 
                api_kwargs, original_temperature, json_mode_requested, json_schema)
    
    def _preprocess_messages(self, messages: List[Dict[str, str]], model: str, json_mode_requested: bool) -> List[Dict[str, str]]:
        """Preprocess messages for the request."""
        if json_mode_requested:
            # Always add JSON instructions when JSON is requested
            return self._add_json_instructions_to_messages(messages, model)
        return messages
    
    def _cleanup_api_kwargs(self, api_kwargs: Dict, model: str, model_config: Dict) -> None:
        """Remove unsupported parameters from api_kwargs."""
        # Remove response_format if not supported by the model
        if self._should_remove_response_format(model) and "response_format" in api_kwargs:
            self.logger.debug(f"Removing response_format for {model} (not supported)")
            api_kwargs.pop("response_format")
            
        # Remove temperature if not supported  
        if not model_config["capabilities"].get("supports_temperature", True):
            if "temperature" in api_kwargs:
                self.logger.debug(f"Removing temperature for {model} (not supported)")
                api_kwargs.pop("temperature")
    
    def _process_response_content(self, response: Any, json_mode_requested: bool) -> str:
        """Process and clean response content."""
        content = response.choices[0].message.content
        
        # Remove think tags if present
        content = self._remove_think_tags(content)
        
        # Extract JSON from markdown if JSON mode was requested
        if json_mode_requested:
            content = self._extract_json_from_markdown(content)
            
        return content
    
    def _validate_json_response(self, content: str, json_schema: Any, api_error: Exception) -> None:
        """Validate JSON response content and schema."""
        if api_error:
            # Force JSON validation to fail so it triggers retry logic
            raise json.JSONDecodeError(f"API JSON validation failed: {str(api_error)}", "", 0)
        else:
            # First parse the JSON
            parsed_json = json.loads(content)
            
            # Then validate against schema if provided
            if json_schema:
                is_valid, schema_error = self._validate_json_schema(parsed_json, json_schema)
                if not is_valid:
                    # Treat schema validation failure like JSON parse failure
                    raise json.JSONDecodeError(f"JSON schema validation failed: {schema_error}", "", 0)
    
    def _handle_json_retry(self, e: json.JSONDecodeError, attempt: int, max_retries: int, 
                          request_id: str, api_kwargs: Dict, original_temperature: float,
                          temperature_reductions: List[float], min_temp: float, tags: List[str]) -> tuple[bool, float]:
        """Handle JSON validation retry logic.
        
        Returns:
            Tuple of (should_continue, new_temperature)
        """
        if attempt < max_retries:
            # Calculate new temperature for retry
            if attempt < len(temperature_reductions):
                new_temp = max(min_temp, original_temperature - temperature_reductions[attempt])
            else:
                new_temp = min_temp
                
            api_kwargs["temperature"] = new_temp
            
            # Track temperature reduction
            self._update_retry_analytics("temperature_reductions", tags)
            
            # Log appropriate message based on error type and track retry type
            if "JSON schema validation failed" in str(e):
                self._update_retry_analytics("json_schema_retries", tags)
                self.logger.warning(
                    f"[{request_id}] JSON schema validation failed (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying with temperature {new_temp}: {str(e)[:100]}..."
                )
            else:
                # Could be API JSON validation or client-side parse failure
                if "API JSON validation failed" in str(e):
                    self._update_retry_analytics("api_json_validation_retries", tags)
                else:
                    self._update_retry_analytics("json_parse_retries", tags)
                self.logger.warning(
                    f"[{request_id}] JSON parse failed (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying with temperature {new_temp}: {str(e)[:50]}..."
                )
            return True, new_temp
        return False, 0.0
    
    def _handle_fallback_strategies(self, attempt: int, max_retries: int, request_id: str,
                                   api_kwargs: Dict, original_temperature: float,
                                   current_model: str, provider_name: str, tags: List[str]) -> tuple[bool, str, str, Any]:
        """Handle fallback strategies when retries are exhausted.
        
        Returns:
            Tuple of (should_continue, new_model, new_model_name, new_provider_client)
        """
        if attempt == max_retries:
            # Try removing response_format first if still present
            if "response_format" in api_kwargs:
                self._update_retry_analytics("response_format_removals", tags)
                self.logger.warning(
                    f"[{request_id}] JSON validation failed after {max_retries + 1} attempts with {current_model}. "
                    f"Removing response_format and retrying with text parsing."
                )
                api_kwargs.pop("response_format")
                api_kwargs["temperature"] = original_temperature  # Reset temperature
                max_attempts = max_retries + 2
                self.logger.info(f"[{request_id}] 🔄 RETRY (format removal) - {current_model} (attempt {attempt + 2}/{max_attempts}) - temp={original_temperature}")
                return True, current_model, current_model.split(":", 1)[1], self._providers[provider_name]
            else:
                # Try fallback model as last resort
                fallback_model = self.config.get_fallback_model(provider_name)
                if fallback_model and fallback_model != current_model:
                    self._update_retry_analytics("fallback_model_usage", tags)
                    self.logger.warning(
                        f"[{request_id}] JSON validation failed after format removal. "
                        f"Trying fallback model: {fallback_model}"
                    )
                    
                    # Parse fallback model
                    fallback_provider, fallback_model_name = self._parse_model_string(fallback_model)
                    
                    # Reset temperature for fallback
                    api_kwargs["temperature"] = original_temperature
                    max_attempts = max_retries + 2
                    self.logger.info(f"[{request_id}] 🔄 RETRY (fallback model) - {fallback_model} (attempt {attempt + 2}/{max_attempts}) - temp={original_temperature}")
                    return True, fallback_model, fallback_model_name, self._providers[fallback_provider]
        
        return False, current_model, current_model.split(":", 1)[1], self._providers[provider_name]
    
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
        
        # Setup request parameters and configurations
        (provider_name, model_name, provider_client, model_config, 
         api_kwargs, original_temperature, json_mode_requested, json_schema) = self._setup_request(
            model, request_id, **kwargs)
        
        # Preprocess messages for JSON mode if needed
        modified_messages = self._preprocess_messages(messages, model, json_mode_requested)
        if json_mode_requested:
            self.logger.debug(f"[{request_id}] 🔧 Injected JSON enforcement instructions for {model} (response_format=json_object)")
        
        # Clean up unsupported API parameters
        self._cleanup_api_kwargs(api_kwargs, model, model_config)
                
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
                    
                    # Process response content
                    content = self._process_response_content(response, json_mode_requested)
                else:
                    # API error case - no content to process
                    content = ""
                
                # Validate JSON if JSON mode was originally requested (even if response_format was removed)
                if json_mode_requested:
                    try:
                        self._validate_json_response(content, json_schema, api_error)
                    except json.JSONDecodeError as e:
                        # Handle JSON validation retry
                        should_continue, new_temp = self._handle_json_retry(
                            e, attempt, max_retries, request_id, api_kwargs, 
                            original_temperature, temperature_reductions, min_temp, tags)
                        
                        if should_continue:
                            continue
                        else:
                            # Try fallback strategies
                            should_continue, current_model, current_model_name, current_provider_client = self._handle_fallback_strategies(
                                attempt, max_retries, request_id, api_kwargs, 
                                original_temperature, current_model, provider_name, tags)
                            
                            if should_continue:
                                continue
                            else:
                                # Track final failure
                                self._update_retry_analytics("final_failures", tags)
                                raise ValueError(f"Failed to generate valid JSON after all retries including fallback: {e}")
                
                # Calculate final duration
                duration = time.time() - start_time
                
                # Log successful response with infrastructure provider info if available
                provider_info = ""
                if hasattr(response, 'provider') and response.provider:
                    # Check if this is an infrastructure provider (provider contains @)
                    provider_name = model_config.get("provider", "")
                    if "@" in provider_name:
                        infrastructure_provider, routing_provider = provider_name.split("@")
                        provider_info = f" via {response.provider} (routed through {routing_provider})"
                    else:
                        provider_info = f" via {response.provider}"
                
                self.logger.info(f"[{request_id}] ✅ REQUEST SUCCESS - {current_model}{provider_info} in {duration:.2f}s")
                self.logger.debug(f"[{request_id}] 📊 Token usage - Input: {total_input_tokens}, Output: {total_output_tokens} tokens")
                if total_reasoning_tokens > 0:
                    self.logger.debug(f"[{request_id}] 🧠 Reasoning tokens used: {total_reasoning_tokens}")
                
                # Extract runtime costs from OpenRouter if applicable
                runtime_costs = self._extract_openrouter_costs(response, current_model)
                
                # Update statistics with accumulated tokens
                self._update_statistics(current_model, total_input_tokens, total_output_tokens, 
                                      total_reasoning_tokens, duration, tags, runtime_costs)
                
                # Update response with cleaned content
                response.choices[0].message.content = content
                
                # Return the original OpenAI-style response object
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
                        # Track rate limit retry
                        self._update_retry_analytics("rate_limit_retries", tags)
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
                "reasoning_cost_usd": 0.0,
                "retry_analytics": {
                    "json_parse_retries": 0,
                    "json_schema_retries": 0,
                    "api_json_validation_retries": 0,
                    "rate_limit_retries": 0,
                    "total_retries": 0,
                    "temperature_reductions": 0,
                    "final_failures": 0,
                    "fallback_model_usage": 0,
                    "response_format_removals": 0
                }
            }
        
        return dict(self._tag_statistics[tag])
        
    def list_models(self) -> Dict[str, Any]:
        """List all available models in OpenAI-compatible format.
        
        Returns a response matching OpenAI's GET /v1/models endpoint but with
        an additional 'available' field indicating if the provider's API key is present.
        """
        models_list = []
        
        for model_key, model_config in self._models.items():
            provider = model_config.get("provider", "unknown")
            
            # Handle infrastructure providers for availability check
            provider_config = self.config.providers.get(provider, {})
            base_provider = provider_config.get("base_provider")
            if base_provider:
                env_var = f"{base_provider.upper()}_API_KEY"
            else:
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