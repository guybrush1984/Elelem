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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import openai
from openai import (
    BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError,
    ConflictError, UnprocessableEntityError, RateLimitError, InternalServerError,
    LengthFinishReasonError, ContentFilterFinishReasonError
)
import pandas as pd
from jsonschema import validate, ValidationError
from .config import Config
from .metrics import MetricsStore


class InfrastructureError(Exception):
    """Errors that should trigger candidate iteration."""
    pass

class ModelError(Exception):
    """Errors that should not trigger candidate iteration."""
    pass


class Elelem:
    """Unified API wrapper with cost tracking, JSON validation, and retry logic."""
    
    def __init__(self, metrics_persist_file: Optional[str] = None, extra_provider_dirs: Optional[List[str]] = None):
        self.logger = logging.getLogger("elelem")
        self.config = Config(extra_provider_dirs)
        self._models = self._load_models()
        self._providers = self._initialize_providers()

        # Initialize metrics system
        self._metrics_store = MetricsStore(persist_file=metrics_persist_file)
    
    def _load_models(self) -> Dict[str, Any]:
        """Load model definitions using the Config system."""
        return self.config.models
    
    def _create_provider_client(self, api_key: str, base_url: str, timeout: int = 120, provider_name: str = None, default_headers: Dict = None):
        """Create an OpenAI-compatible client for any provider."""
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": 600,  # High timeout - we'll manage timeouts at application level
            "max_retries": 0  # Disable OpenAI SDK retries - Elelem handles all retry logic
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
        self._metrics_store.reset()
        
    def _get_model_config(self, model: str) -> tuple[str, str]:
        """Get provider and model_id from model configuration (opaque key lookup)."""
        if model not in self._models:
            raise ValueError(f"Unknown model: {model}")

        model_config = self._models.get(model, {})
        provider = model_config.get("provider")
        model_id = model_config.get("model_id")

        if not provider:
            raise ValueError(f"Model '{model}' missing provider configuration")
        if not model_id:
            raise ValueError(f"Model '{model}' missing model_id configuration")

        if provider not in self._providers:
            available_providers = list(self._providers.keys())
            raise ValueError(f"Provider '{provider}' not available. Available: {available_providers}")

        return provider, model_id
        
        
    async def _collect_streaming_response(self, stream):
        """Collect streaming chunks and reconstruct a normal response object."""
        content_parts = []
        reasoning_content_parts = []
        final_chunk = None
        finish_reason = None

        async for chunk in stream:
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
        
    def _remove_think_tags(self, content: str) -> str:
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
                self.logger.debug(f"Removed thinking content using </think> split: kept {len(cleaned)} chars")
        
        return cleaned
        
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
        
    def _add_json_instructions_to_messages(self, messages: List[Dict[str, str]], capabilities: Dict) -> List[Dict[str, str]]:
        """Add JSON formatting instructions to messages when response_format is JSON."""
        supports_system = capabilities.get("supports_system", True)
        
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
        
    def _calculate_costs(self, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, runtime_costs: Dict = None, candidate_cost_config: Dict = None) -> Dict[str, float]:
        """Calculate costs based on model pricing or runtime data from provider."""
        cost_config = candidate_cost_config
        
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
                    reasoning_tokens = self._extract_reasoning_tokens(response)
                    
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
                          reasoning_tokens: int, duration: float, tags: List[str], runtime_costs: Dict = None, candidate_cost_config: Dict = None, candidate_provider: str = None, requested_model: str = None):
        """Update statistics tracking."""
        costs = self._calculate_costs(model, input_tokens, output_tokens, reasoning_tokens, runtime_costs, candidate_cost_config)

        # Track actual provider used (for OpenRouter)
        actual_provider = runtime_costs.get("actual_provider") if runtime_costs else None

        # Use candidate provider directly
        provider = candidate_provider

        # Add model and provider as automatic tags
        enhanced_tags = list(tags) if tags else []
        enhanced_tags.append(f"model:{model}")
        enhanced_tags.append(f"provider:{provider}")

        self._metrics_store.record_call(
            model=model,
            provider=provider,
            tags=enhanced_tags,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            costs=costs,
            actual_provider=actual_provider,
            requested_model=requested_model
        )
    
    def _update_retry_analytics(self, retry_type: str, tags: List[str], count: int = 1):
        """Update retry analytics.

        Args:
            retry_type: Type of retry ('json_parse_retries', 'json_schema_retries', etc.)
            tags: List of tags to update
            count: Number to increment by (default 1)
        """
        # Record in MetricsStore
        self._metrics_store.record_retry(
            retry_type=retry_type,
            tags=tags,
            count=count
        )
    
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
    
    def _extract_reasoning_tokens(self, response) -> int:
        """Universal reasoning token extraction for all providers.

        Uses mathematical relationship: reasoning_tokens = total_tokens - prompt_tokens - completion_tokens
        This works for:
        - Explicit providers (OpenAI, GROQ, DeepSeek): validates against explicit fields
        - Implicit providers (Gemini): calculates from hidden tokens in total_tokens
        - Non-reasoning providers: correctly returns 0
        """
        if not response or not hasattr(response, 'usage'):
            return 0

        usage = response.usage

        # Method 1: Universal calculation from total_tokens (primary method)
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)

        if total_tokens > 0 and prompt_tokens >= 0 and completion_tokens >= 0:
            calculated_reasoning = max(0, total_tokens - prompt_tokens - completion_tokens)
            if calculated_reasoning > 0:
                self.logger.debug(f"Calculated reasoning tokens from total: {calculated_reasoning} "
                                f"(total: {total_tokens}, prompt: {prompt_tokens}, completion: {completion_tokens})")
                return calculated_reasoning

        # Method 2: Explicit field search (for validation and legacy support)
        explicit_reasoning = self._recursive_search_reasoning_tokens(usage)
        if explicit_reasoning > 0:
            self.logger.debug(f"Found explicit reasoning tokens: {explicit_reasoning}")
            return explicit_reasoning

        # Method 3: Content-based extraction (fallback for models with <think> tags)
        content_reasoning = self._extract_reasoning_from_content(response)
        if content_reasoning > 0:
            return content_reasoning

        return 0

    def _recursive_search_reasoning_tokens(self, obj, target_field="reasoning_tokens", visited=None, depth=0):
        """Recursively search for explicit reasoning_tokens field in nested objects/dicts."""
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
                result = self._recursive_search_reasoning_tokens(value, target_field, visited, depth + 1)
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
                            result = self._recursive_search_reasoning_tokens(attr_value, target_field, visited, depth + 1)
                            if result > 0:
                                return result
                    except (AttributeError, TypeError):
                        continue

        return 0

    def _extract_reasoning_from_content(self, response) -> int:
        """Extract reasoning tokens from content analysis (for models with <think> tags)."""
        if not response or not hasattr(response, 'usage'):
            return 0

        usage = response.usage

        # Check for <think> tags in response content (Parasail DeepSeek, etc.)
        if hasattr(response, 'choices') and response.choices:
            first_choice = response.choices[0]
            if hasattr(first_choice, 'message') and first_choice.message.content:
                content = first_choice.message.content
                self.logger.debug(f"Checking content for <think> tags, length: {len(content)}")

                # Extract reasoning content from <think> tags
                import re
                think_pattern = r'<think>(.*?)</think>'
                think_matches = re.findall(think_pattern, content, re.DOTALL)

                self.logger.debug(f"Found {len(think_matches)} <think> matches")

                if think_matches:
                    reasoning_content = think_matches[0].strip()
                    if reasoning_content:
                        # Estimate reasoning tokens using character count ratio
                        reasoning_chars = len(reasoning_content)

                        # Get actual response content (everything after </think>)
                        actual_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        actual_chars = len(actual_content)
                        total_chars = reasoning_chars + actual_chars

                        self.logger.debug(f"Reasoning chars: {reasoning_chars}, Actual chars: {actual_chars}")

                        if total_chars > 0:
                            reasoning_ratio = reasoning_chars / total_chars
                            total_completion_tokens = getattr(usage, 'completion_tokens', 0)
                            estimated_reasoning_tokens = int(total_completion_tokens * reasoning_ratio)
                            
                            self.logger.debug(f"Extracted reasoning from <think> tags: {estimated_reasoning_tokens} tokens "
                                            f"({reasoning_ratio:.1%} of {total_completion_tokens} total)")
                            return estimated_reasoning_tokens
                
                # Fallback for content that starts with thinking but no opening <think> tag
                elif '</think>' in content:
                    self.logger.debug("Found </think> but no opening <think> tag, using fallback")
                    parts = content.split('</think>', 1)
                    if len(parts) > 1:
                        reasoning_content = parts[0].strip()
                        actual_content = parts[1].strip()
                        
                        reasoning_chars = len(reasoning_content)
                        actual_chars = len(actual_content)
                        total_chars = reasoning_chars + actual_chars
                        
                        self.logger.debug(f"Fallback - Reasoning chars: {reasoning_chars}, Actual chars: {actual_chars}")
                        
                        if total_chars > 0 and reasoning_chars > 0:
                            reasoning_ratio = reasoning_chars / total_chars
                            total_completion_tokens = getattr(usage, 'completion_tokens', 0)
                            estimated_reasoning_tokens = int(total_completion_tokens * reasoning_ratio)
                            
                            self.logger.debug(f"Extracted reasoning via fallback: {estimated_reasoning_tokens} tokens "
                                            f"({reasoning_ratio:.1%} of {total_completion_tokens} total)")
                            return estimated_reasoning_tokens
            
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
                        
                        self.logger.debug(f"Estimated reasoning tokens via char ratio: {estimated_reasoning_tokens} "
                                        f"({reasoning_ratio:.1%} of {total_completion_tokens} total)")
                        return estimated_reasoning_tokens
        
        return 0

    def _process_response_content(self, response: Any, json_mode_requested: bool) -> str:
        """Process and clean response content."""
        # Debug logging to understand response structure
        if hasattr(response, 'model_dump_json'):
            self.logger.debug(f"Full response JSON: {response.model_dump_json()[:500]}")
        
        if hasattr(response, 'choices') and response.choices:
            first_choice = response.choices[0]
            self.logger.debug(f"First choice type: {type(first_choice)}")
            if hasattr(first_choice, 'message'):
                self.logger.debug(f"Message type: {type(first_choice.message)}")
                self.logger.debug(f"Message content: {first_choice.message.content}")
                if hasattr(first_choice.message, 'model_dump_json'):
                    self.logger.debug(f"Message JSON: {first_choice.message.model_dump_json()[:500]}")
        
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
            self.logger.warning("Response content is None despite finish_reason being 'stop'")
            content = ""
        
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
                    f"Retrying with temperature {new_temp}: {str(e)[:500]}..."
                )
            else:
                # Could be API JSON validation or client-side parse failure
                if "API JSON validation failed" in str(e):
                    self._update_retry_analytics("api_json_validation_retries", tags)
                else:
                    self._update_retry_analytics("json_parse_retries", tags)
                self.logger.warning(
                    f"[{request_id}] JSON parse failed (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying with temperature {new_temp}: {str(e)[:300]}..."
                )
            return True, new_temp
        return False, 0.0
    
    
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
        
        # Get model configuration and candidates
        try:
            model_config = self.config.get_model_config(model)
            candidates = model_config['candidates']
        except ValueError as e:
            # Record failure for invalid model
            enhanced_tags = list(tags) if tags else []
            enhanced_tags.append(f"model:{model}")
            enhanced_tags.append("error:ModelNotFound")
            self._update_retry_analytics("final_failures", enhanced_tags)
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
                    tags, start_time, **kwargs
                )
            except InfrastructureError as e:
                self.logger.warning(f"[{request_id}] 🔄 Candidate {candidate_idx + 1} failed: {e}")
                self._update_retry_analytics("candidate_iterations", tags)
                last_error = e
                
                if candidate_idx < len(candidates) - 1:
                    continue  # Try next candidate
                else:
                    # All candidates exhausted - record failure
                    enhanced_tags = list(tags) if tags else []
                    enhanced_tags.append(f"model:{model}")
                    provider = self._get_model_config(model)[0]
                    enhanced_tags.append(f"provider:{provider}")
                    enhanced_tags.append("error:AllCandidatesFailed")
                    self._update_retry_analytics("final_failures", enhanced_tags)
                    raise e
            except ModelError as e:
                # Model/request errors don't trigger candidate iteration
                # Record failure
                enhanced_tags = list(tags) if tags else []
                enhanced_tags.append(f"model:{model}")
                provider = self._get_model_config(model)[0]
                enhanced_tags.append(f"provider:{provider}")
                enhanced_tags.append("error:ModelError")
                self._update_retry_analytics("final_failures", enhanced_tags)
                raise e

        # Should never reach here, but just in case
        enhanced_tags = list(tags) if tags else []
        enhanced_tags.append(f"model:{model}")
        provider = self._get_model_config(model)[0]
        enhanced_tags.append(f"provider:{provider}")
        enhanced_tags.append("error:UnknownError")
        self._update_retry_analytics("final_failures", enhanced_tags)
        raise last_error or Exception(f"All {len(candidates)} candidates failed")
    
    
    async def _attempt_candidate(self, candidate, candidate_idx, total_candidates,
                                messages, original_model, model_config, request_id, 
                                json_mode_requested, json_schema, original_temperature,
                                tags, start_time, **kwargs):
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
        
        # Prepare API parameters
        api_kwargs = kwargs.copy()
        api_kwargs['temperature'] = original_temperature
        
        # Get provider configuration for defaults
        provider_config = self.config.get_provider_config(provider_name)
        
        # Add provider-specific default parameters
        provider_defaults = provider_config.get("default_params", {})
        for key, value in provider_defaults.items():
            if key == "stream":
                # Always apply provider stream default, ignore client input
                api_kwargs[key] = value
            elif key not in api_kwargs:
                api_kwargs[key] = value
        
        # Add model-specific default parameters (overrides provider defaults)
        model_defaults = candidate.get("default_params", {})
        for key, value in model_defaults.items():
            if key not in api_kwargs:
                api_kwargs[key] = value
        
        # Add max_tokens default if provider specifies it and user didn't provide max_tokens
        if "max_tokens" not in api_kwargs:
            max_tokens_default = self.config.get_provider_max_tokens_default(provider_name)
            if max_tokens_default is not None:
                api_kwargs["max_tokens"] = max_tokens_default

        # Add extra_body parameters with precedence: user > model > provider
        provider_extra_body = provider_config.get("extra_body", {})
        model_extra_body = self.config.get_model_extra_body(original_model)
        user_extra_body = api_kwargs.get("extra_body", {})
        
        # Merge with proper precedence (rightmost wins)
        merged_extra_body = {**provider_extra_body, **model_extra_body, **user_extra_body}
        
        if merged_extra_body:
            api_kwargs["extra_body"] = merged_extra_body
        
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
                        response = await self._collect_streaming_response(stream)
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
                    usage = response.usage
                    input_tokens = getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                    reasoning_tokens = self._extract_reasoning_tokens(response)
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_reasoning_tokens += reasoning_tokens
                    
                    content = self._process_response_content(response, json_mode_requested)
                else:
                    content = ""
                
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
                                    self._update_retry_analytics("temperature_reductions", tags)
                                    self.logger.warning(f"[{request_id}] JSON validation failed, reducing temperature to {new_temp}")
                                    continue
                            
                            # Try removing response format
                            if 'response_format' in api_kwargs:
                                api_kwargs.pop('response_format', None)
                                api_kwargs['temperature'] = original_temperature
                                self._update_retry_analytics("response_format_removals", tags)
                                self.logger.warning(f"[{request_id}] Removing response_format and retrying")
                                continue
                        
                        # All JSON retries exhausted - this is a model error, don't iterate candidates
                        self._update_retry_analytics("final_failures", tags)
                        raise ModelError(f"JSON validation failed after all retries: {e}")
                
                # Success! Calculate duration and return
                duration = time.time() - start_time
                
                # Log success with provider info
                provider_info = ""
                if hasattr(response, 'provider') and response.provider:
                    provider_info = f" via {response.provider}"
                
                self.logger.info(f"[{request_id}] ✅ SUCCESS - {candidate_model_name}{provider_info} in {duration:.2f}s")
                
                # Extract runtime costs
                runtime_costs = self._extract_openrouter_costs(response, candidate_model_name)
                
                # Calculate costs for this specific request
                candidate_cost_config = candidate.get('cost', {})
                costs = self._calculate_costs(stats_model_name, total_input_tokens, total_output_tokens,
                                            total_reasoning_tokens, runtime_costs, candidate_cost_config)

                # Update statistics
                self._update_statistics(stats_model_name, total_input_tokens, total_output_tokens,
                                      total_reasoning_tokens, duration, tags, runtime_costs, candidate_cost_config, provider_name, original_model)

                # Update response content
                response.choices[0].message.content = content

                # Convert response to dict for modification
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__

                # Add Elelem-specific metrics
                response_dict["elelem_metrics"] = {
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

                return response_dict
                
            except (InfrastructureError, ModelError):
                # Re-raise classification errors as-is
                raise
            except RateLimitError as e:
                # Handle OpenAI rate limit errors with dedicated retry logic
                if attempt < max_rate_limit_retries:
                    backoff_times = self.config.retry_settings["rate_limit_backoff"]
                    wait_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                    self._update_retry_analytics("rate_limit_retries", tags)
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
                        self._update_retry_analytics("rate_limit_retries", tags)
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
        error_str = str(error).lower()
        
        # Common infrastructure error patterns
        infrastructure_patterns = [
            "connection", "network", "timeout", "503", "502", "500",
            "service unavailable", "bad gateway", "internal server error",
            "401", "403", "unauthorized", "forbidden", "quota", "billing",
            "429", "rate limit", "too many requests"
        ]
        
        return any(pattern in error_str for pattern in infrastructure_patterns)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return self._metrics_store.get_overall_stats()

    def get_stats_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get statistics for a specific tag."""
        return self._metrics_store.get_stats_by_tag(tag)

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
        return self._metrics_store.get_summary(start_time, end_time, tags)

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