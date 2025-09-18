"""
Provider management functions for Elelem.
"""

import os
import logging
from typing import Dict, Any, Tuple
import openai


def create_provider_client(api_key: str, base_url: str, timeout: int,
                          provider_name: str, default_headers: Dict) -> openai.AsyncOpenAI:
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


def initialize_providers(providers_config: Dict, timeout_seconds: int, logger: logging.Logger) -> Dict[str, Any]:
    """Initialize provider clients."""
    providers = {}

    # Initialize all configured providers
    for provider_name, provider_config in providers_config.items():
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

            providers[provider_name] = create_provider_client(
                api_key=api_key,
                base_url=provider_config["endpoint"],
                timeout=timeout_seconds,
                provider_name=provider_name,
                default_headers=custom_headers
            )
            logger.debug(f"Initialized {provider_name} provider")
        else:
            logger.warning(f"No API key found for {provider_name} (env var: {env_var})")

    return providers


def get_model_config(model: str, models_dict: Dict, providers: Dict) -> Tuple[str, str]:
    """Get provider and model_id from model configuration (opaque key lookup)."""
    if model not in models_dict:
        raise ValueError(f"Unknown model: {model}")

    model_config = models_dict.get(model, {})
    provider = model_config.get("provider")
    model_id = model_config.get("model_id")

    if not provider:
        raise ValueError(f"Model '{model}' missing provider configuration")
    if not model_id:
        raise ValueError(f"Model '{model}' missing model_id configuration")

    if provider not in providers:
        available_providers = list(providers.keys())
        raise ValueError(f"Provider '{provider}' not available. Available: {available_providers}")

    return provider, model_id