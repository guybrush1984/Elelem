"""
Provider management functions for Elelem.
"""

import os
import logging
import httpx
from typing import Dict, Any, Tuple, Optional, List
import openai


def fetch_available_models(endpoint: str, timeout: float, api_key: Optional[str] = None) -> Optional[List[str]]:
    """Fetch list of available models from a provider endpoint.

    Args:
        endpoint: Base URL to query (e.g., http://localhost:11434/v1)
        timeout: Timeout in seconds
        api_key: Optional API key for authentication

    Returns:
        List of model IDs if successful, None otherwise
    """
    try:
        models_url = f"{endpoint.rstrip('/')}/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=timeout) as client:
            response = client.get(models_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                # Extract model IDs from the response
                if "data" in data and isinstance(data["data"], list):
                    return [model.get("id") for model in data["data"] if "id" in model]
            return None
    except Exception:
        return None


def probe_endpoint(endpoint: str, timeout: float, logger: logging.Logger, api_key: Optional[str] = None) -> bool:
    """Probe an endpoint to check if it's accessible.

    Args:
        endpoint: Base URL to probe (e.g., http://localhost:11434/v1)
        timeout: Timeout in seconds
        logger: Logger instance
        api_key: Optional API key for authentication

    Returns:
        True if endpoint responds successfully, False otherwise
    """
    try:
        # Try to access /models endpoint which should be available on all OpenAI-compatible APIs
        probe_url = f"{endpoint.rstrip('/')}/models"

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=timeout) as client:
            response = client.get(probe_url, headers=headers)
            # Only accept 200 - 401/403 means auth failed, provider won't work
            if response.status_code == 200:
                logger.debug(f"Endpoint probe successful: {endpoint} (status: {response.status_code})")
                return True
            else:
                logger.debug(f"Endpoint probe failed: {endpoint} (status: {response.status_code})")
                return False
    except Exception as e:
        logger.warning(f"Endpoint probe failed: {endpoint} ({type(e).__name__}: {e})")
        return False


def select_working_endpoint(endpoints: List[str], timeout: float, provider_name: str, logger: logging.Logger, api_key: Optional[str] = None) -> Optional[str]:
    """Try multiple endpoints and return the first one that works.

    Args:
        endpoints: List of endpoint URLs to try
        timeout: Timeout in seconds for each probe
        provider_name: Name of the provider being probed
        logger: Logger instance
        api_key: Optional API key for authentication

    Returns:
        First working endpoint URL, or None if none work
    """
    logger.info(f"[{provider_name}] Probing {len(endpoints)} endpoint(s) to find working connection...")

    for endpoint in endpoints:
        logger.debug(f"[{provider_name}] Trying endpoint: {endpoint}")
        if probe_endpoint(endpoint, timeout, logger, api_key):
            logger.info(f"[{provider_name}] ✅ Selected working endpoint: {endpoint}")
            return endpoint

    logger.warning(f"[{provider_name}] ❌ No working endpoints found among {len(endpoints)} candidates")
    return None


def validate_provider_models(provider_name: str, endpoint: str, api_key: Optional[str], models_config: Dict, logger: logging.Logger):
    """Validate that configured models exist at the provider.

    Args:
        provider_name: Name of the provider
        endpoint: Provider endpoint URL
        api_key: API key for authentication
        models_config: Dictionary of all configured models
        logger: Logger instance
    """
    # Get list of models configured for this provider
    provider_models = {
        model_key: model_config
        for model_key, model_config in models_config.items()
        if model_config.get("provider") == provider_name
    }

    if not provider_models:
        logger.debug(f"[{provider_name}] No models configured, skipping validation")
        return

    # Fetch available models from the provider
    available_models = fetch_available_models(endpoint, timeout=5.0, api_key=api_key)

    if available_models is None:
        logger.warning(f"[{provider_name}] Could not fetch available models for validation")
        return

    # Validate each configured model
    missing_models = []
    for model_key, model_config in provider_models.items():
        model_id = model_config.get("model_id")
        if model_id and model_id not in available_models:
            missing_models.append(f"{model_key} (model_id: {model_id})")

    if missing_models:
        logger.warning(f"[{provider_name}] ⚠️  {len(missing_models)} configured model(s) not found at provider:")
        for missing in missing_models[:5]:  # Show first 5
            logger.warning(f"[{provider_name}]    - {missing}")
        if len(missing_models) > 5:
            logger.warning(f"[{provider_name}]    ... and {len(missing_models) - 5} more")
    else:
        logger.info(f"[{provider_name}] ✅ All {len(provider_models)} configured model(s) validated")


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


def initialize_providers(providers_config: Dict, timeout_seconds: int, logger: logging.Logger, models_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Initialize provider clients and optionally validate configured models.

    Args:
        providers_config: Provider configuration dict
        timeout_seconds: Timeout for requests
        logger: Logger instance
        models_config: Optional models configuration for validation

    Returns:
        Dict of initialized provider clients
    """
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

        # Determine the endpoint to use
        # Support both single 'endpoint' and multiple 'endpoints' with probing
        endpoint = None
        if "endpoints" in provider_config:
            # Multiple endpoints - probe and select the first working one
            probe_timeout = provider_config.get("probe_timeout", 2.0)
            endpoint = select_working_endpoint(
                provider_config["endpoints"],
                probe_timeout,
                provider_name,
                logger,
                api_key
            )
            if not endpoint:
                logger.warning(f"[{provider_name}] No working endpoints found, skipping provider")
                continue
        elif "endpoint" in provider_config:
            # Single endpoint - probe it to verify it's accessible
            probe_timeout = provider_config.get("probe_timeout", 2.0)
            single_endpoint = provider_config["endpoint"]

            logger.info(f"[{provider_name}] Probing endpoint: {single_endpoint}")
            if probe_endpoint(single_endpoint, probe_timeout, logger, api_key):
                logger.info(f"[{provider_name}] ✅ Endpoint is accessible: {single_endpoint}")
                endpoint = single_endpoint
            else:
                logger.warning(f"[{provider_name}] ❌ Endpoint not accessible, skipping provider: {single_endpoint}")
                continue
        else:
            logger.error(f"[{provider_name}] No 'endpoint' or 'endpoints' configured")
            continue

        if api_key:
            # Get custom headers from provider config
            custom_headers = provider_config.get("headers")

            providers[provider_name] = create_provider_client(
                api_key=api_key,
                base_url=endpoint,
                timeout=timeout_seconds,
                provider_name=provider_name,
                default_headers=custom_headers
            )
            logger.debug(f"Initialized {provider_name} provider with endpoint: {endpoint}")

            # Validate configured models if models_config is provided
            if models_config:
                validate_provider_models(provider_name, endpoint, api_key, models_config, logger)
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