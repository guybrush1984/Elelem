"""
Request execution functions for Elelem.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Tuple
from openai import RateLimitError

from ._reasoning_tokens import extract_token_counts
from ._response_processing import collect_streaming_response, process_response_content
from ._json_validation import validate_json_response, is_json_validation_api_error
from ._retry_logic import handle_json_retry, is_infrastructure_error, update_retry_analytics
from ._cost_calculation import calculate_costs, extract_runtime_costs
from ._exceptions import InfrastructureError, ModelError


def prepare_api_kwargs(kwargs: Dict, original_temperature: float, provider_config: Dict,
                      candidate: Dict, original_model: str, config, provider_name: str) -> Dict:
    """Prepare API parameters with proper precedence: user > model > provider.

    Note: 'stream' is an internal Elelem parameter (controls how we query providers),
    not a user parameter. Elelem always returns complete responses to users.
    """
    api_kwargs = kwargs.copy()
    api_kwargs['temperature'] = original_temperature

    # Remove 'stream' from user kwargs - it's an internal-only parameter
    api_kwargs.pop('stream', None)

    # Add provider-specific default parameters
    provider_defaults = provider_config.get("default_params", {})
    for key, value in provider_defaults.items():
        if key == "stream":
            # Always apply provider stream default (internal parameter)
            api_kwargs[key] = value
        elif key not in api_kwargs:
            api_kwargs[key] = value

    # Add model-specific default parameters (overrides provider defaults)
    model_defaults = candidate.get("default_params", {})
    for key, value in model_defaults.items():
        if key == "stream":
            # Always apply model stream default (internal parameter)
            api_kwargs[key] = value
        elif key not in api_kwargs:
            api_kwargs[key] = value

    # Add max_tokens default if provider specifies it and user didn't provide max_tokens
    if "max_tokens" not in api_kwargs:
        max_tokens_default = config.get_provider_max_tokens_default(provider_name)
        if max_tokens_default is not None:
            api_kwargs["max_tokens"] = max_tokens_default

    # Add extra_body parameters with precedence: user > model > provider
    provider_extra_body = provider_config.get("extra_body", {})
    model_extra_body = config.get_model_extra_body(original_model)
    user_extra_body = api_kwargs.get("extra_body", {})

    # Merge with proper precedence (rightmost wins)
    merged_extra_body = {**provider_extra_body, **model_extra_body, **user_extra_body}

    if merged_extra_body:
        api_kwargs["extra_body"] = merged_extra_body

    return api_kwargs