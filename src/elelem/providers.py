"""
Simple provider implementation - all providers follow OpenAI specification
"""

import openai
from typing import Dict, List, Any


def create_provider_client(api_key: str, base_url: str, timeout: int = 120):
    """Create an OpenAI-compatible client for any provider."""
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )