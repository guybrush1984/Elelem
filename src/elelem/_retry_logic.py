"""
Retry logic functions for Elelem.
"""

import json
import logging
from typing import List, Dict, Tuple
from .metrics import MetricsStore


def update_retry_analytics(retry_type: str, tags: List[str], metrics_store: MetricsStore, count: int = 1):
    """Update retry analytics.

    Args:
        retry_type: Type of retry ('json_parse_retries', 'json_schema_retries', etc.)
        tags: List of tags to update
        metrics_store: MetricsStore instance to record to
        count: Number to increment by (default 1)
    """
    # Record in MetricsStore
    metrics_store.record_retry(
        retry_type=retry_type,
        tags=tags,
        count=count
    )


def handle_json_retry(error: json.JSONDecodeError, attempt: int, max_retries: int,
                     request_id: str, api_kwargs: Dict, original_temperature: float,
                     temperature_reductions: List[float], min_temp: float, tags: List[str],
                     request_tracker, logger: logging.Logger) -> Tuple[bool, float]:
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
        request_tracker.record_retry("temperature_reductions")

        # Log appropriate message based on error type and track retry type
        if "JSON schema validation failed" in str(error):
            request_tracker.record_retry("json_schema_retries")
            logger.warning(
                f"[{request_id}] JSON schema validation failed (attempt {attempt + 1}/{max_retries + 1}). "
                f"Retrying with temperature {new_temp}: {str(error)[:500]}..."
            )
        else:
            # Could be API JSON validation or client-side parse failure
            if "API JSON validation failed" in str(error):
                request_tracker.record_retry("api_json_validation_retries")
            else:
                request_tracker.record_retry("json_parse_retries")
            logger.warning(
                f"[{request_id}] JSON parse failed (attempt {attempt + 1}/{max_retries + 1}). "
                f"Retrying with temperature {new_temp}: {str(error)[:300]}..."
            )
        return True, new_temp
    return False, 0.0


def is_infrastructure_error(error) -> bool:
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