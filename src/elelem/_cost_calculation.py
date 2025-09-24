"""
Cost calculation functions for Elelem.
"""

import logging
from typing import Dict, Any, Optional


def calculate_costs(model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int,
                   runtime_costs: Optional[Dict], candidate_cost_config: Any, logger: logging.Logger) -> Dict[str, float]:
    """Calculate costs based on model pricing or runtime data from provider."""
    cost_config = candidate_cost_config

    # Handle runtime-priced models (cost: "runtime")
    if cost_config == "runtime":
        if runtime_costs:
            # Use actual runtime costs from provider
            return {
                "input_cost_usd": runtime_costs.get("input_cost_usd", 0.0),
                "output_cost_usd": runtime_costs.get("output_cost_usd", 0.0),
                "reasoning_cost_usd": runtime_costs.get("reasoning_cost_usd", 0.0),
                "total_cost_usd": runtime_costs.get("total_cost_usd", 0.0)
            }
        else:
            # Runtime pricing configured but no runtime cost data received
            logger.warning(f"Runtime pricing model {model} used without runtime cost data - costs set to 0")
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
    # output_tokens already includes reasoning tokens in our normalized structure
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    reasoning_cost = (reasoning_tokens / 1_000_000) * output_cost_per_1m

    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "reasoning_cost_usd": reasoning_cost,
        "total_cost_usd": input_cost + output_cost
    }


def extract_runtime_costs(response: Any, cost_config: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Extract runtime cost information from response when cost config is 'runtime'."""
    if cost_config != "runtime":
        return None

    runtime_costs = None

    try:
        # Only extract costs if model is configured for runtime pricing
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage

            # Extract cost information from usage field (providers may return cost in USD)
            if hasattr(usage, 'cost'):
                total_cost_usd = usage.cost

                # Distribute the cost proportionally between input/output based on token counts
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = prompt_tokens + completion_tokens

                if total_tokens > 0:
                    input_cost_usd = total_cost_usd * (prompt_tokens / total_tokens)
                    output_cost_usd = total_cost_usd * (completion_tokens / total_tokens)
                else:
                    input_cost_usd = total_cost_usd / 2
                    output_cost_usd = total_cost_usd / 2

                # Extract provider information (which actual provider was used)
                actual_provider = None
                if hasattr(response, 'provider'):
                    actual_provider = response.provider
                else:
                    # Try model_dump fallback for provider info
                    try:
                        response_dict = response.model_dump()
                        actual_provider = response_dict.get('provider')
                    except Exception:
                        pass

                runtime_costs = {
                    "input_cost_usd": input_cost_usd,
                    "output_cost_usd": output_cost_usd,
                    "reasoning_cost_usd": 0.0,  # Most providers include reasoning in output cost
                    "total_cost_usd": total_cost_usd,
                    "actual_provider": actual_provider  # Which provider was actually used
                }

                logger.debug(f"Runtime costs extracted: ${total_cost_usd:.8f} USD")

    except Exception as e:
        logger.debug(f"Could not extract runtime costs: {e}")

    return runtime_costs