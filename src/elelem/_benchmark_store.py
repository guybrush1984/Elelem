"""
Candidate reordering for dynamic routing.

Reorders candidates based on observed performance (tokens/sec) and cost.
Uses epsilon-greedy exploration to discover faster providers.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional


# Routing statistics (module-level for persistence across requests)
_routing_total: int = 0
_routing_explorations: int = 0


def get_routing_stats() -> Dict[str, Any]:
    """Get routing statistics for monitoring.

    Returns:
        Dict with total, explorations, and exploration_rate
    """
    total = _routing_total
    explorations = _routing_explorations
    rate = explorations / total if total > 0 else 0.0
    return {
        "total": total,
        "explorations": explorations,
        "exploration_rate": round(rate, 4),
    }


def reset_routing_stats():
    """Reset routing statistics (for testing)."""
    global _routing_total, _routing_explorations
    _routing_total = 0
    _routing_explorations = 0


def reorder_candidates_by_benchmark(
    candidates: List[Dict[str, Any]],
    speed_weight: float = 1.5,
    min_tokens_per_sec: float = 0.0,
    dynamic_stats: Optional[Dict[str, Any]] = None,
    failed_candidates: Optional[set] = None,
    logger: Optional[logging.Logger] = None,
    request_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Reorder candidates by value score from dynamic performance stats.

    Value score = (tokens_per_second ^ speed_weight) / cost_per_1m

    Candidates are grouped by priority:
    1. always_first - Always tried first, in YAML order among themselves
    2. scored - Have dynamic data, sorted by value score (descending)
    3. unscored - No data at all, in YAML order among themselves
    4. always_last - Always tried last (fallbacks), in YAML order among themselves

    The 'priority' field on candidates controls ordering:
    - 'always_first': Skip routing, always tried first
    - 'always_last': Skip routing, always tried last (for fallbacks)
    - Default (no priority): Routing-based if scored, otherwise middle

    Exploration: With adaptive epsilon (10-100%), candidates are shuffled
    randomly to discover faster providers. Higher epsilon when more providers
    are unexplored (cold start), lower when most have been tested.

    Failed candidates in cooldown are excluded from the candidate list entirely.

    Args:
        candidates: List of resolved candidate dicts (must have 'original_model_ref')
        speed_weight: Exponent for speed in value calculation (default 1.5)
        min_tokens_per_sec: Minimum speed threshold, 0 = no filter
        dynamic_stats: Dict of model_ref -> DynamicStats from recent requests
        failed_candidates: Set of model_refs currently in cooldown (excluded)
        logger: Optional logger for debug output
        request_id: Optional request ID for logging

    Returns:
        Reordered candidate list (never empty if input was non-empty)
    """
    global _routing_total, _routing_explorations

    if not candidates:
        return candidates

    log = logger or logging.getLogger("elelem.routing")
    has_dynamic = bool(dynamic_stats)

    # Filter out failed candidates in cooldown
    if failed_candidates:
        original_count = len(candidates)
        candidates = [
            c for c in candidates
            if c.get('original_model_ref') not in failed_candidates
        ]
        excluded_count = original_count - len(candidates)
        if excluded_count > 0:
            log.info(f"🚫 Excluded {excluded_count} candidate(s) in cooldown: {sorted(failed_candidates)}")

        if not candidates:
            log.warning("All candidates are in cooldown!")
            return []

    # Adaptive epsilon-greedy exploration
    # Higher epsilon when many providers are unexplored (cold start)
    # Lower epsilon when all providers have been tested (steady state)
    min_epsilon = float(os.environ.get('ELELEM_EXPLORATION_EPSILON', '0.1'))
    max_epsilon = float(os.environ.get('ELELEM_EXPLORATION_EPSILON_MAX', '1.0'))

    # Calculate exploration coverage: what % of candidates have dynamic data?
    routable_candidates = [
        c for c in candidates
        if (c.get('priority') or '').lower() not in ('always_first', 'always_last')
    ]
    if routable_candidates and has_dynamic and dynamic_stats:
        explored_count = sum(
            1 for c in routable_candidates
            if c.get('original_model_ref') in dynamic_stats
        )
        explored_ratio = explored_count / len(routable_candidates)
        unexplored_ratio = 1.0 - explored_ratio
    else:
        unexplored_ratio = 1.0  # No dynamic data = all unexplored

    # Adaptive epsilon: lerp between min and max based on unexplored ratio
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * unexplored_ratio
    explore_this_request = random.random() < epsilon

    # Record routing decision for statistics
    _routing_total += 1
    if explore_this_request:
        _routing_explorations += 1
        coverage_pct = (1 - unexplored_ratio) * 100
        log.info(f"🎲 Exploration mode (ε={epsilon:.0%}, coverage={coverage_pct:.0f}%): randomizing candidate order")

    # Group candidates by priority and score
    always_first = []  # priority: always_first
    scored = []        # Have dynamic score
    unscored = []      # No performance data
    filtered_out = []  # Below min_tokens_per_sec threshold
    always_last = []   # priority: always_last (fallbacks)

    for idx, candidate in enumerate(candidates):
        priority = (candidate.get('priority') or '').lower()

        # Handle always_first priority - skip all routing logic
        if priority == 'always_first':
            always_first.append((idx, candidate))
            continue

        # Handle always_last priority - skip all routing logic, kept at end
        if priority == 'always_last':
            always_last.append((idx, candidate))
            continue

        model_ref = candidate.get('original_model_ref')

        if not model_ref:
            # No model ref, keep in unscored
            unscored.append((idx, candidate))
            continue

        # Get cost from candidate's model config (YAML) - always authoritative
        model_cost = candidate.get('cost', {})
        cost_per_1m = model_cost.get('output_cost_per_1m', 0)

        # Get dynamic stats (observed tokens/s from recent requests)
        tps = None
        sample_count = 0

        if has_dynamic and model_ref in dynamic_stats:
            stats = dynamic_stats[model_ref]
            sample_count = stats.sample_count
            tps = stats.avg_tokens_per_sec

            # Check minimum speed threshold
            if min_tokens_per_sec > 0 and tps < min_tokens_per_sec:
                filtered_out.append((idx, candidate, model_ref))
                continue

        # Calculate value score if we have dynamic data
        if tps is not None and tps > 0:
            # Value = speed^weight / cost
            if cost_per_1m and cost_per_1m > 0:
                value_score = (tps ** speed_weight) / cost_per_1m
            else:
                # No cost data, use raw speed as score
                value_score = tps ** speed_weight

            # Store scores in candidate for logging/debugging
            candidate['_value_score'] = value_score
            candidate['_tps'] = tps
            candidate['_sample_count'] = sample_count
            scored.append((value_score, sample_count, idx, candidate))
        else:
            unscored.append((idx, candidate))

    # Check if ALL routable candidates were filtered out - fallback to YAML order
    if filtered_out and not scored and not unscored:
        log.warning(
            f"All {len(filtered_out)} routable candidates below {min_tokens_per_sec} t/s threshold, "
            f"falling back to YAML order for non-priority candidates"
        )
        result = [c for (_, c) in always_first]
        for idx, candidate in enumerate(candidates):
            priority = (candidate.get('priority') or '').lower()
            if priority not in ('always_first', 'always_last'):
                result.append(candidate)
        result.extend([c for (_, c) in always_last])
        return result

    # Log filtering (only if some were filtered but not all)
    if filtered_out:
        filtered_refs = [ref for (_, _, ref) in filtered_out]
        log.debug(f"Filtered out {len(filtered_out)} candidates below {min_tokens_per_sec} t/s: {filtered_refs}")

    # Sort scored candidates by value score (descending - higher is better)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Combine scored and unscored for the routable middle section
    routable = [c for (_, _, _, c) in scored] + [c for (_, c) in unscored]

    # Epsilon-greedy exploration: favor models with fewer samples
    if explore_this_request and routable:
        max_samples = int(os.environ.get('ELELEM_DYNAMIC_ROUTING_MAX_SAMPLES', '5'))

        unscored_explore = [c for c in routable if c.get('_sample_count', 0) == 0]
        low_sample = [c for c in routable if 0 < c.get('_sample_count', 0) < max_samples]
        well_known = [c for c in routable if c.get('_sample_count', 0) >= max_samples]

        # Shuffle within each group
        random.shuffle(unscored_explore)
        random.shuffle(low_sample)
        random.shuffle(well_known)

        # Combine: unknowns first, then learning, then well-known
        routable = unscored_explore + low_sample + well_known

        if unscored_explore or low_sample:
            unknown_refs = [c.get('original_model_ref', '?') for c in unscored_explore]
            learning_refs = [f"{c.get('original_model_ref', '?')}({c.get('_sample_count', 0)})" for c in low_sample]
            log.debug(f"Exploration prioritizing unknowns: {unknown_refs}, learning: {learning_refs}")

    # Build result: always_first → routable (scored+unscored) → always_last
    result = [c for (_, c) in always_first]
    result.extend(routable)
    result.extend([c for (_, c) in always_last])

    # Debug-level details
    if log.isEnabledFor(logging.DEBUG):
        if always_first:
            first_refs = [c.get('original_model_ref', 'unknown') for (_, c) in always_first]
            log.debug(f"Priority always_first: {first_refs}")
        if scored:
            order_info = [(c.get('original_model_ref'), round(s, 2), n) for s, n, _, c in scored]
            log.debug(f"Routing reorder (speed_weight={speed_weight}): {order_info}")
        if always_last:
            last_refs = [c.get('original_model_ref', 'unknown') for (_, c) in always_last]
            log.debug(f"Priority always_last (fallbacks): {last_refs}")

    return result
