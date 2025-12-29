"""
Benchmark data store for dynamic candidate reordering.

Fetches benchmark results from a local file or URL and stores them in memory
for use in candidate reordering based on value score (speed/cost ratio).

Supports the exact format output by telelem_simple.py batch_summary.json.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx


class BenchmarkStore:
    """Thread-safe in-memory store for benchmark data with background fetching."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("elelem.benchmark")
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._last_fetch: Optional[datetime] = None
        self._fetch_error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

        # Routing statistics (for monitoring epsilon-greedy exploration)
        self._routing_total: int = 0
        self._routing_explorations: int = 0

        # Configuration from environment
        self._source = os.getenv('ELELEM_BENCHMARK_SOURCE')  # File path or URL
        self._interval = max(60, int(os.getenv('ELELEM_BENCHMARK_FETCH_INTERVAL', '3600')))
        self._timeout = int(os.getenv('ELELEM_BENCHMARK_FETCH_TIMEOUT', '30'))

    @property
    def enabled(self) -> bool:
        """Check if benchmark fetching is enabled (source configured)."""
        # Re-read from env var for runtime changes (tests)
        return bool(os.getenv('ELELEM_BENCHMARK_SOURCE'))

    @property
    def source(self) -> Optional[str]:
        """Get the configured benchmark source."""
        # Re-read from env var for runtime changes (tests)
        return os.getenv('ELELEM_BENCHMARK_SOURCE')

    def get_benchmark(self, model_ref: str) -> Optional[Dict[str, Any]]:
        """Get benchmark data for a model reference (thread-safe).

        Args:
            model_ref: Model reference string (e.g., "fireworks:deepseek/deepseek-3.2")

        Returns:
            Processed benchmark dict with tokens_per_second, cost_per_request, etc.
            or None if not found
        """
        with self._lock:
            return self._data.get(model_ref)

    def get_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Get all benchmark data (thread-safe copy)."""
        with self._lock:
            return dict(self._data)

    def record_routing_decision(self, is_exploration: bool):
        """Record a routing decision for statistics.

        Args:
            is_exploration: True if this was an exploration (random shuffle)
        """
        self._routing_total += 1
        if is_exploration:
            self._routing_explorations += 1

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring.

        Returns:
            Dict with total, explorations, and exploration_rate
        """
        total = self._routing_total
        explorations = self._routing_explorations
        rate = explorations / total if total > 0 else 0.0
        return {
            "total": total,
            "explorations": explorations,
            "exploration_rate": round(rate, 4),
        }

    def reset_routing_stats(self):
        """Reset routing statistics (for testing)."""
        self._routing_total = 0
        self._routing_explorations = 0

    def calculate_value_score(
        self,
        model_ref: str,
        speed_weight: float = 1.0,
        min_tokens_per_sec: float = 0.0
    ) -> Optional[float]:
        """Calculate value score for a candidate.

        Value score = tokens_per_second^speed_weight / cost_per_1m_output

        Higher speed_weight (>1) favors speed more.
        Lower speed_weight (<1) favors cost more.
        speed_weight=1 is balanced (default).

        Args:
            model_ref: Model reference string
            speed_weight: Exponent for speed (default 1.0 = balanced)
            min_tokens_per_sec: Minimum speed threshold (0 = no filter)

        Returns:
            Value score, or None if no data or filtered out
        """
        benchmark = self.get_benchmark(model_ref)
        if not benchmark:
            return None

        tokens_per_second = benchmark.get('tokens_per_second', 0)
        cost_per_1m = benchmark.get('cost_per_1m_output', 0)

        # Filter by minimum speed
        if min_tokens_per_sec > 0 and tokens_per_second < min_tokens_per_sec:
            return None

        if tokens_per_second <= 0 or cost_per_1m <= 0:
            return None

        # Value = speed^weight / cost
        return (tokens_per_second ** speed_weight) / cost_per_1m

    def _parse_telelem_format(self, raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Parse telelem batch_summary.json format into normalized benchmark data.

        Args:
            raw_data: Raw JSON from batch_summary.json

        Returns:
            Dict mapping model_ref -> {tokens_per_second, cost_per_1m_output, ...}
        """
        # Check data freshness (discard if older than 6 hours)
        timestamp_str = raw_data.get('timestamp')
        if timestamp_str:
            try:
                # Parse timestamp (format: "2025-12-11 08:04:09 UTC")
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S %Z')
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - timestamp
                max_age_hours = int(os.getenv('ELELEM_BENCHMARK_MAX_AGE_HOURS', '6'))

                if age.total_seconds() > max_age_hours * 3600:
                    self.logger.warning(
                        f"Benchmark data is stale ({age.total_seconds() / 3600:.1f}h old, "
                        f"max {max_age_hours}h). Discarding."
                    )
                    return {}
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")

        result = {}
        models = raw_data.get('models', {})

        for model_ref, model_data in models.items():
            try:
                # Extract output tokens and duration
                output_tokens = model_data.get('tokens', {}).get('output', {}).get('avg', 0)
                duration = model_data.get('duration', {}).get('avg', 0)
                cost_total = model_data.get('costs', {}).get('avg', 0)
                success_rate = model_data.get('requests', {}).get('success_rate', 0)

                # Calculate tokens per second
                tokens_per_second = output_tokens / duration if duration > 0 else 0

                # Calculate cost per 1M output tokens (from actual cost and tokens)
                # cost_total is for ~output_tokens, so extrapolate to 1M
                cost_per_1m_output = (cost_total / output_tokens * 1_000_000) if output_tokens > 0 else 0

                result[model_ref] = {
                    'tokens_per_second': round(tokens_per_second, 2),
                    'cost_per_1m_output': round(cost_per_1m_output, 4),
                    'avg_duration': round(duration, 3),
                    'avg_output_tokens': round(output_tokens, 0),
                    'success_rate': success_rate,
                    'sample_count': model_data.get('requests', {}).get('total', 0)
                }
            except (KeyError, TypeError, ZeroDivisionError) as e:
                self.logger.warning(f"Failed to parse benchmark for {model_ref}: {e}")
                continue

        return result

    async def fetch_once(self) -> bool:
        """Fetch benchmark data once from the configured source.

        Supports:
        - Local file paths (absolute or relative)
        - file:// URLs
        - http:// and https:// URLs

        Returns:
            True if fetch succeeded, False otherwise
        """
        # Re-read source from env var (allows runtime changes for tests)
        self._source = os.getenv('ELELEM_BENCHMARK_SOURCE')

        if not self._source:
            return False

        try:
            raw_data = await self._load_source()
            if raw_data is None:
                return False

            # Parse telelem format
            parsed = self._parse_telelem_format(raw_data)

            if not parsed:
                self.logger.warning("Benchmark source contained no valid model data")
                return False

            # Update store (thread-safe)
            with self._lock:
                self._data = parsed
                self._last_fetch = datetime.now(timezone.utc)
                self._fetch_error = None

            self.logger.info(
                f"Loaded {len(self._data)} benchmark entries from {self._source}"
            )
            return True

        except Exception as e:
            self._fetch_error = str(e)
            self.logger.error(f"Benchmark fetch error: {self._fetch_error}")
            return False

    async def _load_source(self) -> Optional[Dict[str, Any]]:
        """Load JSON from the configured source (file or URL)."""
        source = self._source

        # Handle file:// URLs
        if source.startswith('file://'):
            source = source[7:]  # Strip file://

        # Check if it's a local file path
        if not source.startswith(('http://', 'https://')):
            return self._load_file(source)

        # HTTP(S) URL
        return await self._load_url(source)

    def _load_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Load JSON from a local file."""
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                # Relative to current working directory
                file_path = Path.cwd() / file_path

            if not file_path.exists():
                self._fetch_error = f"File not found: {file_path}"
                self.logger.warning(self._fetch_error)
                return None

            with open(file_path, 'r') as f:
                return json.load(f)

        except json.JSONDecodeError as e:
            self._fetch_error = f"Invalid JSON in file: {e}"
            self.logger.error(self._fetch_error)
            return None
        except Exception as e:
            self._fetch_error = f"File read error: {e}"
            self.logger.error(self._fetch_error)
            return None

    async def _load_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Load JSON from an HTTP(S) URL."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            self._fetch_error = f"HTTP {e.response.status_code}"
            self.logger.warning(f"Benchmark fetch failed: {self._fetch_error}")
            return None
        except httpx.RequestError as e:
            self._fetch_error = str(e)
            self.logger.warning(f"Benchmark fetch failed: {self._fetch_error}")
            return None
        except json.JSONDecodeError as e:
            self._fetch_error = f"Invalid JSON from URL: {e}"
            self.logger.error(self._fetch_error)
            return None

    async def start_background_fetch(self):
        """Start the background fetch task. Call from FastAPI startup."""
        if not self.enabled:
            self.logger.info("Benchmark-based routing disabled (ELELEM_BENCHMARK_SOURCE not set)")
            return

        self.logger.info(
            f"Starting benchmark fetch (interval: {self._interval}s, source: {self._source})"
        )

        # Initial fetch (non-blocking - server starts immediately)
        asyncio.create_task(self._initial_fetch())

        # Start periodic fetch loop
        self._task = asyncio.create_task(self._fetch_loop())

    async def _initial_fetch(self):
        """Perform initial fetch without blocking startup."""
        await self.fetch_once()

    async def _fetch_loop(self):
        """Background loop that fetches benchmarks periodically."""
        while True:
            await asyncio.sleep(self._interval)
            await self.fetch_once()

    def stop(self):
        """Stop the background fetch task. Call from FastAPI shutdown."""
        if self._task:
            self._task.cancel()
            self._task = None

    def get_status(self) -> Dict[str, Any]:
        """Get status information for health checks."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "source": self._source,
                "interval_seconds": self._interval,
                "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
                "last_error": self._fetch_error,
                "entries_count": len(self._data),
            }


# Global singleton for easy access
_benchmark_store: Optional[BenchmarkStore] = None


def get_benchmark_store() -> BenchmarkStore:
    """Get or create the global benchmark store singleton."""
    global _benchmark_store
    if _benchmark_store is None:
        _benchmark_store = BenchmarkStore()
    return _benchmark_store


def reorder_candidates_by_benchmark(
    candidates: List[Dict[str, Any]],
    speed_weight: float = 1.0,
    min_tokens_per_sec: float = 0.0,
    dynamic_stats: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    request_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Reorder candidates by value score from benchmark data, blended with dynamic stats.

    Scoring: gist is treated as 1 synthetic sample, blended with real observations.
    As more real samples come in, gist influence naturally shrinks (1/(1+N)).

    Candidates are grouped by priority:
    1. always_first - Always tried first, in YAML order among themselves
    2. scored - Have benchmark or dynamic data, sorted by blended score (descending)
    3. unscored - No data at all, in YAML order among themselves
    4. always_last - Always tried last (fallbacks), in YAML order among themselves

    The 'priority' field on candidates controls ordering:
    - 'always_first': Skip routing, always tried first
    - 'always_last': Skip routing, always tried last (for fallbacks)
    - Default (no priority): Routing-based if scored, otherwise middle

    If min_tokens_per_sec filter would exclude ALL routable candidates,
    falls back to original YAML order (no filtering applied).

    Exploration: With probability epsilon (default 10%), candidates are shuffled
    randomly to discover faster providers.

    Args:
        candidates: List of resolved candidate dicts (must have 'original_model_ref')
        speed_weight: Exponent for speed in value calculation (default 1.0)
        min_tokens_per_sec: Minimum speed threshold, 0 = no filter
        dynamic_stats: Optional dict of model_ref -> DynamicStats for blending
        logger: Optional logger for debug output

    Returns:
        Reordered candidate list (never empty if input was non-empty)
    """
    store = get_benchmark_store()

    # Can proceed if either gist or dynamic data is available
    has_gist = store.enabled
    has_dynamic = bool(dynamic_stats)

    if not (has_gist or has_dynamic) or not candidates:
        return candidates

    log = logger or logging.getLogger("elelem.benchmark")

    # Epsilon-greedy exploration: with probability epsilon, randomize order
    # Epsilon-greedy works well in serverless (stateless)
    import random
    epsilon = float(os.environ.get('ELELEM_EXPLORATION_EPSILON', '0.1'))
    explore_this_request = random.random() < epsilon

    # Record routing decision for statistics
    store.record_routing_decision(explore_this_request)

    if explore_this_request:
        log.info(f"🎲 Exploration mode (ε={epsilon:.0%}): randomizing candidate order")

    # Group candidates by priority and score
    always_first = []  # priority: always_first
    scored = []        # Have benchmark score
    unscored = []      # No benchmark data
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

        # Get gist benchmark data (raw tokens_per_second and cost)
        benchmark = store.get_benchmark(model_ref) if has_gist else None
        gist_tps = None  # tokens per second from gist
        cost_per_1m = None
        if benchmark:
            gist_tps = benchmark.get('tokens_per_second', 0)
            cost_per_1m = benchmark.get('cost_per_1m_output', 0)
            # Check minimum speed threshold (applies to gist data)
            if min_tokens_per_sec > 0 and gist_tps < min_tokens_per_sec:
                filtered_out.append((idx, candidate, model_ref))
                continue

        # Get dynamic stats (observed tokens/s from recent requests)
        dynamic_tps = None
        sample_count = 0

        if has_dynamic and model_ref in dynamic_stats:
            stats = dynamic_stats[model_ref]
            sample_count = stats.sample_count
            dynamic_tps = stats.avg_tokens_per_sec

        # Blend speeds (gist as 1 sample + dynamic samples), then apply cost
        if gist_tps is not None or dynamic_tps is not None:
            from elelem._dynamic_routing import blend_scores
            # Blend: gist counts as 1 sample, dynamic_tps is avg of sample_count samples
            blended_tps, gist_weight = blend_scores(gist_tps, dynamic_tps, sample_count)

            if blended_tps is not None and blended_tps > 0:
                # Apply cost to get final value score: speed^weight / cost
                if cost_per_1m and cost_per_1m > 0:
                    final_score = (blended_tps ** speed_weight) / cost_per_1m
                else:
                    # No cost data, use raw speed as score
                    final_score = blended_tps ** speed_weight

                # Store scores in candidate for logging/debugging
                candidate['_benchmark_score'] = final_score
                candidate['_gist_tps'] = gist_tps
                candidate['_dynamic_tps'] = dynamic_tps
                candidate['_blended_tps'] = blended_tps
                candidate['_gist_weight'] = gist_weight
                candidate['_sample_count'] = sample_count
                scored.append((final_score, sample_count, idx, candidate))

                # Debug log for score blending
                if log.isEnabledFor(logging.DEBUG) and (gist_tps is not None and dynamic_tps is not None):
                    log.debug(
                        f"Model {model_ref}: gist={gist_tps:.1f}, dynamic={dynamic_tps:.1f}, "
                        f"blended={blended_tps:.1f}, gist_weight={gist_weight:.0%}, final={final_score:.2f}"
                    )
            else:
                unscored.append((idx, candidate))
        else:
            unscored.append((idx, candidate))

    # Check if ALL routable candidates were filtered out - fallback to YAML order
    # (always_first and always_last candidates are not affected by this check)
    if filtered_out and not scored and not unscored:
        log.warning(
            f"All {len(filtered_out)} routable candidates below {min_tokens_per_sec} t/s threshold, "
            f"falling back to YAML order for non-priority candidates"
        )
        # Return always_first + original order for the rest + always_last
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

    # Epsilon-greedy exploration: randomly shuffle if in exploration mode
    if explore_this_request and scored:
        random.shuffle(scored)

    # Build result: always_first, then scored, then unscored, then always_last
    result = [c for (_, c) in always_first]
    result.extend([c for (_, _, _, c) in scored])
    result.extend([c for (_, c) in unscored])
    result.extend([c for (_, c) in always_last])


    # Debug-level details
    if log.isEnabledFor(logging.DEBUG):
        if always_first:
            first_refs = [c.get('original_model_ref', 'unknown') for (_, c) in always_first]
            log.debug(f"Priority always_first: {first_refs}")
        if scored:
            order_info = [(c.get('original_model_ref'), round(s, 2), n) for s, n, _, c in scored]
            log.debug(f"Benchmark reorder (speed_weight={speed_weight}): {order_info}")
        if always_last:
            last_refs = [c.get('original_model_ref', 'unknown') for (_, c) in always_last]
            log.debug(f"Priority always_last (fallbacks): {last_refs}")

    return result
