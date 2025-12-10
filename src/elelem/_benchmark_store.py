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

        # Configuration from environment
        self._source = os.getenv('ELELEM_BENCHMARK_SOURCE')  # File path or URL
        self._interval = max(60, int(os.getenv('ELELEM_BENCHMARK_FETCH_INTERVAL', '3600')))
        self._timeout = int(os.getenv('ELELEM_BENCHMARK_FETCH_TIMEOUT', '30'))

    @property
    def enabled(self) -> bool:
        """Check if benchmark fetching is enabled (source configured)."""
        return bool(self._source)

    @property
    def source(self) -> Optional[str]:
        """Get the configured benchmark source."""
        return self._source

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
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """Reorder candidates by value score from benchmark data.

    Candidates are grouped by priority:
    1. always_first - Always tried first, in YAML order among themselves
    2. scored - Have benchmark data, sorted by value score (descending)
    3. unscored - No benchmark data, in YAML order among themselves

    The 'priority' field on candidates controls ordering:
    - 'always_first': Skip routing, always tried first
    - Default (no priority): Routing-based if scored, otherwise last

    If min_tokens_per_sec filter would exclude ALL routable candidates,
    falls back to original YAML order (no filtering applied).

    Args:
        candidates: List of resolved candidate dicts (must have 'original_model_ref')
        speed_weight: Exponent for speed in value calculation (default 1.0)
        min_tokens_per_sec: Minimum speed threshold, 0 = no filter
        logger: Optional logger for debug output

    Returns:
        Reordered candidate list (never empty if input was non-empty)
    """
    store = get_benchmark_store()

    if not store.enabled or not candidates:
        return candidates

    log = logger or logging.getLogger("elelem.benchmark")

    # Group candidates by priority and score
    always_first = []  # priority: always_first
    scored = []        # Have benchmark score
    unscored = []      # No benchmark data
    filtered_out = []  # Below min_tokens_per_sec threshold

    for idx, candidate in enumerate(candidates):
        priority = (candidate.get('priority') or '').lower()

        # Handle always_first priority - skip all routing logic
        if priority == 'always_first':
            always_first.append((idx, candidate))
            continue

        model_ref = candidate.get('original_model_ref')

        if not model_ref:
            # No model ref, keep in unscored
            unscored.append((idx, candidate))
            continue

        benchmark = store.get_benchmark(model_ref)

        if not benchmark:
            # No benchmark data, keep in unscored
            unscored.append((idx, candidate))
            continue

        tokens_per_sec = benchmark.get('tokens_per_second', 0)

        # Check minimum speed threshold
        if min_tokens_per_sec > 0 and tokens_per_sec < min_tokens_per_sec:
            filtered_out.append((idx, candidate, model_ref))
            continue

        # Calculate value score
        score = store.calculate_value_score(model_ref, speed_weight, min_tokens_per_sec)

        if score is not None:
            # Store score in candidate for logging purposes
            candidate['_benchmark_score'] = score
            scored.append((score, idx, candidate))
        else:
            unscored.append((idx, candidate))

    # Check if ALL routable candidates were filtered out - fallback to YAML order
    # (always_first candidates are not affected by this check)
    if filtered_out and not scored and not unscored:
        log.warning(
            f"All {len(filtered_out)} routable candidates below {min_tokens_per_sec} t/s threshold, "
            f"falling back to YAML order for non-priority candidates"
        )
        # Return always_first + original order for the rest
        result = [c for (_, c) in always_first]
        for idx, candidate in enumerate(candidates):
            if candidate.get('priority', '').lower() != 'always_first':
                result.append(candidate)
        return result

    # Log filtering (only if some were filtered but not all)
    if filtered_out:
        filtered_refs = [ref for (_, _, ref) in filtered_out]
        log.debug(f"Filtered out {len(filtered_out)} candidates below {min_tokens_per_sec} t/s: {filtered_refs}")

    # Sort scored candidates by value score (descending - higher is better)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build result: always_first, then scored, then unscored
    result = [c for (_, c) in always_first]
    result.extend([c for (_, _, c) in scored])
    result.extend([c for (_, c) in unscored])

    # Log reordering
    if log.isEnabledFor(logging.DEBUG):
        if always_first:
            first_refs = [c.get('original_model_ref', 'unknown') for (_, c) in always_first]
            log.debug(f"Priority always_first: {first_refs}")
        if scored:
            order_info = [(c.get('original_model_ref'), round(s, 2)) for s, _, c in scored]
            log.debug(f"Benchmark reorder (speed_weight={speed_weight}): {order_info}")

    return result
