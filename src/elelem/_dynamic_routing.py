"""
Dynamic routing for virtual model candidate selection.

This module provides real-time performance-based routing using
observed performance data from recent requests.

Exploration is handled via epsilon-greedy in _benchmark_store.py.
"""

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger("elelem")


@dataclass
class DynamicStats:
    """Performance statistics for a model/provider combination."""

    model_ref: str
    avg_tokens_per_sec: float
    sample_count: int
    last_updated: datetime


class DynamicRoutingStore:
    """Per-container cached dynamic routing stats.

    This store queries the metrics database for recent performance data
    and caches results to minimize DB load. Each container maintains its
    own cache, which is refreshed based on TTL.

    Exploration is handled via epsilon-greedy in _benchmark_store.py.
    """

    def __init__(
        self,
        metrics_store: Any,  # MetricsStore, but avoid circular import
        cache_ttl: Optional[int] = None,
        window_minutes: Optional[int] = None,
        max_samples: Optional[int] = None,
        cooldown_minutes: Optional[int] = None,
    ):
        """Initialize dynamic routing store.

        Args:
            metrics_store: MetricsStore instance for DB access
            cache_ttl: Cache TTL in seconds (default: 30)
            window_minutes: Time window for recent data (default: 120 = 2 hours)
            max_samples: Max samples per model for averaging (default: 5)
            cooldown_minutes: Cooldown period for failed candidates (default: 5)
        """
        self._metrics_store = metrics_store
        self._cache_ttl = cache_ttl or int(
            os.getenv("ELELEM_DYNAMIC_ROUTING_CACHE_TTL", "30")
        )
        self._window_minutes = window_minutes or int(
            os.getenv("ELELEM_DYNAMIC_ROUTING_WINDOW_MINUTES", "240")
        )
        self._max_samples = max_samples or int(
            os.getenv("ELELEM_DYNAMIC_ROUTING_MAX_SAMPLES", "5")
        )
        self._cooldown_minutes = cooldown_minutes or int(
            os.getenv("ELELEM_DYNAMIC_ROUTING_COOLDOWN_MINUTES", "15")
        )

        self._cache: Dict[str, DynamicStats] = {}
        self._cache_time: Optional[datetime] = None
        self._failed: Dict[str, datetime] = {}  # candidate -> failure timestamp
        self._lock = threading.Lock()

        # Check if dynamic routing is enabled
        self._enabled = os.getenv("ELELEM_DYNAMIC_ROUTING_ENABLED", "true").lower() == "true"

    @property
    def enabled(self) -> bool:
        """Check if dynamic routing is enabled."""
        return self._enabled and self._metrics_store is not None

    def get_dynamic_stats(self) -> Dict[str, DynamicStats]:
        """Get cached stats, refresh if stale.

        Returns:
            Dict mapping model_ref to DynamicStats
        """
        if not self.enabled:
            return {}

        with self._lock:
            now = datetime.utcnow()

            # Check if cache is valid
            if self._cache_time and (now - self._cache_time).total_seconds() < self._cache_ttl:
                return self._cache.copy()

            # Refresh cache
            try:
                self._cache = self._fetch_stats_from_db()
                self._cache_time = now
                if self._cache:
                    # Log model stats at INFO level so users can see what's happening
                    # Show provider:model format for clarity
                    stats_summary = ", ".join(
                        f"{ref}:{s.sample_count}x@{s.avg_tokens_per_sec:.0f}t/s"
                        for ref, s in sorted(self._cache.items(), key=lambda x: -x[1].avg_tokens_per_sec)[:5]
                    )
                    logger.info(f"🔄 Dynamic routing: {len(self._cache)} models tracked [{stats_summary}]")
            except Exception as e:
                logger.warning(f"Dynamic routing: failed to fetch stats: {e}")
                # Return stale cache on error
                if self._cache:
                    return self._cache.copy()
                return {}

            return self._cache.copy()

    def mark_failed(self, candidate_ref: str) -> None:
        """Mark a candidate as failed (starts cooldown period).

        Args:
            candidate_ref: The model reference (e.g., "gmi:openai/gpt-oss-120b")
        """
        with self._lock:
            self._failed[candidate_ref] = datetime.utcnow()
            logger.info(f"🚫 Cooldown: {candidate_ref} marked failed ({self._cooldown_minutes}min cooldown)")

    def get_failed_candidates(self) -> set[str]:
        """Get candidates that failed recently and are in cooldown.

        Returns:
            Set of model_ref strings that should be skipped
        """
        with self._lock:
            now = datetime.utcnow()
            cooldown_delta = timedelta(minutes=self._cooldown_minutes)

            # Clean up expired entries and return active ones
            active_failures = set()
            expired = []

            for candidate, failure_time in self._failed.items():
                if now - failure_time < cooldown_delta:
                    active_failures.add(candidate)
                else:
                    expired.append(candidate)

            # Remove expired entries
            for candidate in expired:
                del self._failed[candidate]

            return active_failures

    def _fetch_stats_from_db(self) -> Dict[str, DynamicStats]:
        """Query recent metrics aggregated by selected_candidate.

        Returns:
            Dict mapping model_ref to DynamicStats
        """
        if not self._metrics_store or not hasattr(self._metrics_store, "engine"):
            return {}

        from sqlalchemy import text

        # Calculate time window
        window_start = datetime.utcnow() - timedelta(minutes=self._window_minutes)

        # Query aggregated stats using pre-computed total_tokens_per_second
        # This field uses llm_duration (actual API call time) for accurate tokens/sec
        # Use window function to get only the N most recent samples per model
        query = text("""
            WITH ranked AS (
                SELECT
                    selected_candidate,
                    total_tokens_per_second,
                    ROW_NUMBER() OVER (
                        PARTITION BY selected_candidate
                        ORDER BY timestamp DESC
                    ) as rn
                FROM request_metrics
                WHERE status = 'success'
                  AND timestamp > :window_start
                  AND selected_candidate IS NOT NULL
                  AND total_tokens_per_second > 0
            )
            SELECT
                selected_candidate,
                AVG(total_tokens_per_second) as avg_tokens_per_sec,
                COUNT(*) as sample_count
            FROM ranked
            WHERE rn <= :max_samples
            GROUP BY selected_candidate
        """)

        try:
            with self._metrics_store.engine.connect() as conn:
                result = conn.execute(query, {
                    "window_start": window_start,
                    "max_samples": self._max_samples
                })
                rows = result.fetchall()

            stats = {}
            now = datetime.utcnow()
            for row in rows:
                model_ref = row[0]
                avg_tps = row[1]
                count = row[2]

                if model_ref and avg_tps and avg_tps > 0:
                    stats[model_ref] = DynamicStats(
                        model_ref=model_ref,
                        avg_tokens_per_sec=float(avg_tps),
                        sample_count=int(count),
                        last_updated=now,
                    )

            return stats

        except Exception as e:
            logger.error(f"Dynamic routing DB query failed: {e}")
            return {}

    def invalidate_cache(self):
        """Force cache refresh on next access."""
        with self._lock:
            self._cache_time = None
