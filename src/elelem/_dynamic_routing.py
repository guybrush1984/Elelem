"""
Dynamic routing for virtual model candidate selection.

This module provides real-time performance-based routing that blends
observed performance with static benchmark data (gist).

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

    Exploration is handled via epsilon-greedy in _benchmark_store.py (10% random).
    """

    def __init__(
        self,
        metrics_store: Any,  # MetricsStore, but avoid circular import
        cache_ttl: Optional[int] = None,
        window_minutes: Optional[int] = None,
    ):
        """Initialize dynamic routing store.

        Args:
            metrics_store: MetricsStore instance for DB access
            cache_ttl: Cache TTL in seconds (default: 30)
            window_minutes: Time window for recent data (default: 30)
        """
        self._metrics_store = metrics_store
        self._cache_ttl = cache_ttl or int(
            os.getenv("ELELEM_DYNAMIC_ROUTING_CACHE_TTL", "30")
        )
        self._window_minutes = window_minutes or int(
            os.getenv("ELELEM_DYNAMIC_ROUTING_WINDOW_MINUTES", "30")
        )

        self._cache: Dict[str, DynamicStats] = {}
        self._cache_time: Optional[datetime] = None
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
        query = text("""
            SELECT
                selected_candidate,
                AVG(total_tokens_per_second) as avg_tokens_per_sec,
                COUNT(*) as sample_count
            FROM request_metrics
            WHERE status = 'success'
              AND timestamp > :window_start
              AND selected_candidate IS NOT NULL
              AND total_tokens_per_second > 0
            GROUP BY selected_candidate
        """)

        try:
            with self._metrics_store.engine.connect() as conn:
                result = conn.execute(query, {"window_start": window_start})
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


def blend_scores(
    gist_tps: Optional[float],
    dynamic_avg_tps: Optional[float],
    sample_count: int,
) -> tuple[Optional[float], float]:
    """Blend gist and dynamic scores by treating gist as 1 additional sample.

    Gist is treated as a single synthetic sample. The final score is a
    weighted average where gist has weight 1 and dynamic has weight sample_count.

    Formula: (gist * 1 + dynamic_avg * sample_count) / (1 + sample_count)

    As more real samples come in, gist influence naturally shrinks:
    - 0 dynamic samples: 100% gist
    - 1 dynamic sample: 50% gist, 50% dynamic
    - 4 dynamic samples: 20% gist, 80% dynamic
    - 9 dynamic samples: 10% gist, 90% dynamic

    Args:
        gist_tps: Static benchmark tokens/sec (may be None)
        dynamic_avg_tps: Average observed tokens/sec (may be None)
        sample_count: Number of dynamic samples

    Returns:
        Tuple of (blended_tps, gist_weight) where gist_weight is 1/(1+sample_count)
    """
    if gist_tps is not None and dynamic_avg_tps is not None and sample_count > 0:
        # Weighted average: gist counts as 1 sample
        total_samples = 1 + sample_count
        blended = (gist_tps + dynamic_avg_tps * sample_count) / total_samples
        gist_weight = 1.0 / total_samples
        return blended, gist_weight
    elif dynamic_avg_tps is not None and sample_count > 0:
        # No gist, use dynamic only
        return dynamic_avg_tps, 0.0
    elif gist_tps is not None:
        # No dynamic, use gist only
        return gist_tps, 1.0
    else:
        # No data at all
        return None, 0.0
