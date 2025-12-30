"""
Tests for dynamic routing with epsilon-greedy exploration.
"""

import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from elelem._dynamic_routing import (
    DynamicStats,
    DynamicRoutingStore,
)


class TestDynamicStats:
    """Tests for DynamicStats dataclass."""

    def test_create_dynamic_stats(self):
        """Should create DynamicStats with all fields."""
        now = datetime.utcnow()
        stats = DynamicStats(
            model_ref="gmi:deepseek/deepseek-3.2",
            avg_tokens_per_sec=150.5,
            sample_count=10,
            last_updated=now,
        )
        assert stats.model_ref == "gmi:deepseek/deepseek-3.2"
        assert stats.avg_tokens_per_sec == 150.5
        assert stats.sample_count == 10
        assert stats.last_updated == now


class TestDynamicRoutingStore:
    """Tests for DynamicRoutingStore class."""

    def test_disabled_when_env_false(self):
        """Store should be disabled when ELELEM_DYNAMIC_ROUTING_ENABLED=false."""
        with patch.dict(os.environ, {"ELELEM_DYNAMIC_ROUTING_ENABLED": "false"}):
            store = DynamicRoutingStore(metrics_store=None)
            assert not store.enabled

    def test_enabled_by_default(self):
        """Store should be enabled by default with metrics store."""
        with patch.dict(os.environ, {}, clear=True):
            mock_metrics = MagicMock()
            store = DynamicRoutingStore(metrics_store=mock_metrics)
            assert store.enabled

    def test_disabled_without_metrics_store(self):
        """Store should be disabled without metrics store."""
        with patch.dict(os.environ, {}, clear=True):
            store = DynamicRoutingStore(metrics_store=None)
            assert not store.enabled

    def test_get_dynamic_stats_returns_empty_when_disabled(self):
        """Should return empty dict when disabled."""
        with patch.dict(os.environ, {"ELELEM_DYNAMIC_ROUTING_ENABLED": "false"}):
            store = DynamicRoutingStore(metrics_store=MagicMock())
            assert store.get_dynamic_stats() == {}

    def test_cache_respects_ttl(self):
        """Should use cached data within TTL."""
        mock_metrics = MagicMock()
        mock_metrics.engine = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            store = DynamicRoutingStore(metrics_store=mock_metrics, cache_ttl=30)

            # Populate cache
            now = datetime.utcnow()
            store._cache = {
                "test:model": DynamicStats(
                    model_ref="test:model",
                    avg_tokens_per_sec=100.0,
                    sample_count=5,
                    last_updated=now,
                )
            }
            store._cache_time = now

            # Should return cached data without DB query
            with patch.object(store, "_fetch_stats_from_db") as mock_fetch:
                stats = store.get_dynamic_stats()
                mock_fetch.assert_not_called()
                assert "test:model" in stats

    def test_cache_invalidation(self):
        """invalidate_cache should force refresh on next access."""
        mock_metrics = MagicMock()
        mock_metrics.engine = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            store = DynamicRoutingStore(metrics_store=mock_metrics, cache_ttl=30)
            store._cache_time = datetime.utcnow()
            store.invalidate_cache()
            assert store._cache_time is None

    def test_configuration_from_env(self):
        """Should read configuration from environment variables."""
        env_vars = {
            "ELELEM_DYNAMIC_ROUTING_CACHE_TTL": "60",
            "ELELEM_DYNAMIC_ROUTING_WINDOW_MINUTES": "45",
            "ELELEM_DYNAMIC_ROUTING_MAX_SAMPLES": "10",
            "ELELEM_DYNAMIC_ROUTING_COOLDOWN_MINUTES": "20",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            store = DynamicRoutingStore(metrics_store=MagicMock())
            assert store._cache_ttl == 60
            assert store._window_minutes == 45
            assert store._max_samples == 10
            assert store._cooldown_minutes == 20


class TestReorderWithDynamicStats:
    """Tests for reorder_candidates_by_benchmark with dynamic stats."""

    def test_dynamic_only_scoring(self):
        """Should use dynamic stats for scoring."""
        from elelem._benchmark_store import reorder_candidates_by_benchmark

        candidates = [
            {"original_model_ref": "slow:model", "model_id": "slow", "cost": {"output_cost_per_1m": 1.0}},
            {"original_model_ref": "fast:model", "model_id": "fast", "cost": {"output_cost_per_1m": 1.0}},
        ]

        now = datetime.utcnow()
        dynamic_stats = {
            "slow:model": DynamicStats(
                model_ref="slow:model",
                avg_tokens_per_sec=50.0,
                sample_count=10,
                last_updated=now,
            ),
            "fast:model": DynamicStats(
                model_ref="fast:model",
                avg_tokens_per_sec=150.0,
                sample_count=10,
                last_updated=now,
            ),
        }

        # Disable exploration for deterministic test
        with patch.dict(os.environ, {"ELELEM_EXPLORATION_EPSILON": "0", "ELELEM_EXPLORATION_EPSILON_MAX": "0"}):
            result = reorder_candidates_by_benchmark(
                candidates,
                dynamic_stats=dynamic_stats,
            )

        # Fast model should be first
        assert result[0]["model_id"] == "fast"
        assert result[1]["model_id"] == "slow"

    def test_priority_candidates_unaffected(self):
        """always_first and always_last candidates should be unaffected by scoring."""
        from elelem._benchmark_store import reorder_candidates_by_benchmark

        candidates = [
            {"original_model_ref": "slow:model", "model_id": "slow", "cost": {"output_cost_per_1m": 1.0}},
            {"original_model_ref": "first:model", "model_id": "first", "priority": "always_first", "cost": {"output_cost_per_1m": 1.0}},
            {"original_model_ref": "fast:model", "model_id": "fast", "cost": {"output_cost_per_1m": 1.0}},
            {"original_model_ref": "last:model", "model_id": "last", "priority": "always_last", "cost": {"output_cost_per_1m": 1.0}},
        ]

        now = datetime.utcnow()
        dynamic_stats = {
            "slow:model": DynamicStats(
                model_ref="slow:model",
                avg_tokens_per_sec=50.0,
                sample_count=10,
                last_updated=now,
            ),
            "fast:model": DynamicStats(
                model_ref="fast:model",
                avg_tokens_per_sec=150.0,
                sample_count=10,
                last_updated=now,
            ),
            "first:model": DynamicStats(
                model_ref="first:model",
                avg_tokens_per_sec=10.0,  # Very slow but priority
                sample_count=10,
                last_updated=now,
            ),
            "last:model": DynamicStats(
                model_ref="last:model",
                avg_tokens_per_sec=200.0,  # Very fast but always_last
                sample_count=10,
                last_updated=now,
            ),
        }

        # Disable exploration for deterministic test
        with patch.dict(os.environ, {"ELELEM_EXPLORATION_EPSILON_MAX": "0"}):
            result = reorder_candidates_by_benchmark(
                candidates,
                dynamic_stats=dynamic_stats,
            )

        # First position should be always_first
        assert result[0]["model_id"] == "first"
        # Last position should be always_last
        assert result[-1]["model_id"] == "last"
        # Middle should be scored (fast before slow)
        assert result[1]["model_id"] == "fast"
        assert result[2]["model_id"] == "slow"


class TestEpsilonExplorationRate:
    """Tests for epsilon-greedy exploration rate."""

    def test_exploration_rate_matches_epsilon(self):
        """Verify exploration rate is approximately equal to epsilon over many trials."""
        from elelem._benchmark_store import reorder_candidates_by_benchmark, get_routing_stats, reset_routing_stats

        # Set epsilon to 10% (no max, so it's always 10%)
        with patch.dict(os.environ, {"ELELEM_EXPLORATION_EPSILON": "0.1", "ELELEM_EXPLORATION_EPSILON_MAX": "0.1"}):
            # Reset counters
            reset_routing_stats()

            candidates = [
                {"original_model_ref": "model_a", "model_id": "a", "provider": "p1", "cost": {"output_cost_per_1m": 1.0}},
                {"original_model_ref": "model_b", "model_id": "b", "provider": "p2", "cost": {"output_cost_per_1m": 1.0}},
                {"original_model_ref": "model_c", "model_id": "c", "provider": "p3", "cost": {"output_cost_per_1m": 1.0}},
            ]

            now = datetime.utcnow()
            dynamic_stats = {
                "model_a": DynamicStats(model_ref="model_a", avg_tokens_per_sec=100.0, sample_count=10, last_updated=now),
                "model_b": DynamicStats(model_ref="model_b", avg_tokens_per_sec=90.0, sample_count=10, last_updated=now),
                "model_c": DynamicStats(model_ref="model_c", avg_tokens_per_sec=80.0, sample_count=10, last_updated=now),
            }

            num_trials = 1000

            for _ in range(num_trials):
                reorder_candidates_by_benchmark([c.copy() for c in candidates], dynamic_stats=dynamic_stats)

            # Get stats from counters
            stats = get_routing_stats()
            actual_rate = stats["exploration_rate"]

            # With 1000 trials and p=0.1, allow range [0.07, 0.13]
            min_acceptable = 0.07
            max_acceptable = 0.13

            assert min_acceptable <= actual_rate <= max_acceptable, \
                f"Exploration rate {actual_rate:.1%} outside acceptable range " \
                f"[{min_acceptable:.1%}, {max_acceptable:.1%}] for epsilon=10%"

    def test_exploration_disabled_with_zero_epsilon(self):
        """With epsilon=0, no exploration should occur."""
        from elelem._benchmark_store import reorder_candidates_by_benchmark, get_routing_stats, reset_routing_stats

        with patch.dict(os.environ, {"ELELEM_EXPLORATION_EPSILON": "0", "ELELEM_EXPLORATION_EPSILON_MAX": "0"}):
            # Reset counters
            reset_routing_stats()

            candidates = [
                {"original_model_ref": "model_a", "model_id": "a", "provider": "p1", "cost": {"output_cost_per_1m": 1.0}},
                {"original_model_ref": "model_b", "model_id": "b", "provider": "p2", "cost": {"output_cost_per_1m": 1.0}},
            ]

            now = datetime.utcnow()
            dynamic_stats = {
                "model_a": DynamicStats(model_ref="model_a", avg_tokens_per_sec=100.0, sample_count=10, last_updated=now),
                "model_b": DynamicStats(model_ref="model_b", avg_tokens_per_sec=90.0, sample_count=10, last_updated=now),
            }

            for _ in range(100):
                reorder_candidates_by_benchmark([c.copy() for c in candidates], dynamic_stats=dynamic_stats)

            stats = get_routing_stats()

            assert stats["explorations"] == 0, \
                f"Expected 0 explorations with epsilon=0, got {stats['explorations']}"
            assert stats["total"] == 100, \
                f"Expected 100 total routing decisions, got {stats['total']}"
