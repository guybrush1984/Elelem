"""
Tests for the benchmark store and candidate reordering.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Import after setting up mocks if needed
from elelem._benchmark_store import (
    BenchmarkStore,
    get_benchmark_store,
    reorder_candidates_by_benchmark
)


class TestBenchmarkStore:
    """Tests for BenchmarkStore class."""

    def test_disabled_when_no_source(self):
        """Store should be disabled when no source is configured."""
        with patch.dict(os.environ, {}, clear=True):
            store = BenchmarkStore()
            assert not store.enabled
            assert store.source is None

    def test_enabled_when_source_configured(self):
        """Store should be enabled when source is configured."""
        with patch.dict(os.environ, {'ELELEM_BENCHMARK_SOURCE': '/path/to/file.json'}):
            store = BenchmarkStore()
            assert store.enabled
            assert store.source == '/path/to/file.json'

    def test_get_benchmark_returns_none_for_unknown_model(self):
        """Should return None for unknown models."""
        store = BenchmarkStore()
        store._data = {
            "fireworks:deepseek/deepseek-3.2": {"tokens_per_second": 100.0}
        }
        assert store.get_benchmark("unknown:model") is None

    def test_get_benchmark_returns_data_for_known_model(self):
        """Should return data for known models."""
        store = BenchmarkStore()
        benchmark_data = {"tokens_per_second": 100.0, "cost_per_1m_output": 1.0}
        store._data = {"test:model": benchmark_data}

        result = store.get_benchmark("test:model")
        assert result == benchmark_data

    def test_calculate_value_score_basic(self):
        """Value score should be speed / cost with default weight."""
        store = BenchmarkStore()
        store._data = {
            "test:model": {
                "tokens_per_second": 100.0,
                "cost_per_1m_output": 2.0
            }
        }

        score = store.calculate_value_score("test:model")
        assert score == 50.0  # 100 / 2

    def test_calculate_value_score_with_speed_weight(self):
        """Higher speed_weight should favor faster models."""
        store = BenchmarkStore()
        store._data = {
            "test:model": {
                "tokens_per_second": 100.0,
                "cost_per_1m_output": 2.0
            }
        }

        # speed_weight=2 means speed^2 / cost
        score = store.calculate_value_score("test:model", speed_weight=2.0)
        assert score == 5000.0  # 100^2 / 2

    def test_calculate_value_score_filters_slow_models(self):
        """Should return None for models below min_tokens_per_sec."""
        store = BenchmarkStore()
        store._data = {
            "test:model": {
                "tokens_per_second": 10.0,
                "cost_per_1m_output": 1.0
            }
        }

        # Model has 10 t/s, filter requires 20
        score = store.calculate_value_score("test:model", min_tokens_per_sec=20.0)
        assert score is None

    def test_calculate_value_score_returns_none_for_missing_model(self):
        """Should return None for models not in benchmark data."""
        store = BenchmarkStore()
        store._data = {}

        score = store.calculate_value_score("unknown:model")
        assert score is None

    def test_parse_telelem_format(self):
        """Should correctly parse telelem batch_summary.json format."""
        store = BenchmarkStore()

        raw_data = {
            "models": {
                "fireworks:deepseek/deepseek-3.2": {
                    "tokens": {
                        "output": {"avg": 338.0}
                    },
                    "duration": {"avg": 4.38},
                    "costs": {"avg": 0.000624},
                    "requests": {"success_rate": 1.0, "total": 1}
                }
            }
        }

        result = store._parse_telelem_format(raw_data)

        assert "fireworks:deepseek/deepseek-3.2" in result
        benchmark = result["fireworks:deepseek/deepseek-3.2"]

        # tokens_per_second = 338 / 4.38 ≈ 77.17
        assert 77 < benchmark["tokens_per_second"] < 78

        # cost_per_1m = (0.000624 / 338) * 1_000_000 ≈ 1.846
        # (cost is already in dollars, extrapolated to 1M tokens)
        assert 1.8 < benchmark["cost_per_1m_output"] < 1.9

        assert benchmark["success_rate"] == 1.0
        assert benchmark["sample_count"] == 1

    def test_get_status(self):
        """Should return correct status information."""
        with patch.dict(os.environ, {'ELELEM_BENCHMARK_SOURCE': 'https://example.com/benchmarks.json'}):
            store = BenchmarkStore()
            store._data = {"model1": {}, "model2": {}}

            status = store.get_status()

            assert status["enabled"] is True
            assert status["source"] == "https://example.com/benchmarks.json"
            assert status["entries_count"] == 2
            assert status["interval_seconds"] == 3600  # default


class TestReorderCandidates:
    """Tests for reorder_candidates_by_benchmark function."""

    def test_returns_original_order_when_disabled(self):
        """Should return original order when benchmark store is disabled."""
        candidates = [
            {"original_model_ref": "slow:model", "provider": "slow"},
            {"original_model_ref": "fast:model", "provider": "fast"},
        ]

        with patch('elelem._benchmark_store.get_benchmark_store') as mock_store:
            mock_store.return_value.enabled = False
            result = reorder_candidates_by_benchmark(candidates)

        assert result == candidates

    def test_reorders_by_value_score(self):
        """Should reorder candidates by value score (highest first)."""
        candidates = [
            {"original_model_ref": "slow:model", "provider": "slow"},
            {"original_model_ref": "fast:model", "provider": "fast"},
            {"original_model_ref": "medium:model", "provider": "medium"},
        ]

        # Create a mock store with benchmark data
        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "slow:model": {"tokens_per_second": 10.0},
            "fast:model": {"tokens_per_second": 100.0},
            "medium:model": {"tokens_per_second": 50.0},
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "slow:model": 10.0,
            "fast:model": 100.0,
            "medium:model": 50.0,
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        # Should be sorted: fast (100), medium (50), slow (10)
        assert result[0]["original_model_ref"] == "fast:model"
        assert result[1]["original_model_ref"] == "medium:model"
        assert result[2]["original_model_ref"] == "slow:model"

    def test_unscored_candidates_placed_after_scored(self):
        """Candidates without benchmark data should come after scored ones."""
        candidates = [
            {"original_model_ref": "unknown:model", "provider": "unknown"},
            {"original_model_ref": "known:model", "provider": "known"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "known:model": {"tokens_per_second": 50.0},
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "known:model": 50.0,
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        # Known model first (has score), unknown model second (no score)
        assert result[0]["original_model_ref"] == "known:model"
        assert result[1]["original_model_ref"] == "unknown:model"

    def test_filters_slow_candidates(self):
        """Should filter out candidates below min_tokens_per_sec."""
        candidates = [
            {"original_model_ref": "slow:model", "provider": "slow"},
            {"original_model_ref": "fast:model", "provider": "fast"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "slow:model": {"tokens_per_second": 5.0, "cost_per_1m_output": 1.0},
            "fast:model": {"tokens_per_second": 100.0, "cost_per_1m_output": 1.0},
        }.get(ref)

        # calculate_value_score returns None for slow model when min_tokens_per_sec=20
        def mock_calculate(ref, speed_weight=1.0, min_tokens_per_sec=0.0):
            speeds = {"slow:model": 5.0, "fast:model": 100.0}
            scores = {"slow:model": 5.0, "fast:model": 100.0}
            if min_tokens_per_sec > 0 and speeds.get(ref, 0) < min_tokens_per_sec:
                return None
            return scores.get(ref)

        mock_store.calculate_value_score.side_effect = mock_calculate

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(
                candidates,
                min_tokens_per_sec=20.0
            )

        # Only fast model should remain (slow was filtered)
        assert len(result) == 1
        assert result[0]["original_model_ref"] == "fast:model"

    def test_fallback_to_yaml_order_when_all_filtered(self):
        """Should fallback to YAML order if ALL candidates are filtered out."""
        candidates = [
            {"original_model_ref": "slow1:model", "provider": "slow1"},
            {"original_model_ref": "slow2:model", "provider": "slow2"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "slow1:model": {"tokens_per_second": 5.0, "cost_per_1m_output": 1.0},
            "slow2:model": {"tokens_per_second": 10.0, "cost_per_1m_output": 1.0},
        }.get(ref)

        # All models are below 100 t/s threshold
        def mock_calculate(ref, speed_weight=1.0, min_tokens_per_sec=0.0):
            speeds = {"slow1:model": 5.0, "slow2:model": 10.0}
            if min_tokens_per_sec > 0 and speeds.get(ref, 0) < min_tokens_per_sec:
                return None
            return speeds.get(ref)

        mock_store.calculate_value_score.side_effect = mock_calculate

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(
                candidates,
                min_tokens_per_sec=100.0  # All candidates below this
            )

        # Should fallback to original YAML order (not empty!)
        assert len(result) == 2
        assert result[0]["original_model_ref"] == "slow1:model"
        assert result[1]["original_model_ref"] == "slow2:model"

    def test_empty_candidates_returns_empty(self):
        """Should handle empty candidate list."""
        mock_store = MagicMock()
        mock_store.enabled = True

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark([])

        assert result == []

    def test_priority_always_first_overrides_score(self):
        """Candidates with priority: always_first should be placed first regardless of score."""
        candidates = [
            {"original_model_ref": "fast:model", "provider": "fast"},
            {"original_model_ref": "slow:model", "provider": "slow", "priority": "always_first"},
            {"original_model_ref": "medium:model", "provider": "medium"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "fast:model": {"tokens_per_second": 100.0},
            "slow:model": {"tokens_per_second": 10.0},
            "medium:model": {"tokens_per_second": 50.0},
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "fast:model": 100.0,
            "slow:model": 10.0,
            "medium:model": 50.0,
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        # slow:model should be FIRST despite lowest score (has always_first)
        # Then fast (100), then medium (50)
        assert result[0]["original_model_ref"] == "slow:model"
        assert result[1]["original_model_ref"] == "fast:model"
        assert result[2]["original_model_ref"] == "medium:model"

    def test_multiple_always_first_preserves_yaml_order(self):
        """Multiple always_first candidates should preserve their YAML order."""
        candidates = [
            {"original_model_ref": "first:model", "provider": "first", "priority": "always_first"},
            {"original_model_ref": "second:model", "provider": "second", "priority": "always_first"},
            {"original_model_ref": "fast:model", "provider": "fast"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "first:model": {"tokens_per_second": 10.0},
            "second:model": {"tokens_per_second": 20.0},
            "fast:model": {"tokens_per_second": 100.0},
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "first:model": 10.0,
            "second:model": 20.0,
            "fast:model": 100.0,
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        # Both always_first candidates should come before fast:model
        # And maintain their YAML order (first, second)
        assert result[0]["original_model_ref"] == "first:model"
        assert result[1]["original_model_ref"] == "second:model"
        assert result[2]["original_model_ref"] == "fast:model"

    def test_mixed_scored_unscored_and_priority(self):
        """Test ordering with mix of scored, unscored, and priority candidates."""
        candidates = [
            {"original_model_ref": "unscored1:model", "provider": "unscored1"},
            {"original_model_ref": "scored_low:model", "provider": "scored_low"},
            {"original_model_ref": "priority:model", "provider": "priority", "priority": "always_first"},
            {"original_model_ref": "scored_high:model", "provider": "scored_high"},
            {"original_model_ref": "unscored2:model", "provider": "unscored2"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        # Only some models have benchmark data
        mock_store.get_benchmark.side_effect = lambda ref: {
            "scored_low:model": {"tokens_per_second": 10.0},
            "scored_high:model": {"tokens_per_second": 100.0},
            "priority:model": {"tokens_per_second": 5.0},  # Has benchmark but priority overrides
            # unscored1 and unscored2 have NO benchmark data
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "scored_low:model": 10.0,
            "scored_high:model": 100.0,
            "priority:model": 5.0,
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        refs = [c["original_model_ref"] for c in result]

        # Expected order:
        # 1. priority:model (always_first)
        # 2. scored_high:model (highest score: 100)
        # 3. scored_low:model (lower score: 10)
        # 4. unscored1:model (no data, YAML order preserved)
        # 5. unscored2:model (no data, YAML order preserved)
        assert refs[0] == "priority:model"
        assert refs[1] == "scored_high:model"
        assert refs[2] == "scored_low:model"
        # Unscored should be last, in their original YAML order
        assert refs[3] == "unscored1:model"
        assert refs[4] == "unscored2:model"

    def test_priority_always_last_placed_at_end(self):
        """Candidates with priority: always_last should be placed last regardless of score."""
        candidates = [
            {"original_model_ref": "fallback:model", "provider": "fallback", "priority": "always_last"},
            {"original_model_ref": "fast:model", "provider": "fast"},
            {"original_model_ref": "slow:model", "provider": "slow"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "fast:model": {"tokens_per_second": 100.0},
            "slow:model": {"tokens_per_second": 10.0},
            "fallback:model": {"tokens_per_second": 1000.0},  # Highest score but should be last
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "fast:model": 100.0,
            "slow:model": 10.0,
            "fallback:model": 1000.0,  # Best score but always_last
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        refs = [c["original_model_ref"] for c in result]

        # fallback:model should be LAST despite highest score (has always_last)
        assert refs[0] == "fast:model"
        assert refs[1] == "slow:model"
        assert refs[2] == "fallback:model"

    def test_multiple_always_last_preserves_yaml_order(self):
        """Multiple always_last candidates should preserve their YAML order among themselves."""
        candidates = [
            {"original_model_ref": "fast:model", "provider": "fast"},
            {"original_model_ref": "fallback1:model", "provider": "fallback1", "priority": "always_last"},
            {"original_model_ref": "fallback2:model", "provider": "fallback2", "priority": "always_last"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "fast:model": {"tokens_per_second": 100.0},
            "fallback1:model": {"tokens_per_second": 50.0},
            "fallback2:model": {"tokens_per_second": 200.0},  # Better than fallback1 but still last
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "fast:model": 100.0,
            "fallback1:model": 50.0,
            "fallback2:model": 200.0,
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        refs = [c["original_model_ref"] for c in result]

        # fast:model first, then both always_last in YAML order
        assert refs[0] == "fast:model"
        assert refs[1] == "fallback1:model"
        assert refs[2] == "fallback2:model"

    def test_always_first_and_always_last_together(self):
        """Test ordering with both always_first and always_last candidates."""
        candidates = [
            {"original_model_ref": "fallback:model", "provider": "fallback", "priority": "always_last"},
            {"original_model_ref": "priority:model", "provider": "priority", "priority": "always_first"},
            {"original_model_ref": "scored:model", "provider": "scored"},
            {"original_model_ref": "unscored:model", "provider": "unscored"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        mock_store.get_benchmark.side_effect = lambda ref: {
            "scored:model": {"tokens_per_second": 50.0},
            "priority:model": {"tokens_per_second": 10.0},
            "fallback:model": {"tokens_per_second": 100.0},
            # unscored:model has no benchmark data
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "scored:model": 50.0,
            "priority:model": 10.0,
            "fallback:model": 100.0,
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        refs = [c["original_model_ref"] for c in result]

        # Expected order:
        # 1. priority:model (always_first)
        # 2. scored:model (scored, middle)
        # 3. unscored:model (unscored, after scored)
        # 4. fallback:model (always_last)
        assert refs[0] == "priority:model"
        assert refs[1] == "scored:model"
        assert refs[2] == "unscored:model"
        assert refs[3] == "fallback:model"

    def test_all_candidates_missing_benchmark_preserves_yaml_order(self):
        """When no candidates have benchmark data, YAML order is preserved."""
        candidates = [
            {"original_model_ref": "model_a:test", "provider": "a"},
            {"original_model_ref": "model_b:test", "provider": "b"},
            {"original_model_ref": "model_c:test", "provider": "c"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True
        # None of the models have benchmark data
        mock_store.get_benchmark.return_value = None
        mock_store.calculate_value_score.return_value = None

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        # Order should be unchanged (YAML order)
        refs = [c["original_model_ref"] for c in result]
        assert refs == ["model_a:test", "model_b:test", "model_c:test"]

    def test_partial_benchmark_data_handles_gracefully(self):
        """Candidates with partial benchmark data (missing fields) should be treated as unscored."""
        candidates = [
            {"original_model_ref": "complete:model", "provider": "complete"},
            {"original_model_ref": "partial:model", "provider": "partial"},
            {"original_model_ref": "missing:model", "provider": "missing"},
        ]

        mock_store = MagicMock()
        mock_store.enabled = True

        # complete:model has full data
        # partial:model has data but calculate_value_score returns None (e.g., zero cost)
        # missing:model has no data at all
        mock_store.get_benchmark.side_effect = lambda ref: {
            "complete:model": {"tokens_per_second": 100.0, "cost_per_1m_output": 1.0},
            "partial:model": {"tokens_per_second": 50.0, "cost_per_1m_output": 0},  # Invalid cost
        }.get(ref)
        mock_store.calculate_value_score.side_effect = lambda ref, speed_weight=1.0, min_tokens_per_sec=0.0: {
            "complete:model": 100.0,
            # partial:model returns None because cost is 0
        }.get(ref)

        with patch('elelem._benchmark_store.get_benchmark_store', return_value=mock_store):
            result = reorder_candidates_by_benchmark(candidates)

        refs = [c["original_model_ref"] for c in result]

        # complete:model should be first (has valid score)
        # partial:model and missing:model should follow in YAML order (both unscored)
        assert refs[0] == "complete:model"
        assert refs[1] == "partial:model"
        assert refs[2] == "missing:model"


class TestBenchmarkStoreFileLoading:
    """Tests for file loading functionality."""

    def test_load_local_file(self, tmp_path):
        """Should load benchmark data from a local file."""
        # Create a test file
        test_data = {
            "models": {
                "test:model": {
                    "tokens": {"output": {"avg": 100.0}},
                    "duration": {"avg": 1.0},
                    "costs": {"avg": 0.001},
                    "requests": {"success_rate": 1.0, "total": 10}
                }
            }
        }
        test_file = tmp_path / "benchmarks.json"
        test_file.write_text(json.dumps(test_data))

        store = BenchmarkStore()
        result = store._load_file(str(test_file))

        assert result == test_data

    def test_load_nonexistent_file_returns_none(self, tmp_path):
        """Should return None for nonexistent files."""
        store = BenchmarkStore()
        result = store._load_file(str(tmp_path / "nonexistent.json"))

        assert result is None
        assert store._fetch_error is not None


@pytest.mark.asyncio
class TestBenchmarkStoreAsync:
    """Async tests for BenchmarkStore."""

    async def test_fetch_once_from_file(self, tmp_path):
        """Should fetch and parse data from a local file."""
        test_data = {
            "models": {
                "provider:model": {
                    "tokens": {"output": {"avg": 200.0}},
                    "duration": {"avg": 2.0},
                    "costs": {"avg": 0.002},
                    "requests": {"success_rate": 0.95, "total": 100}
                }
            }
        }
        test_file = tmp_path / "benchmarks.json"
        test_file.write_text(json.dumps(test_data))

        with patch.dict(os.environ, {'ELELEM_BENCHMARK_SOURCE': str(test_file)}):
            store = BenchmarkStore()
            result = await store.fetch_once()

        assert result is True
        assert store.get_benchmark("provider:model") is not None
        assert store.get_benchmark("provider:model")["tokens_per_second"] == 100.0  # 200/2

    async def test_fetch_once_handles_invalid_json(self, tmp_path):
        """Should handle invalid JSON gracefully."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json")

        with patch.dict(os.environ, {'ELELEM_BENCHMARK_SOURCE': str(test_file)}):
            store = BenchmarkStore()
            result = await store.fetch_once()

        assert result is False
        assert store._fetch_error is not None
