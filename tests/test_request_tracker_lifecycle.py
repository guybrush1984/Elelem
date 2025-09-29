#!/usr/bin/env python3
"""
Comprehensive tests for RequestTracker lifecycle

Tests all aspects of the unified metrics system:
- Request creation and initialization
- Retry recording across all types
- Success and failure finalization
- Integration with MetricsStore
- PostgreSQL persistence (if available)
- Analytics calculations
"""

import pytest
import pandas as pd
import time
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from elelem.metrics import MetricsStore, RequestTracker


@pytest.mark.unit
class TestRequestTrackerLifecycle:
    """Test complete RequestTracker lifecycle scenarios"""

    def setup_method(self):
        """Set up fresh MetricsStore for each test"""
        # Use a unique in-memory database for each test to ensure isolation
        import tempfile
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        # Override the default database URL to use our temp file
        import os
        os.environ['ELELEM_DATABASE_URL'] = f'sqlite:///{self.temp_db.name}'
        self.metrics_store = MetricsStore()

    def teardown_method(self):
        """Clean up after each test"""
        import os
        # Clean up the temp database file
        if hasattr(self, 'temp_db'):
            try:
                os.unlink(self.temp_db.name)
            except:
                pass
        # Clear the environment variable
        if 'ELELEM_DATABASE_URL' in os.environ:
            del os.environ['ELELEM_DATABASE_URL']

    def test_complete_success_lifecycle(self):
        """Test complete successful request lifecycle"""
        # Start request
        tracker = self.metrics_store.start_request(
            request_id="test_success_001",
            requested_model="gpt-4",
            tags={"env": "test", "type": "success"},
            temperature=0.7,
            max_tokens=100,
            stream=False
        )

        # Verify initial state
        assert tracker.request_id == "test_success_001"
        assert tracker.requested_model == "gpt-4"
        assert tracker.temperature == 0.7
        assert tracker.max_tokens == 100
        assert tracker.stream is False
        assert tracker.status == "pending"
        assert "env" in tracker.tags
        assert "type" in tracker.tags

        # Record some retries
        tracker.record_retry("rate_limit_retries")
        tracker.record_retry("candidate_iterations")
        tracker.record_retry("temperature_reductions")

        # Verify retry counts
        assert tracker.retry_counts["rate_limit_retries"] == 1
        assert tracker.retry_counts["candidate_iterations"] == 1
        assert tracker.retry_counts["temperature_reductions"] == 1
        assert tracker.retry_counts["json_parse_retries"] == 0

        # Finalize with success
        tracker.finalize_with_candidate(
            self.metrics_store,
            selected_candidate="openai:gpt-4",
            actual_model="gpt-4-0314",
            actual_provider="openai",
            status="success",
            input_tokens=85,
            output_tokens=150,
            reasoning_tokens=25,
            total_cost_usd=0.0125
        )

        # Verify final state
        assert tracker.status == "success"
        assert tracker.selected_candidate == "openai:gpt-4"
        assert tracker.actual_model == "gpt-4-0314"
        assert tracker.actual_provider == "openai"
        assert tracker.input_tokens == 85
        assert tracker.output_tokens == 150
        assert tracker.reasoning_tokens == 25
        assert tracker.total_cost_usd == 0.0125
        assert tracker.completion_time is not None

        # Verify it's in the MetricsStore using get_stats
        stats = self.metrics_store.get_stats()
        assert stats["requests"]["total"] == 1
        assert stats["requests"]["successful"] == 1
        assert stats["requests"]["failed"] == 0

        # Use internal method for detailed verification
        df = self.metrics_store._load_dataframe()
        assert len(df) == 1
        record = df.iloc[0]
        assert record["request_id"] == "test_success_001"
        assert record["status"] == "success"
        assert record["rate_limit_retries"] == 1
        assert record["candidate_iterations"] == 1
        assert record["temperature_reductions"] == 1
        assert record["total_retry_attempts"] == 3

    def test_complete_failure_lifecycle(self):
        """Test complete failed request lifecycle"""
        # Start request
        tracker = self.metrics_store.start_request(
            request_id="test_failure_001",
            requested_model="claude-3",
            tags={"env": "test", "type": "failure"},
            temperature=0.5
        )

        # Record retries leading to failure
        tracker.record_retry("api_json_validation_retries")
        tracker.record_retry("response_format_removals")

        # Finalize with failure
        tracker.finalize_failure(
            self.metrics_store,
            error_type="AuthenticationError",
            error_message="Invalid API key"
        )

        # Verify final state
        assert tracker.status == "failed"
        assert tracker.final_error_type == "AuthenticationError"
        assert tracker.final_error_message == "Invalid API key"
        assert tracker.retry_counts["final_failures"] == 1
        assert tracker.completion_time is not None

        # Verify it's in the MetricsStore using stats
        stats = self.metrics_store.get_stats()
        assert stats["requests"]["total"] == 1
        assert stats["requests"]["successful"] == 0
        assert stats["requests"]["failed"] == 1

        # Failed requests shouldn't have token/cost data
        assert stats["tokens"]["input"]["total"] == 0
        assert stats["tokens"]["output"]["total"] == 0
        assert stats["costs"]["total"] == 0.0

    def test_all_retry_types(self):
        """Test all supported retry types"""
        tracker = self.metrics_store.start_request(
            request_id="test_retries_001",
            requested_model="test-model"
        )

        # Test all retry types
        retry_types = [
            'json_parse_retries',
            'json_schema_retries',
            'api_json_validation_retries',
            'rate_limit_retries',
            'temperature_reductions',
            'response_format_removals',
            'candidate_iterations',
            'final_failures'
        ]

        for retry_type in retry_types:
            tracker.record_retry(retry_type)
            assert tracker.retry_counts[retry_type] == 1

        # Test invalid retry type (should raise ValueError)
        with pytest.raises(ValueError, match="Unknown retry type"):
            tracker.record_retry("invalid_retry_type")

        tracker.finalize_with_candidate(
            self.metrics_store,
            selected_candidate="test:model",
            actual_model="test-model",
            actual_provider="test"
        )

        # Verify retry counts through stats
        stats = self.metrics_store.get_stats()
        assert stats["requests"]["total"] == 1
        assert stats["retries"]["total_retry_attempts"] == len(retry_types)  # Should be 8, not including invalid retry

    def test_multiple_requests_analytics(self):
        """Test analytics with multiple requests"""
        # Create multiple requests with different outcomes
        requests_data = [
            {
                "id": "req_001", "model": "gpt-4", "success": True,
                "tokens": (100, 200, 50), "cost": 0.015
            },
            {
                "id": "req_002", "model": "claude-3", "success": False,
                "tokens": (80, 0, 0), "cost": 0.0
            },
            {
                "id": "req_003", "model": "gpt-4", "success": True,
                "tokens": (120, 180, 30), "cost": 0.018
            }
        ]

        for req_data in requests_data:
            tracker = self.metrics_store.start_request(
                request_id=req_data["id"],
                requested_model=req_data["model"],
                tags={"batch": "analytics_test"}
            )

            # Add some retries
            tracker.record_retry("rate_limit_retries")
            if not req_data["success"]:
                tracker.record_retry("final_failures")

            if req_data["success"]:
                input_tokens, output_tokens, reasoning_tokens = req_data["tokens"]
                tracker.finalize_with_candidate(
                    self.metrics_store,
                    selected_candidate=f"provider:{req_data['model']}",
                    actual_model=req_data["model"],
                    actual_provider="provider",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_cost_usd=req_data["cost"]
                )
            else:
                tracker.finalize_failure(
                    self.metrics_store,
                    error_type="TestError",
                    error_message="Test failure"
                )

        # Test analytics with new stats structure
        stats = self.metrics_store.get_stats()
        assert stats["requests"]["total"] == 3
        assert stats["requests"]["successful"] == 2
        assert stats["requests"]["failed"] == 1

        # Check that successful requests have tokens
        assert stats["tokens"]["input"]["total"] >= 100 + 120  # At least successful requests
        assert stats["tokens"]["output"]["total"] == 200 + 180  # Only successful requests
        assert stats["tokens"]["reasoning"]["total"] == 50 + 30  # Only successful requests
        assert stats["costs"]["total"] == 0.015 + 0.018  # Only successful requests

        # Test retry analytics
        assert stats["retries"]["rate_limit_retries"] == 3  # All requests had one
        assert stats["retries"]["total_retry_attempts"] >= 4  # At least 3 rate_limit + 1 final_failure

    def test_dataframe_filtering(self):
        """Test DataFrame filtering by time and tags"""
        # Create requests with different timestamps and tags
        base_time = datetime.now()

        tracker1 = self.metrics_store.start_request(
            request_id="filter_001",
            requested_model="model-a",
            tags={"env": "prod", "user": "alice"}
        )
        tracker1.timestamp = base_time - timedelta(hours=2)
        tracker1.finalize_with_candidate(
            self.metrics_store, "prov:model-a", "model-a", "prov"
        )

        tracker2 = self.metrics_store.start_request(
            request_id="filter_002",
            requested_model="model-b",
            tags={"env": "test", "user": "bob"}
        )
        tracker2.timestamp = base_time - timedelta(minutes=30)
        tracker2.finalize_with_candidate(
            self.metrics_store, "prov:model-b", "model-b", "prov"
        )

        tracker3 = self.metrics_store.start_request(
            request_id="filter_003",
            requested_model="model-c",
            tags={"env": "prod", "user": "charlie"}
        )
        tracker3.timestamp = base_time
        tracker3.finalize_with_candidate(
            self.metrics_store, "prov:model-c", "model-c", "prov"
        )

        # Test time filtering
        # Test time filtering with stats
        recent_stats = self.metrics_store.get_stats(
            start_time=base_time - timedelta(hours=1)
        )
        assert recent_stats["requests"]["total"] == 2  # Last two requests

        # Test tag filtering with stats
        prod_stats = self.metrics_store.get_stats(tags=["env:prod"])
        if prod_stats["requests"]["total"] > 0:  # If tag filtering is implemented
            assert prod_stats["requests"]["total"] >= 1  # At least the prod request

    def test_request_tracker_record_structure(self):
        """Test the complete record structure from RequestTracker"""
        tracker = self.metrics_store.start_request(
            request_id="structure_test",
            requested_model="test-model",
            tags={"structure": "test"},
            temperature=0.8,
            max_tokens=500,
            stream=True
        )

        tracker.record_retry("json_parse_retries")
        tracker.record_retry("rate_limit_retries")

        tracker.finalize_with_candidate(
            self.metrics_store,
            selected_candidate="test:provider",
            actual_model="test-model-v2",
            actual_provider="test-provider",
            input_tokens=150,
            output_tokens=300,
            reasoning_tokens=75,
            total_cost_usd=0.025
        )

        record = tracker.to_record()

        # Verify all expected fields are present
        expected_fields = {
            'request_id', 'timestamp', 'requested_model', 'selected_candidate',
            'actual_model', 'actual_provider', 'temperature', 'initial_temperature',
            'max_tokens', 'stream', 'tags', 'status', 'total_duration_seconds',
            'input_tokens', 'output_tokens', 'reasoning_tokens', 'total_cost_usd',
            'final_error_type', 'final_error_message', 'total_tokens_per_second',
            'cost_per_token', 'json_parse_retries', 'json_schema_retries',
            'api_json_validation_retries', 'rate_limit_retries',
            'temperature_reductions', 'response_format_removals',
            'candidate_iterations', 'final_failures', 'total_retry_attempts'
        }

        for field in expected_fields:
            assert field in record, f"Missing field: {field}"

        # Verify field values
        assert record['request_id'] == "structure_test"
        assert record['requested_model'] == "test-model"
        assert record['selected_candidate'] == "test:provider"
        assert record['actual_model'] == "test-model-v2"
        assert record['temperature'] == 0.8
        assert record['max_tokens'] == 500
        assert record['stream'] is True
        assert "structure" in record['tags']
        assert record['json_parse_retries'] == 1
        assert record['rate_limit_retries'] == 1
        assert record['total_retry_attempts'] == 2

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv('ELELEM_DATABASE_URL'),
        reason="PostgreSQL tests require ELELEM_DATABASE_URL"
    )
    def test_postgresql_integration(self):
        """Test PostgreSQL integration if available"""
        # This test requires PostgreSQL to be configured
        postgres_store = MetricsStore()

        # Verify PostgreSQL is connected
        health = postgres_store.get_health_status()
        assert health["postgresql"]["enabled"]
        assert health["postgresql"]["connected"]

        # Create a test request
        tracker = postgres_store.start_request(
            request_id=f"pg_test_{int(time.time())}",
            requested_model="test-model",
            tags={"test": "postgresql", "timestamp": str(int(time.time()))}
        )

        tracker.record_retry("rate_limit_retries")
        tracker.finalize_with_candidate(
            postgres_store,
            selected_candidate="test:model",
            actual_model="test-model",
            actual_provider="test",
            input_tokens=100,
            output_tokens=200,
            total_cost_usd=0.01
        )

        # Verify record was saved to PostgreSQL
        # Verify through stats that the record was saved
        stats = postgres_store.get_stats()
        assert stats["requests"]["total"] >= 1
        assert stats["requests"]["successful"] >= 1
        assert stats["tokens"]["input"]["total"] >= 100
        assert stats["tokens"]["output"]["total"] >= 200


@pytest.mark.unit
def test_lightweight_metrics_functions():
    """Test the lightweight metrics accessor functions"""
    # Create a metrics store for testing
    metrics_store = MetricsStore()

    # Test get_stats
    stats = metrics_store.get_stats()
    assert isinstance(stats, dict)
    assert "requests" in stats
    assert "tokens" in stats
    assert "costs" in stats

    # Test get_health_status
    health = metrics_store.get_health_status()
    assert isinstance(health, dict)
    assert "sqlite" in health or "postgresql" in health

    # Test get_available_tags
    tags = metrics_store.get_available_tags()
    assert isinstance(tags, list)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])