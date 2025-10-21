#!/usr/bin/env python3
"""
Integration tests for Elelem server API using pytest fixtures.
Tests the real deployment scenario: API calls -> Elelem server -> PostgreSQL metrics
"""

import pytest
import time
import requests
import json
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.server
class TestElelemAPIIntegration:
    """Test Elelem API server integration with Docker services."""

    def test_health_endpoint(self, elelem_server_url):
        """Test that the health endpoint is accessible."""
        response = requests.get(f"{elelem_server_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_list_models(self, elelem_server_url):
        """Test the /v1/models endpoint."""
        response = requests.get(f"{elelem_server_url}/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

    def test_chat_completion_basic(self, api_client):
        """Test basic chat completion through OpenAI client."""
        response = api_client.chat.completions.create(
            model="cerebras@openrouter:openai/gpt-oss-120b?reasoning=low",
            messages=[
                {"role": "user", "content": "Say 'test successful' and nothing else"}
            ]
        )

        assert response.choices[0].message.content is not None
        assert "test" in response.choices[0].message.content.lower() or \
               "successful" in response.choices[0].message.content.lower()

    def test_metrics_recording(self, elelem_server_url, api_client):
        """Test that API calls are recorded in metrics."""
        # Get initial metrics
        initial_response = requests.get(f"{elelem_server_url}/v1/metrics/summary")
        initial_stats = initial_response.json() if initial_response.status_code == 200 else {"requests": {"total": 0}}
        initial_count = initial_stats.get("requests", {}).get("total", 0)

        # Make an API call
        api_client.chat.completions.create(
            model="cerebras@openrouter:openai/gpt-oss-120b?reasoning=low",
            messages=[{"role": "user", "content": "test"}]
        )

        # Wait a moment for metrics to be recorded
        time.sleep(0.5)

        # Check metrics were updated
        final_response = requests.get(f"{elelem_server_url}/v1/metrics/summary")
        assert final_response.status_code == 200

        final_stats = final_response.json()
        final_count = final_stats["requests"]["total"]

        # Should have at least one more request
        assert final_count > initial_count

    def test_metrics_endpoints(self, elelem_server_url):
        """Test all metrics endpoints are accessible."""
        endpoints = [
            "/v1/metrics/summary",
            "/v1/metrics/tags",
            "/v1/metrics/data"
        ]

        for endpoint in endpoints:
            response = requests.get(f"{elelem_server_url}{endpoint}")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("application/json")

    def test_dashboard_accessible(self, dashboard_url):
        """Test that the dashboard is accessible."""
        response = requests.get(dashboard_url)
        assert response.status_code == 200
        assert "streamlit" in response.text.lower() or "elelem" in response.text.lower()

    def test_error_handling(self, api_client):
        """Test that errors are handled properly."""
        with pytest.raises(Exception) as exc_info:
            api_client.chat.completions.create(
                model="non-existent-model",
                messages=[{"role": "user", "content": "test"}]
            )

        # Should get an appropriate error
        assert exc_info.value is not None

    @pytest.mark.slow
    def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request(i):
            response = api_client.chat.completions.create(
                model="cerebras@openrouter:openai/gpt-oss-120b?reasoning=low",
                messages=[{"role": "user", "content": f"Say the number {i}"}]            )
            return response.choices[0].message.content is not None

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]
            results = [f.result(timeout=30) for f in futures]

        # All requests should complete successfully
        assert all(results)


@pytest.mark.integration
def test_database_connectivity(elelem_server_url):
    """Test that the database is properly connected."""
    response = requests.get(f"{elelem_server_url}/health")
    assert response.status_code == 200

    data = response.json()
    metrics = data.get("metrics", {})

    # Check PostgreSQL status
    postgres_info = metrics.get("postgresql", {})
    assert postgres_info.get("enabled") is True
    assert postgres_info.get("connected") is True


@pytest.mark.integration
@pytest.mark.server
def test_cache_hit_and_miss(api_client, elelem_server_url):
    """Test cache hit/miss behavior with PostgreSQL backend."""
    import time

    # Make first request - should be cache MISS
    response1 = api_client.chat.completions.create(
        model="cerebras@openrouter:openai/gpt-oss-120b?reasoning=low",
        messages=[
            {"role": "user", "content": "Say exactly: 'cache test response'"}
        ],
        temperature=0  # Deterministic for caching
    )

    # Check cache status in health endpoint
    health = requests.get(f"{elelem_server_url}/health").json()
    assert health.get("cache", {}).get("enabled") is True

    # Make identical request - should be cache HIT
    response2 = api_client.chat.completions.create(
        model="cerebras@openrouter:openai/gpt-oss-120b?reasoning=low",
        messages=[
            {"role": "user", "content": "Say exactly: 'cache test response'"}
        ],
        temperature=0
    )

    # Both responses should have same content
    assert response1.choices[0].message.content == response2.choices[0].message.content

    # Check elelem_metrics for cache indication
    # Note: OpenAI client may not expose elelem_metrics, so we verify via behavior
    # Cache hit should be much faster than cache miss (no actual API call)

    # Make request with different temperature - should be cache MISS (different cache key)
    response3 = api_client.chat.completions.create(
        model="cerebras@openrouter:openai/gpt-oss-120b?reasoning=low",
        messages=[
            {"role": "user", "content": "Say exactly: 'cache test response'"}
        ],
        temperature=0.5  # Different temperature = different cache key
    )

    # Content may differ due to different temperature
    # Just verify it completes successfully
    assert response3.choices[0].message.content is not None


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])