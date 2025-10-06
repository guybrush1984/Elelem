"""
Tests for PostgreSQL response caching.

Tests cache functionality using faker to avoid real API calls.
"""

import pytest
import asyncio
import time
from pathlib import Path
import sys

# Add src and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from elelem import Elelem
from faker.server import ModelFaker


class TestResponseCache:
    """Test response caching with faker."""

    @pytest.fixture(scope="function")
    def faker_server(self):
        """Start a faker server for testing."""
        faker = ModelFaker(port=6666)  # Must match port in faker.yaml
        try:
            faker.start()
            time.sleep(0.5)  # Wait for server to be ready
            yield faker
        finally:
            try:
                faker.stop()
                time.sleep(1.0)  # Give time for socket to be released
            except Exception as e:
                print(f"Warning: Error during faker cleanup: {e}")

    @pytest.fixture(scope="function")
    def elelem_with_cache(self, faker_server, monkeypatch, tmp_path):
        """Create Elelem instance with cache enabled and faker support."""
        # Set environment variable for faker
        monkeypatch.setenv("FAKER_API_KEY", "fake-key-123")

        # Use a unique temporary SQLite database for each test
        db_file = tmp_path / "test_cache.db"
        monkeypatch.setenv("ELELEM_DATABASE_URL", f"sqlite:///{db_file}")

        # Update faker provider configuration to use the actual server port
        from elelem.config import Config
        config = Config()
        faker_provider = config.get_provider_config("faker")
        faker_provider["endpoint"] = f"http://localhost:{faker_server.port}/v1"

        # Configure faker with happy path scenario
        faker_server.configure_scenario('happy_path')
        faker_server.reset_state()

        # Create Elelem instance with cache enabled
        elelem = Elelem(
            extra_provider_dirs=["tests/providers"],
            cache_enabled=True,
            cache_ttl=60,
            cache_max_size=50000
        )

        return elelem, faker_server

    @pytest.mark.asyncio
    async def test_basic_caching_works(self, elelem_with_cache):
        """Test that basic caching works as expected - cache hits reduce costs to zero."""
        elelem, faker = elelem_with_cache

        # First request - should be a cache miss
        response1 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Hello'}],
            temperature=0,
            tags=['test:cache_basic']
        )

        # Check that first response was NOT cached
        assert response1.elelem_metrics.get('cached') is not True
        cost1 = response1.elelem_metrics['costs_usd']['total_cost_usd']
        assert cost1 > 0  # Should have cost

        # Second identical request - should be a cache hit
        response2 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Hello'}],
            temperature=0,
            tags=['test:cache_basic']
        )

        # Check that second response WAS cached
        assert response2.elelem_metrics.get('cached') is True
        assert response2.elelem_metrics['costs_usd']['total_cost_usd'] == 0.0  # Cached = free
        assert response2.elelem_metrics.get('cache_age_seconds') is not None
        assert response2.elelem_metrics.get('cache_age_seconds') >= 0

        # Content should be identical
        assert response1.choices[0].message.content == response2.choices[0].message.content

        print("✓ Basic caching works - cache hits have zero cost")

    @pytest.mark.asyncio
    async def test_caching_can_be_disabled(self, elelem_with_cache):
        """Test that caching can be disabled with cache=False parameter."""
        elelem, faker = elelem_with_cache

        # First request - populate cache
        response1 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Disable test'}],
            temperature=0,
            tags=['test:cache_disable']
        )

        assert response1.elelem_metrics.get('cached') is not True
        cost1 = response1.elelem_metrics['costs_usd']['total_cost_usd']
        assert cost1 > 0

        # Second identical request with cache=False - should bypass cache
        response2 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Disable test'}],
            temperature=0,
            cache=False,  # Explicit bypass
            tags=['test:cache_disable']
        )

        # Should NOT be cached (bypassed)
        assert response2.elelem_metrics.get('cached') is not True
        cost2 = response2.elelem_metrics['costs_usd']['total_cost_usd']
        assert cost2 > 0  # Should have cost (not cached)

        print("✓ Cache can be disabled with cache=False")

    @pytest.mark.asyncio
    async def test_cache_deletion_works(self, elelem_with_cache):
        """Test that cache deletion/cleanup works properly."""
        elelem, faker = elelem_with_cache

        # Create some cached entries
        await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Entry 1'}],
            temperature=0,
            tags=['test:cleanup']
        )

        await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Entry 2'}],
            temperature=0,
            tags=['test:cleanup']
        )

        # Check cache has entries
        stats_before = elelem.cache.get_stats()
        assert stats_before['total_entries'] >= 2

        # Note: We can't easily test expired entry deletion without mocking time,
        # but we can test the cleanup function runs without error
        deleted = elelem.cache.cleanup_expired()
        assert deleted >= 0  # Should return count (0 or more, 0 is fine if nothing expired)

        # Verify cache still works after cleanup
        response = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Entry 1'}],
            temperature=0,
            tags=['test:cleanup']
        )

        # Should be cached (entry 1 still exists)
        assert response.elelem_metrics.get('cached') is True

        print("✓ Cache cleanup runs successfully")

    @pytest.mark.asyncio
    async def test_cache_respects_temperature(self, elelem_with_cache):
        """Test that different temperatures create different cache entries."""
        elelem, faker = elelem_with_cache

        # Request with temp=0
        response1 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Temperature test'}],
            temperature=0,
            tags=['test:temp']
        )

        assert response1.elelem_metrics.get('cached') is not True

        # Same request but temp=0.5 - should be cache miss (different key)
        response2 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Temperature test'}],
            temperature=0.5,
            tags=['test:temp']
        )

        assert response2.elelem_metrics.get('cached') is not True
        assert response2.elelem_metrics['costs_usd']['total_cost_usd'] > 0

        # Repeat temp=0 request - should be cache hit
        response3 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Temperature test'}],
            temperature=0,
            tags=['test:temp']
        )

        assert response3.elelem_metrics.get('cached') is True

        print("✓ Cache respects temperature parameter")

    @pytest.mark.asyncio
    async def test_cache_stats_tracking(self, elelem_with_cache):
        """Test cache statistics are properly tracked."""
        elelem, faker = elelem_with_cache

        # Make some requests
        await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Stats test 1'}],
            temperature=0,
            tags=['test:stats']
        )

        await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Stats test 2'}],
            temperature=0,
            tags=['test:stats']
        )

        # Hit cache for first one
        await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Stats test 1'}],
            temperature=0,
            tags=['test:stats']
        )

        stats = elelem.cache.get_stats()

        assert stats['total_entries'] >= 2
        assert stats['valid_entries'] >= 2
        assert stats['total_hits'] >= 1  # At least one cache hit
        assert stats['total_size_bytes'] > 0

        print(f"✓ Cache stats: {stats['total_entries']} entries, {stats['total_hits']} hits")

    @pytest.mark.asyncio
    async def test_cache_respects_size_limit(self, elelem_with_cache):
        """Test that cache respects max size limits."""
        elelem, faker = elelem_with_cache

        # Override cache with tiny size limit
        elelem.cache.max_size = 100  # Very small

        # Make request - response will likely be too large to cache
        response1 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'This response will be too large for tiny cache'}],
            temperature=0,
            tags=['test:size']
        )

        # Second identical request should NOT be cached (response too large)
        response2 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'This response will be too large for tiny cache'}],
            temperature=0,
            tags=['test:size']
        )

        # Both should NOT be cached (too large)
        assert response1.elelem_metrics.get('cached') is not True
        assert response2.elelem_metrics.get('cached') is not True

        print("✓ Cache respects size limits")

    def test_cache_key_generation(self, elelem_with_cache):
        """Test cache key generation logic."""
        elelem, faker = elelem_with_cache

        # Same parameters should give same key
        key1 = elelem.cache.get_cache_key(
            'faker:basic',
            [{'role': 'user', 'content': 'Test'}],
            temperature=0
        )

        key2 = elelem.cache.get_cache_key(
            'faker:basic',
            [{'role': 'user', 'content': 'Test'}],
            temperature=0
        )

        assert key1 == key2
        assert key1 is not None
        assert key1.startswith('elelem_v1_')

        # Different temperature should give different key
        key3 = elelem.cache.get_cache_key(
            'faker:basic',
            [{'role': 'user', 'content': 'Test'}],
            temperature=0.5
        )

        assert key3 != key1

        # cache=False should return None
        key4 = elelem.cache.get_cache_key(
            'faker:basic',
            [{'role': 'user', 'content': 'Test'}],
            temperature=0,
            cache=False
        )

        assert key4 is None

        print("✓ Cache key generation works correctly")

    @pytest.mark.asyncio
    async def test_cache_expiration_and_cleanup(self, elelem_with_cache):
        """Test that expired cache entries are properly cleaned up."""
        elelem, faker = elelem_with_cache

        # Override cache with very short TTL (1 second) BEFORE creating entries
        elelem.cache.ttl = 1

        # Create a cached entry (will use TTL=1) - use unique content to avoid conflicts
        response1 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Unique expiration test for cleanup'}],
            temperature=0,
            tags=['test:expiration']
        )

        assert response1.elelem_metrics.get('cached') is not True

        # Immediately request again - should be cached
        response2 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Unique expiration test for cleanup'}],
            temperature=0,
            tags=['test:expiration']
        )

        assert response2.elelem_metrics.get('cached') is True

        # Wait for cache to expire (TTL=1s + extra buffer for safety)
        import time
        time.sleep(2.0)  # Increased to 2s to ensure expiration

        # Manually trigger cleanup (simulates background task)
        deleted = elelem.cache.cleanup_expired()
        assert deleted >= 1  # Should delete at least our expired entry

        # Request again - should be cache miss (expired entry was deleted)
        response3 = await elelem.create_chat_completion(
            model='faker:basic',
            messages=[{'role': 'user', 'content': 'Unique expiration test for cleanup'}],
            temperature=0,
            tags=['test:expiration']
        )

        assert response3.elelem_metrics.get('cached') is not True

        print("✓ Cache expiration and cleanup works correctly")
