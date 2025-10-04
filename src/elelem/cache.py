"""
Response caching using PostgreSQL backend.

Provides request/response caching with:
- Temperature-aware caching (each temp gets own cache)
- TTL-based expiration
- Leader-elected cleanup (no race conditions)
- Size limits to prevent bloat
- Cache hit/miss tracking in metrics
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy import text


class PostgresCache:
    """PostgreSQL-based response cache with leader-elected cleanup."""

    # Advisory lock ID for cleanup leader election
    CLEANUP_LOCK_ID = 987654321

    def __init__(self, engine, ttl_seconds=300, max_response_size=50000, logger=None):
        """
        Args:
            engine: SQLAlchemy engine (shared with MetricsStore)
            ttl_seconds: Cache TTL in seconds (default 5 minutes)
            max_response_size: Max response size to cache in bytes (default 50KB)
            logger: Logger instance
        """
        self.engine = engine
        self.ttl = ttl_seconds
        self.max_size = max_response_size
        self.logger = logger or logging.getLogger(__name__)
        self._ensure_schema()

    def _ensure_schema(self):
        """Create cache table and indexes if they don't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            response JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP NOT NULL,
            hit_count INTEGER DEFAULT 0,
            response_size INTEGER DEFAULT 0
        )
        """

        # Index for efficient expiration queries
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_cache_expires_at
        ON response_cache(expires_at)
        """

        # Index for cache statistics
        create_stats_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_cache_created_at
        ON response_cache(created_at)
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.execute(text(create_index_sql))
                conn.execute(text(create_stats_index_sql))
                conn.commit()

            self.logger.info("Response cache schema initialized")
        except Exception as e:
            self.logger.error(f"Failed to create cache schema: {e}")
            raise

    def get_cache_key(self, model: str, messages: list, **kwargs) -> Optional[str]:
        """Generate cache key from request parameters.

        Temperature IS included in cache key, so each temperature value
        gets its own cache. This allows both deterministic (temp=0) and
        non-deterministic (temp>0) requests to be cached.

        Args:
            model: Model identifier
            messages: Message list
            **kwargs: Additional parameters

        Returns:
            Cache key string or None if caching should be bypassed
        """
        # Explicit cache bypass
        if kwargs.get('cache') is False:
            return None

        # Build cache dict from parameters that affect output
        cache_dict = {
            'model': model,
            'messages': messages,
            'temperature': kwargs.get('temperature', 1.0),  # Include temperature
            'max_tokens': kwargs.get('max_tokens'),
            'response_format': kwargs.get('response_format'),
            'json_schema': kwargs.get('json_schema'),
            # Explicitly exclude metadata/format-only params:
            # - tags: metadata for tracking
            # - stream: format preference
            # - user: metadata
            # - cache: control flag
        }

        # Stable JSON serialization
        cache_str = json.dumps(cache_dict, sort_keys=True, default=str)
        hash_hex = hashlib.sha256(cache_str.encode()).hexdigest()

        return f"elelem_v1_{hash_hex}"

    def get(self, cache_key: str) -> Optional[tuple[Dict[str, Any], float]]:
        """Retrieve cached response if valid and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Tuple of (cached response dict, age in seconds) or None
        """
        if not cache_key:
            return None

        try:
            query = """
            SELECT response, created_at
            FROM response_cache
            WHERE cache_key = :cache_key
              AND expires_at > NOW()
            """

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {'cache_key': cache_key}
                ).fetchone()

                if result:
                    # Update hit count (best-effort, ignore failures)
                    try:
                        update_sql = """
                        UPDATE response_cache
                        SET hit_count = hit_count + 1
                        WHERE cache_key = :cache_key
                        """
                        conn.execute(text(update_sql), {'cache_key': cache_key})
                        conn.commit()
                    except:
                        pass  # Hit count is just stats, don't fail on error

                    response_data = result[0]  # JSONB column
                    created_at = result[1]

                    age = (datetime.utcnow() - created_at).total_seconds()
                    self.logger.info(f"Cache HIT (age: {age:.1f}s)")

                    return (response_data, age)

            self.logger.debug("Cache MISS")
            return None

        except Exception as e:
            # Cache failures shouldn't break requests
            self.logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, cache_key: str, model: str, response: Any):
        """Store response in cache with size limits.

        Args:
            cache_key: Cache key
            model: Model identifier
            response: OpenAI response object
        """
        if not cache_key:
            return

        try:
            # Serialize OpenAI response to JSON
            response_data = self._serialize_response(response)
            response_json = json.dumps(response_data)
            response_size = len(response_json)

            # Don't cache responses that are too large
            if response_size > self.max_size:
                self.logger.warning(
                    f"Response too large to cache ({response_size} bytes > {self.max_size} limit)"
                )
                return

            expires_at = datetime.utcnow() + timedelta(seconds=self.ttl)

            insert_sql = """
            INSERT INTO response_cache
                (cache_key, model, response, expires_at, hit_count, response_size)
            VALUES
                (:cache_key, :model, :response, :expires_at, 0, :response_size)
            ON CONFLICT (cache_key) DO UPDATE
                SET response = EXCLUDED.response,
                    expires_at = EXCLUDED.expires_at,
                    created_at = NOW(),
                    hit_count = 0,
                    response_size = EXCLUDED.response_size
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        'cache_key': cache_key,
                        'model': model,
                        'response': response_json,
                        'expires_at': expires_at,
                        'response_size': response_size
                    }
                )
                conn.commit()

            self.logger.debug(f"Cached response ({response_size} bytes)")

        except Exception as e:
            # Cache failures shouldn't break requests
            self.logger.warning(f"Cache set failed: {e}")

    def _serialize_response(self, response) -> Dict:
        """Convert OpenAI response to JSON-serializable dict.

        Args:
            response: OpenAI ChatCompletion object

        Returns:
            JSON-serializable dict
        """
        return {
            'id': response.id,
            'object': 'chat.completion',
            'created': getattr(response, 'created', int(datetime.utcnow().timestamp())),
            'model': response.model,
            'choices': [{
                'index': choice.index,
                'message': {
                    'role': choice.message.role,
                    'content': choice.message.content,
                    'reasoning': getattr(choice.message, 'reasoning', None)
                },
                'finish_reason': choice.finish_reason
            } for choice in response.choices],
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            'elelem_metrics': response.elelem_metrics
        }

    def cleanup_expired(self):
        """Remove expired cache entries using leader election.

        Uses PostgreSQL advisory locks to ensure only one container
        performs cleanup at a time (no race conditions).

        Returns:
            Number of entries deleted (0 if not leader)
        """
        try:
            with self.engine.connect() as conn:
                # Try to acquire advisory lock (non-blocking)
                lock_result = conn.execute(
                    text(f"SELECT pg_try_advisory_lock({self.CLEANUP_LOCK_ID})")
                ).scalar()

                if not lock_result:
                    # Another instance is already cleaning up
                    self.logger.debug("Another instance is handling cache cleanup")
                    return 0

                # We got the lock - we're the leader!
                try:
                    delete_sql = "DELETE FROM response_cache WHERE expires_at < NOW()"
                    result = conn.execute(text(delete_sql))
                    conn.commit()

                    deleted_count = result.rowcount
                    if deleted_count > 0:
                        self.logger.info(f"Cache cleanup: removed {deleted_count} expired entries")

                    return deleted_count

                finally:
                    # Always release the lock
                    conn.execute(text(f"SELECT pg_advisory_unlock({self.CLEANUP_LOCK_ID})"))

        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats (total entries, hits, sizes, etc.)
        """
        try:
            stats_sql = """
            SELECT
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE expires_at > NOW()) as valid_entries,
                SUM(hit_count) as total_hits,
                AVG(hit_count) as avg_hits_per_entry,
                SUM(response_size) as total_size_bytes,
                AVG(response_size) as avg_size_bytes
            FROM response_cache
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(stats_sql)).fetchone()

                return {
                    'total_entries': result[0] or 0,
                    'valid_entries': result[1] or 0,
                    'total_hits': result[2] or 0,
                    'avg_hits_per_entry': float(result[3] or 0),
                    'total_size_bytes': result[4] or 0,
                    'avg_size_bytes': float(result[5] or 0)
                }
        except Exception as e:
            self.logger.warning(f"Failed to get cache stats: {e}")
            return {}
