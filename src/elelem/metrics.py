"""
MetricsStore - Unified SQLAlchemy-based metrics collection with pandas analytics for Elelem.

This module provides a unified metrics storage system using SQLAlchemy for storage
(SQLite for local, PostgreSQL for production) and pandas for readable analytics.
"""

import pandas as pd
import time
import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class RequestTracker:
    """Tracks metrics for a single request lifecycle.

    This class is the single source of truth for request data structure.
    MetricsStore will use this to build its unified DataFrame.
    """

    # Define retry types as class constant for consistency
    RETRY_TYPES = [
        'json_parse_retries',
        'json_schema_retries',
        'api_json_validation_retries',
        'rate_limit_retries',
        'temperature_reductions',
        'response_format_removals',
        'candidate_iterations',
        'final_failures'
    ]

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.timestamp = datetime.now()

        # Request details
        self.requested_model: Optional[str] = None
        self.selected_candidate: Optional[str] = None
        self.actual_model: Optional[str] = None
        self.actual_provider: Optional[str] = None
        self.tags: List[str] = []

        # Request parameters
        self.temperature: Optional[float] = None
        self.initial_temperature: Optional[float] = None
        self.max_tokens: Optional[int] = None
        self.stream: bool = False

        # Outcome
        self.status: str = 'pending'
        self.final_error_type: Optional[str] = None
        self.final_error_message: Optional[str] = None

        # Performance metrics
        self.first_byte_time: Optional[float] = None
        self.completion_time: Optional[float] = None

        # Token & cost metrics
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.reasoning_tokens: int = 0
        self.total_cost_usd: float = 0.0

        # Initialize retry counters
        self.retry_counts = {retry_type: 0 for retry_type in self.RETRY_TYPES}

    def validate_tags(self, tags: List[str]) -> List[str]:
        """Validate and clean tags list.

        Args:
            tags: List of tag strings

        Returns:
            Validated and cleaned tags list

        Raises:
            ValueError: If validation fails
        """
        if not tags:
            return []

        if len(tags) > 10:
            raise ValueError(f"Too many tags: {len(tags)}. Maximum 10 tags allowed.")

        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ValueError(f"Tag must be string, got {type(tag)}: {tag}")

            # Clean and validate tag
            clean_tag = tag.strip()
            if not clean_tag:
                continue  # Skip empty tags

            if len(clean_tag) > 50:
                raise ValueError(f"Tag too long: '{clean_tag}'. Maximum 50 characters allowed.")

            if ',' in clean_tag:
                raise ValueError(f"Tag cannot contain commas: '{clean_tag}'")

            validated_tags.append(clean_tag)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in validated_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags

    def record_retry(self, retry_type: str, count: int = 1):
        """Record a retry event."""
        if retry_type not in self.RETRY_TYPES:
            raise ValueError(f"Unknown retry type: {retry_type}. Valid types: {', '.join(self.RETRY_TYPES)}")

        self.retry_counts[retry_type] += count

    def mark_first_byte(self):
        """Mark when first byte was received."""
        self.first_byte_time = time.time()

    def finalize(self, status: str = 'success', **kwargs):
        """Finalize the request with outcome data.

        Args:
            status: 'success', 'failed', or 'timeout'
            **kwargs: Additional fields like input_tokens, output_tokens, etc.
        """
        self.status = status
        self.completion_time = time.time()

        # Update any provided fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def finalize_with_candidate(self, metrics_store, selected_candidate: str, actual_model: str,
                               actual_provider: str, status: str = 'success', **kwargs):
        """Set candidate details, finalize, and store in one call.

        Args:
            metrics_store: The MetricsStore instance to finalize with
            selected_candidate: The candidate that was selected (e.g., "openai:gpt-4")
            actual_model: The actual model used
            actual_provider: The actual provider used
            status: 'success', 'failed', or 'timeout'
            **kwargs: Additional fields like input_tokens, output_tokens, etc.
        """
        self.selected_candidate = selected_candidate
        self.actual_model = actual_model
        self.actual_provider = actual_provider
        self.finalize(status=status, **kwargs)
        metrics_store.finalize_request(self)

    def finalize_failure(self, metrics_store, error_type: str, error_message: str = None, retry_type: str = "final_failures"):
        """Record a retry, finalize as failed, and store in one call.

        Args:
            metrics_store: The MetricsStore instance to finalize with
            error_type: Type of error (e.g., "ModelNotFound", "RateLimitError")
            error_message: Error message
            retry_type: Type of retry to record (defaults to "final_failures")
        """
        self.record_retry(retry_type)
        self.finalize(status="failed", final_error_type=error_type, final_error_message=error_message or error_type)
        metrics_store.finalize_request(self)

    def to_record(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for DataFrame/SQL insertion."""
        total_duration = (self.completion_time or time.time()) - self.start_time

        # Calculate derived metrics
        record = {
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'requested_model': self.requested_model,
            'selected_candidate': self.selected_candidate,
            'actual_model': self.actual_model,
            'actual_provider': self.actual_provider,
            'temperature': self.temperature,
            'initial_temperature': self.initial_temperature,
            'max_tokens': self.max_tokens,
            'stream': self.stream,
            'tags': self.tags.copy() if self.tags else [],
            'status': self.status,
            'total_duration_seconds': total_duration,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'reasoning_tokens': self.reasoning_tokens,
            'total_cost_usd': self.total_cost_usd,
            'final_error_type': self.final_error_type,
            'final_error_message': self.final_error_message
        }

        # Add performance metrics if available
        if self.first_byte_time:
            record['first_byte_latency_ms'] = int((self.first_byte_time - self.start_time) * 1000)

        if self.completion_time and self.first_byte_time:
            completion_ms = int((self.completion_time - self.first_byte_time) * 1000)
            record['completion_latency_ms'] = completion_ms
            if self.output_tokens > 0 and completion_ms > 0:
                record['tokens_per_second'] = self.output_tokens / (completion_ms / 1000)

        total_tokens = self.input_tokens + self.output_tokens

        # Always include these fields, even if 0
        record['total_tokens_per_second'] = total_tokens / total_duration if total_duration > 0 and total_tokens > 0 else 0.0
        record['cost_per_token'] = self.total_cost_usd / total_tokens if total_tokens > 0 and self.total_cost_usd > 0 else 0.0

        # Add all retry counts
        record.update(self.retry_counts)
        record['total_retry_attempts'] = sum(self.retry_counts.values())

        return record

    @classmethod
    def get_schema(cls) -> Dict[str, type]:
        """Get the schema for creating DataFrames or SQL tables."""
        return {
            'request_id': str,
            'timestamp': 'datetime64[ns]',
            'requested_model': str,
            'selected_candidate': str,
            'actual_model': str,
            'actual_provider': str,
            'temperature': float,
            'initial_temperature': float,
            'max_tokens': int,
            'stream': bool,
            'tags': list,
            'status': str,
            'total_duration_seconds': float,
            'first_byte_latency_ms': int,
            'completion_latency_ms': int,
            'input_tokens': int,
            'output_tokens': int,
            'reasoning_tokens': int,
            'tokens_per_second': float,
            'total_tokens_per_second': float,
            'total_cost_usd': float,
            'cost_per_token': float,
            **{retry_type: int for retry_type in cls.RETRY_TYPES},
            'total_retry_attempts': int,
            'final_error_type': str,
            'final_error_message': str
        }


class MetricsStore:
    """Unified metrics storage using SQLAlchemy with pandas for analytics."""

    def __init__(self):
        """Initialize MetricsStore with unified SQLAlchemy backend."""
        self.logger = logging.getLogger("elelem.metrics")

        if not SQLALCHEMY_AVAILABLE:
            self.logger.error("SQLAlchemy not available. Install with: pip install sqlalchemy")
            raise ImportError("SQLAlchemy required for metrics storage")

        # Unified storage backend - auto-detect SQLite vs PostgreSQL
        database_url = os.getenv('ELELEM_DATABASE_URL')
        if not database_url:
            # Default to SQLite file storage for local development
            database_url = "sqlite:///elelem_metrics.db"

        self.engine = create_engine(database_url)
        self.active_requests: Dict[str, RequestTracker] = {}

        # Ensure database schema exists
        self._ensure_schema()

    def _ensure_schema(self):
        """Create metrics table if it doesn't exist (works for both SQLite and PostgreSQL)."""
        try:
            # Create table with schema compatible with both SQLite and PostgreSQL
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS request_metrics (
                request_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                requested_model TEXT,
                selected_candidate TEXT,
                actual_model TEXT,
                actual_provider TEXT,
                temperature REAL,
                initial_temperature REAL,
                max_tokens INTEGER,
                stream BOOLEAN DEFAULT false,
                tags TEXT DEFAULT '[]',  -- JSON as TEXT for SQLite compatibility
                status TEXT NOT NULL,
                total_duration_seconds REAL NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                final_error_type TEXT,
                final_error_message TEXT,
                total_tokens_per_second REAL,
                cost_per_token REAL,
                json_parse_retries INTEGER DEFAULT 0,
                json_schema_retries INTEGER DEFAULT 0,
                api_json_validation_retries INTEGER DEFAULT 0,
                rate_limit_retries INTEGER DEFAULT 0,
                temperature_reductions INTEGER DEFAULT 0,
                response_format_removals INTEGER DEFAULT 0,
                candidate_iterations INTEGER DEFAULT 0,
                final_failures INTEGER DEFAULT 0,
                total_retry_attempts INTEGER DEFAULT 0
            )
            """

            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()

            self.logger.info("Metrics database schema initialized")
        except Exception as e:
            self.logger.error(f"Failed to create database schema: {e}")
            raise

    def start_request(self, request_id: Optional[str] = None, requested_model: str = None,
                     tags: List[str] = None, temperature: float = None,
                     max_tokens: int = None, stream: bool = False) -> RequestTracker:
        """Start tracking a new request.

        Args:
            request_id: Optional request ID
            requested_model: Model requested by user
            tags: Request tags
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            stream: Whether streaming is enabled

        Returns:
            RequestTracker instance for the request
        """
        tracker = RequestTracker(request_id)

        # Set initial request details
        if requested_model:
            tracker.requested_model = requested_model
        if tags:
            # Validate tags before setting
            tracker.tags = tracker.validate_tags(tags)

        # Set request parameters
        tracker.temperature = temperature
        tracker.initial_temperature = temperature
        tracker.max_tokens = max_tokens
        tracker.stream = stream

        self.active_requests[tracker.request_id] = tracker
        return tracker

    def finalize_request(self, tracker: RequestTracker):
        """Finalize and record a completed request using unified SQLAlchemy backend.

        Args:
            tracker: The RequestTracker to finalize and store
        """
        # Convert tracker to record
        record = tracker.to_record()

        # Remove from active requests
        self.active_requests.pop(tracker.request_id, None)

        # Save to unified database backend (SQLite or PostgreSQL)
        self._save_record(record)

    def _save_record(self, record: Dict[str, Any]):
        """Save a single request record to the unified database backend."""
        try:
            # Convert tags list to JSON string for database storage
            record_copy = record.copy()
            if 'tags' in record_copy:
                record_copy['tags'] = json.dumps(record_copy['tags']) if record_copy['tags'] else '[]'

            # Direct SQLAlchemy insert - works for both SQLite and PostgreSQL
            insert_sql = """
            INSERT INTO request_metrics (
                request_id, timestamp, requested_model, selected_candidate, actual_model, actual_provider,
                temperature, initial_temperature, max_tokens, stream, tags, status, total_duration_seconds,
                input_tokens, output_tokens, reasoning_tokens, total_cost_usd, final_error_type, final_error_message,
                total_tokens_per_second, cost_per_token, json_parse_retries, json_schema_retries,
                api_json_validation_retries, rate_limit_retries, temperature_reductions, response_format_removals,
                candidate_iterations, final_failures, total_retry_attempts
            ) VALUES (
                :request_id, :timestamp, :requested_model, :selected_candidate, :actual_model, :actual_provider,
                :temperature, :initial_temperature, :max_tokens, :stream, :tags, :status, :total_duration_seconds,
                :input_tokens, :output_tokens, :reasoning_tokens, :total_cost_usd, :final_error_type, :final_error_message,
                :total_tokens_per_second, :cost_per_token, :json_parse_retries, :json_schema_retries,
                :api_json_validation_retries, :rate_limit_retries, :temperature_reductions, :response_format_removals,
                :candidate_iterations, :final_failures, :total_retry_attempts
            )
            """

            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), record_copy)
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to save request to database: {e}")

    def get_stats(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive statistics using pandas for readable analytics.

        Args:
            start_time: Filter requests after this time (inclusive). None = no lower bound
            end_time: Filter requests before this time (inclusive). None = no upper bound
            tags: Filter by specific tags. None = all tags

        Returns:
            Dict with aggregated metrics for tokens, costs, duration, and retry analytics
        """
        # Load data from database using pandas
        df = self._load_dataframe(start_time, end_time, tags)

        if df.empty:
            return self._empty_stats()

        # Use pandas for readable analytics
        successful = df[df['status'] == 'success']
        failed = df[df['status'] == 'failed']

        # Helper to compute stats for a column
        def compute_stats(data, column_name):
            if data.empty or column_name not in data.columns:
                return {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0}
            return {
                'total': float(data[column_name].sum()),
                'avg': float(data[column_name].mean()),
                'min': float(data[column_name].min()),
                'max': float(data[column_name].max())
            }

        return {
            'requests': {
                'total': len(df),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': float(len(successful) / len(df)) if len(df) > 0 else 0.0
            },
            'tokens': {
                'input': compute_stats(successful, 'input_tokens'),
                'output': compute_stats(successful, 'output_tokens'),
                'reasoning': compute_stats(successful, 'reasoning_tokens')
            },
            'costs': compute_stats(successful, 'total_cost_usd'),
            'duration': compute_stats(df, 'total_duration_seconds'),
            'providers': successful.groupby('actual_provider')['total_cost_usd'].sum().to_dict() if not successful.empty else {},
            'models': successful.groupby('actual_model')['total_cost_usd'].sum().to_dict() if not successful.empty else {},
            'retries': {
                'json_parse_retries': int(df['json_parse_retries'].sum()) if 'json_parse_retries' in df.columns else 0,
                'rate_limit_retries': int(df['rate_limit_retries'].sum()) if 'rate_limit_retries' in df.columns else 0,
                'total_retry_attempts': int(df['total_retry_attempts'].sum()) if 'total_retry_attempts' in df.columns else 0
            }
        }

    def _load_dataframe(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, tags: Optional[List[str]] = None) -> pd.DataFrame:
        """Load filtered data from database using pandas."""
        try:
            # Build query with proper conditional WHERE clause
            query = "SELECT * FROM request_metrics"
            conditions = []
            params = {}

            if start_time:
                conditions.append("timestamp >= :start_time")
                params['start_time'] = start_time

            if end_time:
                conditions.append("timestamp <= :end_time")
                params['end_time'] = end_time

            if tags:
                # Filter by tags - works for JSON stored as TEXT
                tag_conditions = []
                for i, tag in enumerate(tags):
                    tag_key = f'tag_{i}'
                    tag_conditions.append(f"tags LIKE :{tag_key}")
                    params[tag_key] = f'%"{tag}"%'

                if tag_conditions:
                    conditions.append(f"({' OR '.join(tag_conditions)})")

            # Add WHERE clause only if we have conditions
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY timestamp DESC"

            # Load with pandas
            df = pd.read_sql(query, self.engine, params=params)

            # Convert JSON strings back to lists for tags
            if 'tags' in df.columns and not df.empty:
                df['tags'] = df['tags'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
                )

            return df

        except Exception as e:
            self.logger.error(f"Failed to load data from database: {e}")
            return pd.DataFrame()

    def get_dataframe(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, tags: Optional[List[str]] = None) -> pd.DataFrame:
        """Public method to get filtered metrics data as DataFrame.

        Args:
            start_time: Filter calls after this time (inclusive). None = no lower bound
            end_time: Filter calls before this time (inclusive). None = no upper bound
            tags: Filter by specific tags. None = all tags

        Returns:
            Filtered pandas DataFrame with all metrics data
        """
        return self._load_dataframe(start_time, end_time, tags)

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty stats structure."""
        return {
            'requests': {'total': 0, 'successful': 0, 'failed': 0, 'success_rate': 0.0},
            'tokens': {
                'input': {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0},
                'output': {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0},
                'reasoning': {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0}
            },
            'costs': {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0},
            'duration': {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0},
            'providers': {},
            'models': {},
            'retries': {'json_parse_retries': 0, 'rate_limit_retries': 0, 'total_retry_attempts': 0}
        }

    def get_available_tags(self) -> List[str]:
        """Get all unique tags from the database."""
        try:
            query = "SELECT DISTINCT tags FROM request_metrics WHERE tags IS NOT NULL AND tags != '[]'"
            df = pd.read_sql(query, self.engine)

            all_tags = set()
            for tags_json in df['tags']:
                if isinstance(tags_json, str):
                    tags_list = json.loads(tags_json)
                    all_tags.update(tags_list)

            return sorted(list(all_tags))
        except Exception as e:
            self.logger.error(f"Failed to get available tags: {e}")
            return []

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the metrics system."""
        is_postgresql = "postgresql" in str(self.engine.url)

        try:
            # Test database connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM request_metrics"))
                total_requests = result.scalar()

            if is_postgresql:
                return {
                    "status": "healthy",
                    "total_requests": total_requests,
                    "postgresql": {
                        "enabled": True,
                        "connected": True
                    }
                }
            else:
                return {
                    "status": "healthy",
                    "total_requests": total_requests,
                    "postgresql": {
                        "enabled": False,
                        "connected": False
                    }
                }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "total_requests": 0,
                "postgresql": {
                    "enabled": is_postgresql,
                    "connected": False,
                    "error": str(e)
                }
            }


