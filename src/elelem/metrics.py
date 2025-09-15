"""
MetricsStore - Pandas-based metrics collection and analytics for Elelem.

This module provides a loosely-coupled metrics storage system using pandas
for efficient data operations and analytics.
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging


class MetricsStore:
    """Pandas-based metrics storage and analytics."""

    def __init__(self, persist_file: Optional[str] = None):
        """Initialize MetricsStore.

        Args:
            persist_file: Optional file path to persist metrics data
        """
        self.logger = logging.getLogger("elelem.metrics")
        self.persist_file = persist_file

        # Initialize empty DataFrames with proper schema and dtypes
        self.calls_df = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'requested_model': pd.Series(dtype='str'),
            'model': pd.Series(dtype='str'),
            'provider': pd.Series(dtype='str'),
            'tags': pd.Series(dtype='str'),
            'duration_seconds': pd.Series(dtype='float64'),
            'input_tokens': pd.Series(dtype='int64'),
            'output_tokens': pd.Series(dtype='int64'),
            'reasoning_tokens': pd.Series(dtype='int64'),
            'total_tokens': pd.Series(dtype='int64'),
            'input_cost_usd': pd.Series(dtype='float64'),
            'output_cost_usd': pd.Series(dtype='float64'),
            'reasoning_cost_usd': pd.Series(dtype='float64'),
            'total_cost_usd': pd.Series(dtype='float64'),
            'actual_provider': pd.Series(dtype='str'),
            'call_id': pd.Series(dtype='str')
        })

        self.retries_df = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'call_id': pd.Series(dtype='str'),
            'retry_type': pd.Series(dtype='str'),
            'tags': pd.Series(dtype='str'),
            'count': pd.Series(dtype='int64')
        })

        # Load persisted data if file exists
        if self.persist_file and Path(self.persist_file).exists():
            self._load_from_file()

    def record_call(self,
                   model: str,
                   provider: str,
                   tags: List[str],
                   duration_seconds: float,
                   input_tokens: int,
                   output_tokens: int,
                   reasoning_tokens: int,
                   costs: Dict[str, float],
                   actual_provider: Optional[str] = None,
                   call_id: Optional[str] = None,
                   requested_model: Optional[str] = None) -> None:
        """Record a single API call with all its metrics."""

        if call_id is None:
            call_id = f"{int(time.time() * 1000000)}_{hash((model, provider, str(tags)))}"

        record = {
            'timestamp': pd.Timestamp.now(),
            'requested_model': requested_model or model,  # Default to actual model if not specified
            'model': model,
            'provider': provider,
            'tags': ','.join(sorted(tags)) if tags else '',
            'duration_seconds': duration_seconds,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'reasoning_tokens': reasoning_tokens,
            'total_tokens': input_tokens + output_tokens,
            'input_cost_usd': costs.get('input_cost_usd', 0.0),
            'output_cost_usd': costs.get('output_cost_usd', 0.0),
            'reasoning_cost_usd': costs.get('reasoning_cost_usd', 0.0),
            'total_cost_usd': costs.get('total_cost_usd', 0.0),
            'actual_provider': actual_provider,
            'call_id': call_id
        }

        # Use pd.concat instead of deprecated append
        self.calls_df = pd.concat([self.calls_df, pd.DataFrame([record])], ignore_index=True)

        if self.persist_file:
            self._save_to_file()

    def record_retry(self,
                    retry_type: str,
                    tags: List[str],
                    count: int = 1,
                    call_id: Optional[str] = None) -> None:
        """Record retry events."""

        record = {
            'timestamp': pd.Timestamp.now(),
            'call_id': call_id or 'unknown',
            'retry_type': retry_type,
            'tags': ','.join(sorted(tags)) if tags else '',
            'count': count
        }

        self.retries_df = pd.concat([self.retries_df, pd.DataFrame([record])], ignore_index=True)

        if self.persist_file:
            self._save_to_file()

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics compatible with current API."""
        if self.calls_df.empty:
            return self._empty_stats()

        stats = {
            "total_input_tokens": int(self.calls_df['input_tokens'].sum()),
            "total_output_tokens": int(self.calls_df['output_tokens'].sum()),
            "total_tokens": int(self.calls_df['total_tokens'].sum()),
            "total_input_cost_usd": float(self.calls_df['input_cost_usd'].sum()),
            "total_output_cost_usd": float(self.calls_df['output_cost_usd'].sum()),
            "total_cost_usd": float(self.calls_df['total_cost_usd'].sum()),
            "total_calls": len(self.calls_df),
            "total_duration_seconds": float(self.calls_df['duration_seconds'].sum()),
            "avg_duration_seconds": float(self.calls_df['duration_seconds'].mean()),
            "reasoning_tokens": int(self.calls_df['reasoning_tokens'].sum()),
            "reasoning_cost_usd": float(self.calls_df['reasoning_cost_usd'].sum()),
            "providers": self._get_provider_stats(),
            "retry_analytics": self._get_retry_analytics()
        }

        return stats

    def get_stats_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get statistics for a specific tag."""
        # Filter calls that include this tag
        tag_calls = self.calls_df[self.calls_df['tags'].str.contains(tag, na=False)]

        if tag_calls.empty:
            return self._empty_stats_by_tag()

        stats = {
            "total_input_tokens": int(tag_calls['input_tokens'].sum()),
            "total_output_tokens": int(tag_calls['output_tokens'].sum()),
            "total_tokens": int(tag_calls['total_tokens'].sum()),
            "total_input_cost_usd": float(tag_calls['input_cost_usd'].sum()),
            "total_output_cost_usd": float(tag_calls['output_cost_usd'].sum()),
            "total_cost_usd": float(tag_calls['total_cost_usd'].sum()),
            "total_calls": len(tag_calls),
            "total_duration_seconds": float(tag_calls['duration_seconds'].sum()),
            "avg_duration_seconds": float(tag_calls['duration_seconds'].mean()),
            "reasoning_tokens": int(tag_calls['reasoning_tokens'].sum()),
            "reasoning_cost_usd": float(tag_calls['reasoning_cost_usd'].sum()),
            "retry_analytics": self._get_retry_analytics_by_tag(tag)
        }

        return stats

    def _get_provider_stats(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
        """Get provider usage statistics."""
        if df is None:
            df = self.calls_df

        if df.empty:
            return {}

        # Group by actual_provider (for OpenRouter tracking)
        provider_stats = {}
        actual_providers = df[df['actual_provider'].notna()]

        for provider in actual_providers['actual_provider'].unique():
            provider_data = actual_providers[actual_providers['actual_provider'] == provider]
            provider_stats[provider] = {
                "count": len(provider_data),
                "total_cost_usd": float(provider_data['total_cost_usd'].sum()),
                "total_tokens": int(provider_data['total_tokens'].sum()),
                "total_input_tokens": int(provider_data['input_tokens'].sum()),
                "total_output_tokens": int(provider_data['output_tokens'].sum())
            }

        return provider_stats

    def _get_retry_analytics(self) -> Dict[str, int]:
        """Get retry analytics for overall stats."""
        if self.retries_df.empty:
            return {
                "json_parse_retries": 0,
                "json_schema_retries": 0,
                "api_json_validation_retries": 0,
                "rate_limit_retries": 0,
                "total_retries": 0,
                "temperature_reductions": 0,
                "final_failures": 0,
                "response_format_removals": 0,
                "candidate_iterations": 0
            }

        retry_counts = self.retries_df.groupby('retry_type')['count'].sum().to_dict()

        # Fill in missing retry types with 0
        all_retry_types = [
            "json_parse_retries", "json_schema_retries", "api_json_validation_retries",
            "rate_limit_retries", "temperature_reductions", "final_failures",
            "response_format_removals", "candidate_iterations"
        ]

        result = {retry_type: retry_counts.get(retry_type, 0) for retry_type in all_retry_types}
        result["total_retries"] = sum(result.values())

        return result

    def _get_retry_analytics_by_tag(self, tag: str) -> Dict[str, int]:
        """Get retry analytics for a specific tag."""
        tag_retries = self.retries_df[self.retries_df['tags'].str.contains(tag, na=False)]

        if tag_retries.empty:
            return {
                "json_parse_retries": 0,
                "json_schema_retries": 0,
                "api_json_validation_retries": 0,
                "rate_limit_retries": 0,
                "total_retries": 0,
                "temperature_reductions": 0,
                "final_failures": 0,
                "response_format_removals": 0,
                "candidate_iterations": 0
            }

        retry_counts = tag_retries.groupby('retry_type')['count'].sum().to_dict()

        all_retry_types = [
            "json_parse_retries", "json_schema_retries", "api_json_validation_retries",
            "rate_limit_retries", "temperature_reductions", "final_failures",
            "response_format_removals", "candidate_iterations"
        ]

        result = {retry_type: retry_counts.get(retry_type, 0) for retry_type in all_retry_types}
        result["total_retries"] = sum(result.values())

        return result

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty stats structure."""
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_input_cost_usd": 0.0,
            "total_output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_duration_seconds": 0.0,
            "avg_duration_seconds": 0.0,
            "reasoning_tokens": 0,
            "reasoning_cost_usd": 0.0,
            "providers": {},
            "retry_analytics": {
                "json_parse_retries": 0,
                "json_schema_retries": 0,
                "api_json_validation_retries": 0,
                "rate_limit_retries": 0,
                "total_retries": 0,
                "temperature_reductions": 0,
                "final_failures": 0,
                "response_format_removals": 0,
                "candidate_iterations": 0
            }
        }

    def _empty_stats_by_tag(self) -> Dict[str, Any]:
        """Return empty stats structure for tag-specific queries (no providers field)."""
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_input_cost_usd": 0.0,
            "total_output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_duration_seconds": 0.0,
            "avg_duration_seconds": 0.0,
            "reasoning_tokens": 0,
            "reasoning_cost_usd": 0.0,
            "retry_analytics": {
                "json_parse_retries": 0,
                "json_schema_retries": 0,
                "api_json_validation_retries": 0,
                "rate_limit_retries": 0,
                "total_retries": 0,
                "temperature_reductions": 0,
                "final_failures": 0,
                "response_format_removals": 0,
                "candidate_iterations": 0
            }
        }

    # ============= Phase 5: Time-Series Query Capabilities =============

    def get_summary(self,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive summary statistics for a time range.

        Args:
            start_time: Filter calls after this time (inclusive). None = no lower bound
            end_time: Filter calls before this time (inclusive). None = no upper bound
            tags: Filter by specific tags. None = all tags

        Returns:
            Dict with aggregated metrics for tokens, costs, duration, and retry analytics
        """
        # Get filtered DataFrames
        calls_df = self.get_dataframe(start_time, end_time, tags)

        # Filter retries DataFrame with same criteria
        retries_df = self.retries_df.copy()
        if not retries_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(retries_df['timestamp']):
                retries_df['timestamp'] = pd.to_datetime(retries_df['timestamp'])
            if start_time:
                retries_df = retries_df[retries_df['timestamp'] >= start_time]
            if end_time:
                retries_df = retries_df[retries_df['timestamp'] <= end_time]
            if tags:
                mask = retries_df['tags'].apply(lambda x: any(tag in str(x).split(',') for tag in tags))
                retries_df = retries_df[mask]

        if calls_df.empty:
            return self._empty_summary()

        # Helper to compute stats for a column
        def compute_stats(column_name):
            return {
                'total': float(calls_df[column_name].sum()),
                'avg': float(calls_df[column_name].mean()),
                'min': float(calls_df[column_name].min()),
                'max': float(calls_df[column_name].max())
            }

        # Calculate success/failure counts from retry analytics
        total_failures = self._calculate_retry_analytics(retries_df).get('final_failures', 0)
        total_successes = len(calls_df)  # Calls that made it to the DataFrame are successful
        total_attempts = total_successes + total_failures

        return {
            # Overall metrics
            "total_calls": total_attempts,  # All attempts (success + failure)
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": float(total_successes / total_attempts) if total_attempts > 0 else 0.0,

            # Token metrics
            "input_tokens": compute_stats('input_tokens'),
            "output_tokens": compute_stats('output_tokens'),
            "reasoning_tokens": compute_stats('reasoning_tokens'),

            # Cost metrics
            "input_cost_usd": compute_stats('input_cost_usd'),
            "output_cost_usd": compute_stats('output_cost_usd'),
            "reasoning_cost_usd": compute_stats('reasoning_cost_usd'),
            "total_cost_usd": compute_stats('total_cost_usd'),

            # Duration metrics
            "duration_seconds": compute_stats('duration_seconds'),

            # Retry analytics
            "retry_analytics": self._calculate_retry_analytics(retries_df)
        }

    def get_dataframe(self,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     tags: Optional[List[str]] = None) -> pd.DataFrame:
        """Get filtered DataFrame for custom analysis.

        Args:
            start_time: Filter calls after this time (inclusive). None = no lower bound
            end_time: Filter calls before this time (inclusive). None = no upper bound
            tags: Filter by specific tags. None = all tags

        Returns:
            Filtered copy of the calls DataFrame with columns:
            timestamp, model, provider, tags, duration_seconds,
            input_tokens, output_tokens, reasoning_tokens, total_tokens,
            input_cost_usd, output_cost_usd, reasoning_cost_usd, total_cost_usd,
            actual_provider, call_id
        """
        df = self.calls_df.copy()

        if df.empty:
            return df

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Apply filters
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        if tags:
            mask = df['tags'].apply(lambda x: any(tag in str(x).split(',') for tag in tags))
            df = df[mask]

        return df

    def _calculate_retry_analytics(self, retries_df: pd.DataFrame) -> Dict[str, int]:
        """Calculate retry analytics from filtered retries DataFrame."""
        if retries_df.empty:
            return {
                "json_parse_retries": 0,
                "json_schema_retries": 0,
                "api_json_validation_retries": 0,
                "rate_limit_retries": 0,
                "total_retries": 0,
                "temperature_reductions": 0,
                "final_failures": 0,
                "response_format_removals": 0,
                "candidate_iterations": 0
            }

        retry_counts = retries_df.groupby('retry_type')['count'].sum().to_dict()

        all_retry_types = [
            "json_parse_retries", "json_schema_retries", "api_json_validation_retries",
            "rate_limit_retries", "temperature_reductions", "final_failures",
            "response_format_removals", "candidate_iterations"
        ]

        result = {retry_type: int(retry_counts.get(retry_type, 0)) for retry_type in all_retry_types}
        result["total_retries"] = sum(result.values())

        return result

    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure for time-range queries."""
        empty_stats = {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0}

        return {
            "total_calls": 0,
            "total_successes": 0,
            "total_failures": 0,
            "success_rate": 0.0,
            "input_tokens": empty_stats.copy(),
            "output_tokens": empty_stats.copy(),
            "reasoning_tokens": empty_stats.copy(),
            "input_cost_usd": empty_stats.copy(),
            "output_cost_usd": empty_stats.copy(),
            "reasoning_cost_usd": empty_stats.copy(),
            "total_cost_usd": empty_stats.copy(),
            "duration_seconds": empty_stats.copy(),
            "retry_analytics": {
                "json_parse_retries": 0,
                "json_schema_retries": 0,
                "api_json_validation_retries": 0,
                "rate_limit_retries": 0,
                "total_retries": 0,
                "temperature_reductions": 0,
                "final_failures": 0,
                "response_format_removals": 0,
                "candidate_iterations": 0
            }
        }

    def get_unique_tags(self) -> List[str]:
        """Get all unique tags from the metrics data.

        Returns:
            Sorted list of unique tags
        """
        if self.calls_df.empty:
            return []

        # Get all tags from calls
        all_tags = []
        for tags_str in self.calls_df['tags'].dropna():
            if tags_str:
                all_tags.extend(tags_str.split(','))

        # Get unique tags and sort
        unique_tags = sorted(set(all_tags))
        return unique_tags

    # ============= End of Phase 5 =============

    def _save_to_file(self) -> None:
        """Save metrics data to file."""
        try:
            # Save both DataFrames to separate sheets in Excel format
            # or use pickle for now (we can enhance this later)
            data = {
                'calls': self.calls_df,
                'retries': self.retries_df
            }
            pd.to_pickle(data, self.persist_file)
        except Exception as e:
            self.logger.warning(f"Failed to save metrics to {self.persist_file}: {e}")

    def _load_from_file(self) -> None:
        """Load metrics data from file."""
        try:
            data = pd.read_pickle(self.persist_file)
            self.calls_df = data.get('calls', pd.DataFrame())
            self.retries_df = data.get('retries', pd.DataFrame())
            self.logger.info(f"Loaded {len(self.calls_df)} calls and {len(self.retries_df)} retry records")
        except Exception as e:
            self.logger.warning(f"Failed to load metrics from {self.persist_file}: {e}")
            # Initialize empty DataFrames on failure
            self.calls_df = pd.DataFrame()
            self.retries_df = pd.DataFrame()

    def reset(self) -> None:
        """Reset all metrics data."""
        self.calls_df = pd.DataFrame(columns=self.calls_df.columns)
        self.retries_df = pd.DataFrame(columns=self.retries_df.columns)

        if self.persist_file:
            self._save_to_file()