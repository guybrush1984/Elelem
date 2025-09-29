"""
Elelem Request Metrics Table Schema

This module defines the database schema for Elelem request metrics.
It's independent of the main Elelem codebase and can be used by any dashboard.
"""

# Table schema for request_metrics
REQUEST_METRICS_COLUMNS = [
    'request_id',
    'timestamp',
    'requested_model',
    'selected_candidate',
    'actual_model',
    'actual_provider',
    'temperature',
    'initial_temperature',
    'max_tokens',
    'stream',
    'tags',
    'status',
    'total_duration_seconds',
    'input_tokens',
    'output_tokens',
    'reasoning_tokens',
    'total_cost_usd',
    'final_error_type',
    'final_error_message',
    'total_tokens_per_second',
    'cost_per_token',
    'json_parse_retries',
    'json_schema_retries',
    'api_json_validation_retries',
    'rate_limit_retries',
    'temperature_reductions',
    'response_format_removals',
    'candidate_iterations',
    'final_failures',
    'total_retry_attempts'
]

# Important columns for dashboard display
DISPLAY_COLUMNS = [
    'timestamp',
    'requested_model',
    'actual_model',
    'status',
    'total_duration_seconds',
    'input_tokens',
    'output_tokens',
    'reasoning_tokens',
    'total_cost_usd',
    'tags',
    'final_error_type'
]

# Column data types for proper pandas handling
COLUMN_TYPES = {
    'request_id': 'string',
    'timestamp': 'datetime64[ns]',
    'requested_model': 'string',
    'selected_candidate': 'string',
    'actual_model': 'string',
    'actual_provider': 'string',
    'temperature': 'float64',
    'initial_temperature': 'float64',
    'max_tokens': 'Int64',
    'stream': 'boolean',
    'tags': 'object',  # JSON
    'status': 'string',
    'total_duration_seconds': 'float64',
    'input_tokens': 'Int64',
    'output_tokens': 'Int64',
    'reasoning_tokens': 'Int64',
    'total_cost_usd': 'float64',
    'final_error_type': 'string',
    'final_error_message': 'string',
    'total_tokens_per_second': 'float64',
    'cost_per_token': 'float64',
    'json_parse_retries': 'Int64',
    'json_schema_retries': 'Int64',
    'api_json_validation_retries': 'Int64',
    'rate_limit_retries': 'Int64',
    'temperature_reductions': 'Int64',
    'response_format_removals': 'Int64',
    'candidate_iterations': 'Int64',
    'final_failures': 'Int64',
    'total_retry_attempts': 'Int64'
}