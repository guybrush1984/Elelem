"""
Elelem OpenAI API Compatibility Server

A FastAPI server that provides OpenAI-compatible endpoints using Elelem's
multi-provider backend for resilient AI API access.

Usage:
    uvicorn elelem.server.main:app --host 0.0.0.0 --port 8000
"""

import logging
import traceback
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from elelem import Elelem
from elelem import __version__
from elelem._benchmark_store import get_benchmark_store
from elelem.server.models import (
    ChatCompletionRequest,
    ErrorResponse,
    HealthResponse
)

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Elelem OpenAI API Server",
    description="OpenAI-compatible API server with multi-provider backend",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
elelem = None


@app.on_event("startup")
async def startup_event():
    """Initialize Elelem on server startup."""
    global elelem
    logger.info("🚀 Starting Elelem server...")

    # Read cache configuration from environment
    cache_enabled = os.getenv('ELELEM_CACHE_ENABLED', 'false').lower() == 'true'
    cache_ttl = int(os.getenv('ELELEM_CACHE_TTL', '300'))
    cache_max_size = int(os.getenv('ELELEM_CACHE_MAX_SIZE', '50000'))

    # Initialize Elelem instance (auto-creates PostgreSQL tables if configured)
    elelem = Elelem(
        cache_enabled=cache_enabled,
        cache_ttl=cache_ttl,
        cache_max_size=cache_max_size
    )

    # Log startup status using proper encapsulation
    if os.getenv('ELELEM_DATABASE_URL'):
        health = elelem.get_health_status()
        if health["postgresql"]["connected"]:
            logger.info("✅ PostgreSQL metrics backend ready")
        else:
            logger.warning(f"⚠️ PostgreSQL connection issue: {health['postgresql']['error']}")
    else:
        logger.info("📊 Running with SQLite metrics backend")

    # Log cache status
    if cache_enabled:
        logger.info(f"✅ Response cache enabled (TTL: {cache_ttl}s, max size: {cache_max_size} bytes)")

        # Start background cleanup task (checks every 10s, cleans when needed)
        import asyncio
        asyncio.create_task(cache_cleanup_task())
    else:
        logger.info("📦 Response cache disabled")

    # Start benchmark fetching if configured
    benchmark_store = get_benchmark_store()
    await benchmark_store.start_background_fetch()
    if benchmark_store.enabled:
        logger.info(f"📊 Benchmark routing enabled (source: {benchmark_store.source})")


async def cache_cleanup_task():
    """Background task to cleanup expired cache entries.

    Each worker tries cleanup at the specified interval.
    PostgreSQL advisory lock ensures only one worker cleans at a time.
    """
    import asyncio

    # Get cleanup interval from environment (default 600s = 10 min)
    cleanup_interval = int(os.getenv('ELELEM_CACHE_CLEANUP_INTERVAL', '600'))

    logger.info(f"Cache cleanup task started (interval: {cleanup_interval}s)")

    while True:
        try:
            await asyncio.sleep(cleanup_interval)

            if elelem and elelem.cache:
                # Try to cleanup (acquires lock, only one worker succeeds)
                deleted = elelem.cache.cleanup_expired()

                if deleted == 0:
                    logger.debug("Cache cleanup: no expired entries or another worker is cleaning")

        except Exception as e:
            logger.error(f"Cache cleanup task error: {e}")



@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on server shutdown."""
    global elelem
    logger.info("🛑 Shutting down Elelem server...")

    # Stop benchmark fetching
    benchmark_store = get_benchmark_store()
    benchmark_store.stop()

    # Clean up using proper encapsulation
    if elelem:
        try:
            elelem.close()
            logger.info("✅ Elelem resources cleaned up")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    logger.info("👋 Shutdown complete")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to return OpenAI-compatible errors."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    # Return OpenAI-compatible error format
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status."""
    if not elelem:
        return {"status": "error", "message": "Elelem not initialized"}

    health = elelem.get_health_status()

    # Simple cache status
    cache_info = {"enabled": elelem.cache is not None}

    # Determine overall status
    overall_status = "healthy"
    if health["postgresql"]["enabled"] and not health["postgresql"]["connected"]:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "version": __version__,
        "metrics": health,
        "cache": cache_info
    }


@app.get("/v1/models")
async def list_models():
    """List available models - OpenAI compatible."""
    try:
        return elelem.list_models()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "models_list_error"
                }
            }
        )


@app.get("/v1/benchmark/status")
async def get_benchmark_status():
    """Get benchmark routing status and current data.

    Returns:
        - enabled: Whether benchmark routing is configured
        - source: The configured benchmark source (file path or URL)
        - interval_seconds: How often benchmarks are refreshed
        - last_fetch: Timestamp of last successful fetch
        - last_error: Error message from last failed fetch (if any)
        - entries_count: Number of models with benchmark data
    """
    store = get_benchmark_store()
    return store.get_status()


@app.get("/v1/benchmark/data")
async def get_benchmark_data():
    """Get all current benchmark data.

    Returns a dict mapping model references to their benchmark metrics:
    - tokens_per_second: Average output tokens per second
    - cost_per_1m_output: Cost per 1M output tokens (extrapolated)
    - avg_duration: Average request duration in seconds
    - success_rate: Request success rate (0.0 to 1.0)
    - sample_count: Number of benchmark samples
    """
    store = get_benchmark_store()
    return {
        "enabled": store.enabled,
        "data": store.get_all_benchmarks()
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Create chat completion - OpenAI compatible."""
    try:
        # Convert Pydantic model to dict and pass to Elelem
        request_dict = request.model_dump(exclude_none=True)

        # Extract messages and model (required)
        messages = request_dict.pop("messages")
        model = request_dict.pop("model")

        # Convert messages to dict format for Elelem
        # Messages are already dicts from OpenAI SDK, but may be Pydantic objects from request
        messages_dict = []
        for msg in messages:
            if hasattr(msg, 'role'):  # Pydantic object
                messages_dict.append({"role": msg.role, "content": msg.content})
            else:  # Already a dict
                messages_dict.append(msg)

        # Call Elelem's create_chat_completion
        response = await elelem.create_chat_completion(
            messages=messages_dict,
            model=model,
            **request_dict
        )

        return response

    except ValueError as e:
        # Handle Elelem validation errors
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }
        )
    except Exception as e:
        # Handle other errors
        logger.error(f"Error in chat completion: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "completion_error"
                }
            }
        )


@app.get("/v1/metrics/summary")
async def get_metrics_summary(
    start_time: Optional[datetime] = Query(None, description="Filter calls after this time (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="Filter calls before this time (ISO format)"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags to filter by")
):
    """Get metrics summary with optional time and tag filters.

    Returns aggregated metrics including:
    - Token usage (input, output, reasoning) with total/avg/min/max
    - Costs breakdown with total/avg/min/max
    - Duration statistics
    - Retry analytics
    """
    try:
        tag_list = tags.split(',') if tags else None
        summary = elelem.get_summary(start_time, end_time, tag_list)
        return summary
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/metrics/tags")
async def get_metrics_tags():
    """Get all unique tags from metrics data.

    Returns list of tags including automatic tags like:
    - model:groq:openai/gpt-oss-20b
    - provider:groq
    - Any user-defined tags
    """
    try:
        tags = elelem.get_metrics_tags()
        return {"tags": tags}
    except Exception as e:
        logger.error(f"Error getting metrics tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/metrics/data")
async def get_metrics_data(
    start_time: Optional[datetime] = Query(None, description="Filter calls after this time (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="Filter calls before this time (ISO format)"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags to filter by"),
    format: str = Query("json", description="Output format (currently only 'json' supported)")
):
    """Get raw unified metrics data as JSON array.

    Returns array of request records from unified metrics structure.
    Works with both PostgreSQL and local pandas.
    """
    try:
        if format != "json":
            raise HTTPException(status_code=400, detail="Only 'json' format is currently supported")

        tag_list = tags.split(',') if tags else None
        df = elelem.get_metrics_dataframe(start_time, end_time, tag_list)

        # Convert DataFrame to JSON records
        # Handle datetime serialization
        df_copy = df.copy()
        if 'timestamp' in df_copy.columns and not df_copy.empty:
            # Convert to datetime if not already
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

        return df_copy.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error getting metrics data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Elelem OpenAI API Server",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "metrics_summary": "/v1/metrics/summary",
            "metrics_data": "/v1/metrics/data"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)