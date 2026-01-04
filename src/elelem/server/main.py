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
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import sentry_sdk
from openai import RateLimitError, InternalServerError

from elelem import Elelem
from elelem import __version__
from elelem._exceptions import InfrastructureError, ModelError


def sentry_before_send(event, hint):
    """
    Filter Sentry events to reduce noise from expected errors.

    DROP (expected, don't alert):
    - InfrastructureError: timeouts, provider failures, max_tokens truncation - handled by fallbacks
    - RateLimitError (429): provider rate limits - expected under load
    - InternalServerError (503): provider unavailable - expected during outages
    - HTTPException: duplicates of above (Sentry auto-captures after we do)

    KEEP (real problems, alert):
    - ModelError: JSON validation failures, content filter, unexpected finish_reason
    - Other unexpected exceptions: bugs, config errors, etc.
    """
    if 'exc_info' not in hint:
        return event

    exc_type, exc_value, _ = hint['exc_info']

    # Drop HTTPException - these are duplicates of errors we already capture
    # (we capture the original exception, then raise HTTPException which Sentry auto-captures)
    if exc_type.__name__ == 'HTTPException':
        return None

    # Drop InfrastructureError - provider timeouts/failures are expected
    if exc_type is InfrastructureError:
        return None

    # Drop RateLimitError (429) - expected under load
    if exc_type is RateLimitError:
        return None

    # Drop InternalServerError (503) - provider outages are expected
    if exc_type is InternalServerError:
        return None

    # Keep everything else (ModelError, unexpected exceptions, bugs)
    return event


# Initialize Sentry for error monitoring (must be before FastAPI app creation)
sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.getenv('SENTRY_ENVIRONMENT', 'development'),
        release=f"elelem@{__version__}",
        # Capture 10% of transactions for performance monitoring
        traces_sample_rate=float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', '0.1')),
        # Enable profiling for sampled transactions
        profile_session_sample_rate=float(os.getenv('SENTRY_PROFILE_SAMPLE_RATE', '0.1')),
        # Send default PII (user IPs, etc.) - disable in production if needed
        send_default_pii=False,
        # Filter out expected errors to reduce noise
        before_send=sentry_before_send,
    )
from elelem._benchmark_store import get_routing_stats as get_exploration_stats
from elelem.server.models import (
    ChatCompletionRequest,
    ErrorResponse,
    HealthResponse,
    WarmupRequest,
    WarmupResponse,
)

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Suppress verbose httpx request logging (default WARNING to reduce noise)
httpx_log_level = os.getenv('HTTPX_LOG_LEVEL', 'WARNING').upper()
logging.getLogger('httpx').setLevel(getattr(logging, httpx_log_level, logging.WARNING))

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

    # JSON fixer: LLM-based repair for schema validation failures (enabled by default)
    json_fixer_enabled = os.getenv('ELELEM_JSON_FIXER_ENABLED', 'true').lower() == 'true'
    json_fixer_model = os.getenv('ELELEM_JSON_FIXER_MODEL')  # Uses cerebras 120B by default

    # Initialize Elelem instance (auto-creates PostgreSQL tables if configured)
    elelem = Elelem(
        cache_enabled=cache_enabled,
        cache_ttl=cache_ttl,
        cache_max_size=cache_max_size,
        json_fixer_enabled=json_fixer_enabled,
        json_fixer_model=json_fixer_model
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

    # Log Sentry status
    if sentry_dsn:
        logger.info(f"🔍 Sentry error monitoring enabled (environment: {os.getenv('SENTRY_ENVIRONMENT', 'development')})")
    else:
        logger.info("🔍 Sentry disabled (set SENTRY_DSN to enable)")


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


@app.get("/v1/routing/stats")
async def get_routing_stats():
    """Get detailed dynamic routing statistics.

    Returns:
        - config: Current routing configuration (window, cooldown, samples, epsilon)
        - dynamic_stats: Per-model speed stats from recent requests
        - failed_candidates: Models currently in cooldown after failures
        - exploration: Current exploration state (epsilon, coverage, stats)
    """
    import os
    from datetime import datetime

    # Get dynamic routing store from elelem instance
    routing_store = elelem._dynamic_routing_store

    # Get dynamic stats, sorted by model (after :) then by speed (descending)
    dynamic_stats = routing_store.get_dynamic_stats()
    sorted_stats = sorted(
        dynamic_stats.items(),
        key=lambda x: (x[0].split(':', 1)[1] if ':' in x[0] else x[0], -x[1].avg_tokens_per_sec)
    )
    dynamic_data = {
        model_ref: {
            "avg_tokens_per_sec": round(stats.avg_tokens_per_sec, 1),
            "sample_count": stats.sample_count,
            "last_updated": stats.last_updated.isoformat() if stats.last_updated else None
        }
        for model_ref, stats in sorted_stats
    }

    # Get failed candidates in cooldown
    failed = routing_store.get_failed_candidates()
    failed_with_time = {}
    for candidate, failure_time in routing_store._failed.items():
        remaining = routing_store._cooldown_minutes * 60 - (datetime.utcnow() - failure_time).total_seconds()
        if remaining > 0:
            failed_with_time[candidate] = {
                "failed_at": failure_time.isoformat(),
                "remaining_seconds": round(remaining)
            }

    # Calculate exploration state
    explored_count = len(dynamic_data)

    min_epsilon = float(os.environ.get('ELELEM_EXPLORATION_EPSILON', '0.1'))
    max_epsilon = float(os.environ.get('ELELEM_EXPLORATION_EPSILON_MAX', '1.0'))

    # Get exploration stats from routing module
    exploration_stats = get_exploration_stats()

    return {
        "config": {
            "window_minutes": routing_store._window_minutes,
            "cooldown_minutes": routing_store._cooldown_minutes,
            "max_samples": routing_store._max_samples,
            "cache_ttl_seconds": routing_store._cache_ttl,
            "min_epsilon": min_epsilon,
            "max_epsilon": max_epsilon,
        },
        "exploration": {
            "total_routing_decisions": exploration_stats["total"],
            "exploration_count": exploration_stats["explorations"],
            "exploration_rate": exploration_stats["exploration_rate"],
            "explored_models": explored_count,
        },
        "dynamic_stats": dynamic_data,
        "failed_candidates": failed_with_time,
    }


async def _run_warmup(candidates: list, prompt: str, parallel: bool):
    """Background task to call all candidates for warmup."""
    import asyncio

    async def call_one(c):
        try:
            await elelem.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=c.get('original_model_ref', c.get('model')),
                cache=False, tags=["warmup"]
            )
            logger.info(f"🔥 Warmup OK: {c.get('provider')}:{c.get('model_id')}")
        except Exception as e:
            logger.warning(f"🔥 Warmup FAIL: {c.get('provider')}:{c.get('model_id')} - {e}")

    if parallel:
        await asyncio.gather(*[call_one(c) for c in candidates])
    else:
        for c in candidates:
            await call_one(c)

    elelem._dynamic_routing_store.invalidate_cache()
    logger.info(f"🔥 Warmup complete: {len(candidates)} candidates tested")


@app.post("/v1/routing/warmup", response_model=WarmupResponse)
async def warmup_routing(request: WarmupRequest):
    """Warmup routing by calling each candidate directly to populate metrics.

    Returns immediately after validating request. Warmup runs in background.
    Check /v1/routing/stats to see results as they populate.
    """
    import asyncio

    # Validate models and collect candidates
    all_candidates = []
    errors = []

    for virtual_model in request.models:
        try:
            candidates = elelem.config.get_model_config(virtual_model).get('candidates', [])
            available = [c for c in candidates if elelem._ensure_provider_initialized(c.get('provider'))]
            all_candidates.extend(available)
        except ValueError as e:
            errors.append(f"{virtual_model}: {e}")

    if errors and not all_candidates:
        raise HTTPException(status_code=400, detail={"errors": errors})

    prompt = request.prompt or f"It's {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Write a poem on LLMs in about 100 words."

    asyncio.create_task(_run_warmup(all_candidates, prompt, request.parallel))

    return WarmupResponse(
        status="started",
        models=request.models,
        total_candidates=len(all_candidates),
        message=f"Warmup started for {len(all_candidates)} candidates. Check /v1/routing/stats for results."
    )


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
        # Handle other errors with Sentry context
        logger.error(f"Error in chat completion: {e}")
        logger.error(traceback.format_exc())

        # Extract provider from exception if available (ModelError/InfrastructureError may have it)
        provider = getattr(e, 'provider', None)
        request_tags = request_dict.get('tags', [])

        # Add Sentry context for better debugging
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("model", model)
            scope.set_tag("error_type", type(e).__name__)
            if provider:
                scope.set_tag("provider", provider)
            if request_tags:
                for tag in request_tags:
                    scope.set_tag(f"request_tag:{tag}", "true")
            scope.set_context("request", {
                "model": model,
                "provider": provider,
                "tags": request_tags,
                "message_count": len(messages_dict),
                "first_message_role": messages_dict[0].get("role") if messages_dict else None,
                "extra_params": list(request_dict.keys()),
            })
            sentry_sdk.capture_exception(e)

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
    import time
    request_start = time.time()
    logger.info(f"📊 /v1/metrics/summary called - tags={tags}, start_time={start_time}, end_time={end_time}")
    try:
        tag_list = tags.split(',') if tags else None
        summary = elelem.get_summary(start_time, end_time, tag_list)
        duration = time.time() - request_start
        logger.info(f"📊 /v1/metrics/summary completed in {duration:.3f}s")
        return summary
    except Exception as e:
        duration = time.time() - request_start
        logger.error(f"📊 /v1/metrics/summary failed after {duration:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/metrics/tags")
async def get_metrics_tags():
    """Get all unique tags from metrics data.

    Returns list of tags including automatic tags like:
    - model:groq:openai/gpt-oss-20b
    - provider:groq
    - Any user-defined tags
    """
    import time
    request_start = time.time()
    logger.info("📊 /v1/metrics/tags called")
    try:
        tags = elelem.get_metrics_tags()
        duration = time.time() - request_start
        logger.info(f"📊 /v1/metrics/tags completed in {duration:.3f}s - {len(tags)} tags")
        return {"tags": tags}
    except Exception as e:
        duration = time.time() - request_start
        logger.error(f"📊 /v1/metrics/tags failed after {duration:.3f}s: {e}")
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
    """
    import time
    request_start = time.time()
    logger.info(f"📊 /v1/metrics/data called - tags={tags}, start_time={start_time}, end_time={end_time}")
    try:
        if format != "json":
            raise HTTPException(status_code=400, detail="Only 'json' format is currently supported")

        tag_list = tags.split(',') if tags else None
        data = elelem.get_metrics_data(start_time, end_time, tag_list)

        # Handle datetime serialization
        for row in data:
            if 'timestamp' in row and row['timestamp']:
                ts = row['timestamp']
                if hasattr(ts, 'strftime'):
                    row['timestamp'] = ts.strftime('%Y-%m-%dT%H:%M:%S.%f')

        duration = time.time() - request_start
        logger.info(f"📊 /v1/metrics/data completed in {duration:.3f}s - {len(data)} rows")
        return data
    except Exception as e:
        duration = time.time() - request_start
        logger.error(f"📊 /v1/metrics/data failed after {duration:.3f}s: {e}")
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
            "routing_stats": "/v1/routing/stats",
            "routing_warmup": "/v1/routing/warmup",
            "metrics_summary": "/v1/metrics/summary",
            "metrics_data": "/v1/metrics/data"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)