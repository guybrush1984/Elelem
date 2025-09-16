"""
Elelem OpenAI API Compatibility Server

A FastAPI server that provides OpenAI-compatible endpoints using Elelem's
multi-provider backend for resilient AI API access.

Usage:
    uvicorn elelem.server.main:app --host 0.0.0.0 --port 8000
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from elelem.elelem import Elelem
from elelem import __version__
from elelem.server.models import (
    ChatCompletionRequest,
    ErrorResponse,
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Elelem OpenAI API Server",
    description="OpenAI-compatible API server with multi-provider backend",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize Elelem instance
elelem = Elelem()


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
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=__version__)


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
    """Get raw metrics data as JSON array.

    Returns array of call records with all fields:
    - timestamp, model, provider, tags
    - Token counts, costs, duration
    - call_id for correlation
    """
    try:
        if format != "json":
            raise HTTPException(status_code=400, detail="Only 'json' format is currently supported")

        tag_list = tags.split(',') if tags else None
        df = elelem.get_metrics_dataframe(start_time, end_time, tag_list)

        # Convert DataFrame to JSON records
        # Handle datetime serialization
        df_copy = df.copy()
        if 'timestamp' in df_copy.columns:
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