"""
Pydantic models for OpenAI API compatibility.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request model."""
    messages: List[Message]
    model: str
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None

    # Elelem-specific parameters
    tags: Optional[Union[str, List[str]]] = None
    json_schema: Optional[Dict[str, Any]] = None
    yaml_schema: Optional[Dict[str, Any]] = None
    csv_schema: Optional[Dict[str, Any]] = None  # Multi-table CSV schema {"tables": {"name": {"columns": {...}}}}
    cache: Optional[bool] = True  # Enable cache by default, set False to bypass
    enforce_schema_in_prompt: Optional[bool] = False  # Force schema injection (default False saves tokens)
    min_tps: Optional[float] = None  # Minimum tokens/sec — abort and try next candidate if too slow


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""
    error: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


class WarmupRequest(BaseModel):
    """Request to warmup routing statistics for virtual models."""
    models: List[str] = Field(..., description="List of virtual model names to warmup")
    prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt to use. Defaults to a ~100 word poem request with current datetime."
    )
    parallel: Optional[bool] = Field(
        default=True,
        description="Run candidates in parallel (default: true)"
    )


class WarmupCandidateResult(BaseModel):
    """Result for a single candidate in warmup."""
    provider: str
    model: str
    tokens_per_sec: float
    output_tokens: int
    duration_seconds: float


class WarmupCandidateFailure(BaseModel):
    """Failure info for a candidate in warmup."""
    provider: str
    model: str
    error: str


class WarmupModelResult(BaseModel):
    """Warmup result for a single virtual model."""
    candidates_tested: int
    succeeded: List[WarmupCandidateResult]  # Ordered by speed (fastest first)
    failed: List[WarmupCandidateFailure]


class WarmupResponse(BaseModel):
    """Response from warmup endpoint (async - returns immediately)."""
    status: str  # "started" or "error"
    models: List[str]  # Models being warmed up
    total_candidates: int  # Total candidates to test
    message: str  # Human-readable status