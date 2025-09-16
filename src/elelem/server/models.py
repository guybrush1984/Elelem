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
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
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


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""
    error: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str