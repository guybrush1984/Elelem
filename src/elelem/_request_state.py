"""
Request state machine types for Elelem.

Defines explicit states for request processing, making the flow visible and traceable.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List


class RequestState(Enum):
    """Explicit states for request processing.

    The state machine flow:

        CALL_API ──────────────────────────────────────────┐
            │                                              │
            ├─ success ──► EXTRACT_TOKENS                  │
            ├─ timeout ──► NEXT_CANDIDATE                  │
            ├─ rate_limit ──► WAIT_RATE_LIMIT ─────────────┘
            └─ other_error ──► NEXT_CANDIDATE or SKIP_MODEL

        EXTRACT_TOKENS
            │
            ├─ no format ──► SUCCESS
            └─ has format ──► VALIDATE_FORMAT

        VALIDATE_FORMAT
            │
            ├─ valid ──► SUCCESS
            ├─ parse_error ──► NEXT_CANDIDATE (infrastructure)
            └─ schema_error ──► TRY_FIXER

        TRY_FIXER
            │
            ├─ fixed ──► SUCCESS
            ├─ unfixable (truncated) ──► NEXT_CANDIDATE
            └─ fixable_but_failed ──► REDUCE_TEMPERATURE

        REDUCE_TEMPERATURE
            │
            ├─ can_reduce ──► CALL_API
            └─ exhausted ──► SKIP_MODEL

        Terminal states:
            SUCCESS ──► return response
            NEXT_CANDIDATE ──► raise InfrastructureError
            SKIP_MODEL ──► raise ModelError
    """
    # Processing states
    CALL_API = auto()
    EXTRACT_TOKENS = auto()
    VALIDATE_FORMAT = auto()
    TRY_FIXER = auto()
    REDUCE_TEMPERATURE = auto()
    WAIT_RATE_LIMIT = auto()

    # Terminal states
    SUCCESS = auto()
    NEXT_CANDIDATE = auto()  # InfrastructureError - try next provider
    SKIP_MODEL = auto()       # ModelError - skip same model_reference


@dataclass
class RequestContext:
    """All state needed for request processing.

    This context is passed between state handlers, containing both
    immutable request info and mutable processing state.
    """
    # === Immutable request info ===
    request_id: str
    messages: List[Dict]
    original_model: str

    # Format handling
    format_handler: Optional[Any] = None  # OutputFormat instance
    format_schema: Optional[Dict] = None

    # Temperature
    original_temperature: float = 1.0

    # === Candidate info ===
    provider_name: str = ""
    model_name: str = ""
    candidate: Dict = field(default_factory=dict)
    timeout: float = 120.0
    chunk_timeout: Optional[float] = None
    min_tps: Optional[float] = None  # Minimum tokens/sec — abort if too slow (no cooldown)
    min_tps_eval_window: int = 10  # Seconds before evaluating tps
    capabilities: Dict = field(default_factory=dict)

    # API kwargs (mutable - temperature may change)
    api_kwargs: Dict = field(default_factory=dict)

    # Provider client
    provider_client: Any = None

    # Stats model name (for cost tracking)
    stats_model_name: str = ""

    # === Mutable processing state ===
    current_temperature: float = 1.0
    attempt: int = 0
    rate_limit_attempts: int = 0

    # Accumulated tokens across retries
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0

    # Current attempt results
    response: Optional[Any] = None
    content: str = ""
    reasoning_content: Optional[str] = None
    error: Optional[Exception] = None
    error_content: str = ""  # Content that failed validation (for fixer)
    chunk_count: Optional[int] = None

    # === Configuration (from retry_settings) ===
    max_retries: int = 3
    max_rate_limit_retries: int = 3
    temperature_reductions: List[float] = field(default_factory=lambda: [0.2, 0.3])
    min_temp: float = 0.0
    rate_limit_backoff: List[float] = field(default_factory=lambda: [1, 2, 5, 10])

    # === Tracking ===
    request_tracker: Any = None  # RequestTracker instance


@dataclass
class StateTransition:
    """Result of a state handler - determines what happens next.

    Each handler returns a StateTransition indicating:
    - next_state: Which state to transition to
    - error: Optional exception (for terminal error states)
    - result: Optional result (for SUCCESS state)
    - path: List of state names traversed (set by run() at the end)
    """
    next_state: RequestState
    error: Optional[Exception] = None
    result: Optional[Any] = None
    path: List[str] = field(default_factory=list)


# Terminal states that end the state machine
TERMINAL_STATES = frozenset({
    RequestState.SUCCESS,
    RequestState.NEXT_CANDIDATE,
    RequestState.SKIP_MODEL,
})
