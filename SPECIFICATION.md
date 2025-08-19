# Elelem Library Specification

## Overview
Elelem is a unified wrapper library for OpenAI, GROQ, and DeepInfra APIs, specifically targeted for JSON output generation with comprehensive cost tracking, retry logic, and error handling. **MUST BE INTEGRABLE IN FABLE WITHOUT ANY FABLE MODIFICATIONS.**

## Directory Structure
```
elelem/
├── SPECIFICATION.md     # This specification document
├── __init__.py          # Package initialization, exports Elelem class
├── elelem.py           # Main Elelem class with unified API wrapper
├── providers.py        # Provider-specific implementations (OpenAI, GROQ, DeepInfra)
├── models.py           # Model definitions with capabilities and pricing
├── config.json         # Built-in configuration file loaded at init
└── simple_test.py      # Test script for validation
```

## Core Elelem Class Interface

### Constructor
```python
class Elelem:
    def __init__(self):
        # Load config.json at initialization
        # Initialize cost tracking, provider clients
        # Load model definitions from models.py
        # No on-the-fly reconfiguration supported
        # NO default provider or model - model MUST be passed every time
        pass
```

### Main API Method
```python
async def create_chat_completion(
    self,
    messages: List[Dict[str, str]],
    model: str,  # REQUIRED - Format: "provider:model" (e.g., "openai:gpt-4.1-mini")
    response_format: Optional[Dict] = None,  # {"type": "json_object"} - AUTO-REMOVED for deepinfra
    tags: Union[str, List[str]] = [],  # For cost tracking by category
    **kwargs  # Additional parameters (temperature, etc.)
) -> Dict:  # Returns OpenAI-compatible response with .choices[0]["message"]["content"]
    """
    Unified chat completion method with:
    - JSON validation and retry logic
    - Rate limit handling with exponential backoff
    - Cost tracking by tags
    - Provider-specific handling
    - Auto-detection and removal of <think> tags from responses
    - Auto-removal of response_format for DeepInfra (they don't support it)
    """
    pass
```

### Statistics Methods (MUST MATCH FABLE EXPECTATIONS EXACTLY)
```python
def get_stats(self) -> Dict[str, Any]:
    """Returns overall statistics across all requests - EXACT FABLE FORMAT"""
    return {
        "total_input_tokens": int,
        "total_output_tokens": int,
        "total_tokens": int,  # input + output
        "total_input_cost_usd": float,
        "total_output_cost_usd": float,
        "total_cost_usd": float,  # input + output costs
        "total_calls": int,
        "total_duration_seconds": float,  # Sum of all request durations
        "avg_duration_seconds": float,  # total_duration / total_calls
        # Optional reasoning tokens for o3/reasoning models
        "reasoning_tokens": int,  # Only if reasoning tokens used
        "reasoning_cost_usd": float  # Only if reasoning tokens used
    }
    
def get_stats_by_tag(self, tag: str) -> Dict[str, Any]:
    """Returns statistics filtered by specific tag - EXACT FABLE FORMAT"""
    # Same structure as get_stats() but filtered by tag
    pass
```

## Supported Models - Structured with .capabilities and .cost

### Models Dictionary Structure
```python
MODELS = {
    # OpenAI Models
    "openai:gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 4096,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 15.0,  # $15 per 1M input tokens
            "output_cost_per_1m": 60.0,  # $60 per 1M output tokens
            "currency": "USD"
        }
    },
    "openai:gpt-4.1-mini": {
        "provider": "openai",
        "model_id": "gpt-4.1-mini",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 4096,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 0.15,  # $0.15 per 1M input tokens
            "output_cost_per_1m": 0.60,  # $0.60 per 1M output tokens
            "currency": "USD"
        }
    },
    "openai:gpt-5": {
        "provider": "openai",
        "model_id": "gpt-5",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 4096,
            "context_window": 200000
        },
        "cost": {
            "input_cost_per_1m": 30.0,
            "output_cost_per_1m": 120.0,
            "currency": "USD"
        }
    },
    "openai:gpt-5-mini": {
        "provider": "openai",
        "model_id": "gpt-5-mini",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 4096,
            "context_window": 200000
        },
        "cost": {
            "input_cost_per_1m": 3.0,
            "output_cost_per_1m": 12.0,
            "currency": "USD"
        }
    },
    "openai:o3": {
        "provider": "openai",
        "model_id": "o3",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": False,  # o-series models don't support temperature
            "supports_reasoning": True,
            "max_tokens": 4096,
            "context_window": 200000
        },
        "cost": {
            "input_cost_per_1m": 60.0,
            "output_cost_per_1m": 240.0,
            "reasoning_cost_per_1m": 240.0,  # Reasoning tokens cost
            "currency": "USD"
        }
    },
    "openai:o3-mini": {
        "provider": "openai",
        "model_id": "o3-mini",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": False,
            "supports_reasoning": True,
            "max_tokens": 4096,
            "context_window": 200000
        },
        "cost": {
            "input_cost_per_1m": 3.0,
            "output_cost_per_1m": 12.0,
            "reasoning_cost_per_1m": 12.0,
            "currency": "USD"
        }
    },
    
    # GROQ Models
    "groq:gpt-oss-120b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",  # GROQ's model ID format
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 32768
        },
        "cost": {
            "input_cost_per_1m": 0.15,
            "output_cost_per_1m": 0.75,
            "currency": "USD"
        }
    },
    "groq:gpt-oss-20b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 32768
        },
        "cost": {
            "input_cost_per_1m": 0.08,
            "output_cost_per_1m": 0.40,
            "currency": "USD"
        }
    },
    "groq:moonshotai/kimi-k2-instruct": {
        "provider": "groq",
        "model_id": "moonshotai/kimi-k2-instruct",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 4096,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 0.0,  # Free tier
            "output_cost_per_1m": 0.0,
            "currency": "USD"
        }
    },
    "groq:meta-llama/llama-4-maverick-17b-128e-instruct": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 0.05,
            "output_cost_per_1m": 0.08,
            "currency": "USD"
        }
    },
    "groq:meta-llama/llama-4-scout-17b-16e-instruct": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "capabilities": {
            "supports_json_mode": True,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 0.05,
            "output_cost_per_1m": 0.08,
            "currency": "USD"
        }
    },
    
    # DeepInfra Models - CRITICAL: ALL supports_json_mode = False
    "deepinfra:gpt-oss-120b": {
        "provider": "deepinfra",
        "model_id": "openai/gpt-oss-120b",
        "capabilities": {
            "supports_json_mode": False,  # DeepInfra doesn't support response_format
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 32768
        },
        "cost": {
            "input_cost_per_1m": 0.30,
            "output_cost_per_1m": 0.30,
            "currency": "USD"
        }
    },
    "deepinfra:gpt-oss-20b": {
        "provider": "deepinfra",
        "model_id": "openai/gpt-oss-20b",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 32768
        },
        "cost": {
            "input_cost_per_1m": 0.13,
            "output_cost_per_1m": 0.13,
            "currency": "USD"
        }
    },
    "deepinfra:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "provider": "deepinfra",
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 0.07,
            "output_cost_per_1m": 0.07,
            "currency": "USD"
        }
    },
    "deepinfra:meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "provider": "deepinfra",
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 0.07,
            "output_cost_per_1m": 0.07,
            "currency": "USD"
        }
    },
    "deepinfra:moonshotai/Kimi-K2-Instruct": {
        "provider": "deepinfra",
        "model_id": "moonshotai/Kimi-K2-Instruct",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True,
            "supports_reasoning": False,
            "max_tokens": 4096,
            "context_window": 128000
        },
        "cost": {
            "input_cost_per_1m": 0.07,
            "output_cost_per_1m": 0.07,
            "currency": "USD"
        }
    },
    "deepinfra:deepseek-ai/DeepSeek-R1-0528": {
        "provider": "deepinfra",
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "capabilities": {
            "supports_json_mode": False,
            "supports_temperature": True,
            "supports_reasoning": True,  # Reasoning model - has <think> tags
            "max_tokens": 4096,
            "context_window": 64000
        },
        "cost": {
            "input_cost_per_1m": 0.14,
            "output_cost_per_1m": 0.28,
            "reasoning_cost_per_1m": 0.28,  # Reasoning tokens cost (if tracked separately)
            "currency": "USD"
        }
    }
}
```

## Configuration Structure (config.json)

### Built-in Configuration File
```json
{
    "retry_settings": {
        "max_json_retries": 3,
        "temperature_step": 0.1,
        "rate_limit_backoff": [1, 2, 4, 8],
        "max_rate_limit_retries": 4
    },
    "provider_endpoints": {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1", 
        "deepinfra": "https://api.deepinfra.com/v1/openai"
    },
    "timeout_seconds": 120,
    "logging_level": "INFO"
}
```

## Key Features

### 1. Model Validation and Routing
- Model MUST be passed on every call (no defaults)
- Parse "provider:model" strings (e.g., "openai:gpt-4.1-mini")
- Lookup model in MODELS dict for capabilities and pricing
- Route to appropriate provider client
- Return clear error for unknown models

### 2. DeepInfra Special Handling
- **AUTO-REMOVE response_format parameter** for all deepinfra models
- DeepInfra doesn't support response_format (causes HTTP 500 errors)
- Models work perfectly with explicit JSON prompting
- Check model["capabilities"]["supports_json_mode"] = False for deepinfra

### 3. Think Tag Removal
- **Auto-detect and remove `<think>` tags** from all responses
- Some reasoning models (like DeepSeek-R1) include thinking in `<think></think>` tags
- Remove these tags before returning response to caller
- Pattern: Remove everything between `<think>` and `</think>` (including the tags)
- Apply to all responses regardless of provider

### 4. JSON Validation & Retry Logic
- Validate response is valid JSON after each request (when response_format is used)
- If invalid JSON:
  1. Reduce temperature by config.retry_settings.temperature_step (0.1)
  2. Retry up to config.retry_settings.max_json_retries times (3)
  3. If still failing, return error (no fallback model - model must be explicit)
- Track retry attempts in statistics
- Preserve original request parameters except temperature

### 5. Rate Limit Handling (429 Errors)
- Detect 429 (Too Many Requests) HTTP status codes
- Implement exponential backoff using config.retry_settings.rate_limit_backoff
- Wait periods: [1, 2, 4, 8] seconds by default
- Max retries from config.retry_settings.max_rate_limit_retries (4)
- Track retry timing and counts in statistics

### 6. Cost Calculation System - EXACT FABLE COMPATIBILITY
- Use MODELS dict to calculate precise costs
- Calculate based on actual input/output/reasoning tokens from API responses
- Must return EXACT structure expected by Fable:
  - `total_input_tokens`, `total_output_tokens`, `total_tokens`
  - `total_input_cost_usd`, `total_output_cost_usd`, `total_cost_usd`
  - `total_calls`, `total_duration_seconds`, `avg_duration_seconds`
  - Optional: `reasoning_tokens`, `reasoning_cost_usd` for o3/reasoning models

### 7. OpenAI-Compatible Response Format
- Return response in OpenAI format with `.choices[0]["message"]["content"]`
- Maintain compatibility with existing Fable code patterns:
  ```python
  content = response.choices[0]["message"]["content"]
  ```
- Handle provider-specific response formats internally

## Expected Usage Pattern (from Fable analysis)

```python
# From story_engine.py - model always passed explicitly
response = await self.elelem.create_chat_completion(
    messages=messages,
    model=self._get_model_string("story_params"),  # e.g., "groq:gpt-oss-120b"
    response_format={"type": "json_object"},  # Auto-removed for deepinfra
    tags=["story_params"],
    **params  # temperature, etc.
)

content = response.choices[0]["message"]["content"]  # MUST WORK EXACTLY LIKE THIS

# Statistics usage in bundle_generator.py
stats = elelem.get_stats_by_tag("story_params")
total_cost = stats["total_cost_usd"]  # MUST EXIST
input_tokens = stats["total_input_tokens"]  # MUST EXIST
```

## Provider-Specific Implementations

### OpenAI Provider
- Use official OpenAI client library
- Support `response_format={"type": "json_object"}`
- Handle OpenAI-specific error codes and rate limits
- Support reasoning tokens for o3/o3-mini models
- Extract reasoning_tokens from usage if present

### GROQ Provider
- Use GROQ API with OpenAI-compatible interface
- Support `response_format={"type": "json_object"}`
- Handle GROQ rate limits and error handling
- Use GROQ-specific model IDs (e.g., "openai/gpt-oss-120b")

### DeepInfra Provider
- Use DeepInfra API with OpenAI-compatible interface
- **CRITICAL: Auto-remove response_format parameter** (causes HTTP 500 errors)
- Handle DeepInfra rate limits and pricing
- Use DeepInfra-specific model IDs
- Auto-remove `<think>` tags from reasoning model responses (DeepSeek-R1)

## Tags Used in Fable
- "story_params" - For story parameter generation
- "blurb" - For story blurb generation  
- "first_chapter" - For first chapter generation
- "story" - For main story content generation
- "ambiance_matching" - For audio ambiance matching
- "character_extraction" - For character extraction
- ["ambiance_matching", "fable"] - Multiple tags supported

## API Keys
- Load from environment variables: OPENAI_API_KEY, GROQ_API_KEY, DEEPINFRA_API_KEY
- No API keys stored in config.json
- Clear error messages when API keys are missing

## Fable Integration Requirements

### ZERO MODIFICATIONS TO FABLE
- Elelem must be drop-in replacement for current system
- All existing calls must work without changes
- Statistics structure must match exactly
- Response format must be OpenAI-compatible
- Error handling must be graceful

### Critical Compatibility Points
1. `response.choices[0]["message"]["content"]` access pattern
2. `get_stats()` and `get_stats_by_tag()` return exact expected structure
3. All model strings used in Fable must be supported
4. All existing parameter combinations must work
5. Async/await pattern maintained

## Success Criteria

- **All existing Fable integrations work without modification**
- Model must be passed explicitly on every call
- DeepInfra response_format auto-removal works perfectly
- Think tag removal works for reasoning models
- Cost tracking matches Fable expectations exactly
- JSON validation and retry logic works
- Rate limit handling prevents failures
- Statistics provide exact structure Fable expects
- Drop-in replacement for current elelem system