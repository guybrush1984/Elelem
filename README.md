# Elelem - Multi-Provider LLM Gateway

Elelem is a Python library and OpenAI-compatible server that routes LLM requests across multiple providers with automatic failover. It solves the practical problem of provider reliability: when one provider has an outage or rate limit, Elelem automatically tries the next one.

**Key capabilities:**
- **Automatic failover** across 8 providers (OpenAI, Groq, Fireworks, DeepInfra, Parasail, Scaleway, OpenRouter, DeepSeek)
- **JSON reliability** with automatic retries, schema validation, and error correction
- **Cost tracking** with per-request metrics, reasoning token extraction, and tag-based analytics
- **Two deployment modes:** Python library or Docker server with OpenAI-compatible API

## Quick Start

### Library Mode

```python
import asyncio
from elelem import Elelem

async def main():
    elelem = Elelem()

    # Basic request
    response = await elelem.create_chat_completion(
        model="groq:openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tags=["experiment:v1", "category:math"]  # Tags are key:value pairs
    )

    print(response.choices[0].message.content)
    # "2+2 equals 4."

    # Get metrics
    stats = elelem.get_stats_by_tag("category:math")
    print(f"Cost: ${stats['costs']['total']:.6f}")
    print(f"Tokens: {stats['tokens']['total']['total']}")

asyncio.run(main())
```

### Server Mode

```bash
# Set your API keys (same for library and server mode)
export GROQ_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Start server
docker-compose -f src/elelem/server/docker-compose.yml up -d
```

Use with OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="anything"  # Not validated in local mode
)

response = client.chat.completions.create(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "tags": ["category:test", "user:123"],  # Tags are key:value pairs
        "json_schema": {...}  # Optional: Elelem-specific validation
    }
)

print(response.choices[0].message.content)
```

**Server endpoints:**
- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)
- `GET /v1/models` - List available models
- `GET /v1/metrics/summary` - Aggregated metrics (optional tags filter)
- `GET /v1/metrics/data` - Raw metrics data
- `GET /v1/metrics/tags` - Available tags
- `GET /health` - Health check

## Response Structure

### Standard OpenAI Fields

```python
response.choices[0].message.content  # Response text
response.choices[0].message.reasoning  # Reasoning content (for o3, DeepSeek, etc.)
response.usage.prompt_tokens          # Input tokens
response.usage.completion_tokens      # Output tokens (including reasoning)
response.usage.total_tokens           # Total tokens
```

### Elelem Extensions

The `response.elelem_metrics` dict contains additional tracking:

```python
{
    "cost_usd": 0.000123,              # Total cost in USD
    "reasoning_tokens": 45,             # Reasoning tokens (subset of completion_tokens)
    "reasoning_content": "thinking...", # Reasoning text (if available)
    "total_duration_seconds": 1.23,     # Request duration
    "actual_provider": "groq",          # Provider that served the request
    "actual_model": "openai/gpt-oss-120b",
    "candidate_iterations": 0,          # Number of provider failovers
    "temperature_reductions": 0,        # JSON retry attempts
    "rate_limit_retries": 0,           # Rate limit backoff count
    "total_retry_attempts": 0          # All retry attempts
}
```

## Core Features

### 1. Candidates & Automatic Failover

Elelem supports three model types:

**Direct models:** `provider:model-name`
```python
model="groq:openai/gpt-oss-120b"
model="openai:gpt-4.1"
```

**Virtual models:** Pre-configured failover chains in `virtual-models.yaml`
```yaml
# virtual-models.yaml
models:
  "virtual:gpt-oss-120b-reliable":
    candidates:
      - model: "groq:openai/gpt-oss-120b"
        timeout: 10
      - model: "fireworks:openai/gpt-oss-120b"
        timeout: 15
      - model: "deepinfra:openai/gpt-oss-120b"
        timeout: 30
```

```python
# Automatically tries Groq → Fireworks → DeepInfra
model="virtual:gpt-oss-120b-reliable"
```

**Dynamic models:** Runtime failover definition
```python
model="dynamic:{candidates: [groq:openai/gpt-oss-120b, openai:gpt-4.1], timeout: 30}"
```

**What triggers candidate iteration (tries next provider):**
- Timeouts
- Connection errors (SSL, network)
- HTTP 500, 502, 503 (server errors)
- HTTP 400 (bad request - might work with another provider)
- HTTP 401, 403 (auth/permission - your key might work elsewhere)
- HTTP 404 (model not found - might exist on another provider)
- HTTP 429 (rate limit, after exhausting retries)

**What causes immediate failure (no iteration):**
- HTTP 409, 422 (conflict/unprocessable - request validation errors)
- JSON validation failures (after all temperature reduction attempts)
- Content filtering/safety violations (finish_reason: content_filter)
- Response truncation due to max_tokens (finish_reason: length)
- All candidates exhausted

### 2. JSON Mode & Schema Validation

Elelem provides robust JSON handling with automatic retry strategies:

```python
response = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Generate user data"}],
    response_format={"type": "json_object"},
    json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    },
    temperature=1.0
)
```

**Retry strategy on JSON errors:**
1. **Parse error:** Reduce temperature by 0.2, retry (up to 3 times)
2. **Still failing:** Remove `response_format`, retry
3. **Still failing:** Try next candidate

**JSON processing:**
- Strips markdown code blocks (```json ... ```)
- Fixes common errors (trailing commas, single quotes)
- Validates against schema if provided
- Works even with models that don't support native JSON mode

### 3. Metrics & Cost Tracking

Every request is stored in SQLite (local) or PostgreSQL (production) with comprehensive metrics:

```python
# Tag your requests with key:value pairs
response = await elelem.create_chat_completion(
    model="openai:gpt-4.1",
    messages=[{"role": "user", "content": "Analyze data"}],
    tags=["env:production", "user:123", "feature:analysis"]
)

# Get aggregated stats by tag
stats = elelem.get_stats_by_tag("env:production")
print(stats)
```

**Stats structure:**
```python
{
    "requests": {"total": 150, "successful": 148, "failed": 2, "success_rate": 0.987},
    "tokens": {
        "input": {"total": 45000, "avg": 300, "min": 50, "max": 1000},
        "output": {"total": 30000, "avg": 200, "min": 20, "max": 800},
        "reasoning": {"total": 5000, "avg": 33.3, "min": 0, "max": 200}
    },
    "costs": {"total": 0.245, "avg": 0.00163, "min": 0.0001, "max": 0.01},
    "duration": {"total": 180.5, "avg": 1.2, "min": 0.3, "max": 5.2},
    "providers": {"groq": 0.05, "openai": 0.195},
    "models": {"groq:openai/gpt-oss-120b": 0.05, "openai:gpt-4.1": 0.195},
    "retries": {
        "json_parse_retries": 5,
        "rate_limit_retries": 2,
        "candidate_iterations": 3,
        "total_retry_attempts": 10
    }
}
```

**Access raw data as pandas DataFrame:**
```python
df = elelem.get_metrics_dataframe(tags=["env:production"])
# Returns pandas DataFrame with all request details

# Filter by time and tags
from datetime import datetime, timedelta
start = datetime.now() - timedelta(hours=24)
df = elelem.get_metrics_dataframe(start_time=start, tags=["user:123"])
```

**Server mode metrics:**
```bash
# Get summary stats
curl "http://localhost:8000/v1/metrics/summary"

# Filter by tags (AND logic - must have ALL tags)
curl "http://localhost:8000/v1/metrics/summary?tags=env:production,user:123"

# Get raw data
curl "http://localhost:8000/v1/metrics/data?tags=env:production"

# List available tags
curl "http://localhost:8000/v1/metrics/tags"
```

### 4. Reasoning Token Extraction

Elelem automatically extracts reasoning tokens from different provider formats:

**OpenAI o3/o3-mini:**
```python
response = await elelem.create_chat_completion(
    model="openai:o3-mini",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print(response.choices[0].message.reasoning)  # Reasoning text
print(response.usage.completion_tokens)        # Includes reasoning tokens
print(response.elelem_metrics["reasoning_tokens"])  # Reasoning token count
```

**Groq/Fireworks DeepSeek:**
- Extracts reasoning tokens from `usage.completion_tokens_details`
- Removes `<think>...</think>` tags from response content
- Estimates reasoning tokens from character count when not provided

**DeepSeek reasoning modes:**
```python
# Standard mode
model="parasail:deepseek-3.1"

# Thinking mode (automatic <think> tag removal)
model="parasail:deepseek-3.1-think"

# Parameterized reasoning
model="groq:deepseek/deepseek-r1-distill-qwen-32b?reasoning=medium"
model="fireworks:deepseek/deepseek-3.1?reasoning=low"
```

## Installation

```bash
# Clone repository
git clone https://github.com/guybrush1984/Elelem.git
cd Elelem

# Install with pip
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

### Environment Variables

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export FIREWORKS_API_KEY="your-key"
export DEEPINFRA_API_KEY="your-key"
export PARASAIL_API_KEY="your-key"
export SCALEWAY_ACCESS_KEY="your-access-key"
export SCALEWAY_SECRET_KEY="your-secret-key"
export OPENROUTER_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```

### Docker Deployment

For production, use PostgreSQL for metrics:

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:17-alpine
    environment:
      POSTGRES_DB: elelem
      POSTGRES_USER: elelem
      POSTGRES_PASSWORD: your-password
    volumes:
      - postgres-data:/var/lib/postgresql/data

  elelem:
    image: your-registry/elelem:latest
    environment:
      - ELELEM_DATABASE_URL=postgresql://elelem:your-password@postgres:5432/elelem
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - postgres
```

## Configuration

### Provider Configuration

Providers and models are defined in YAML files under `src/elelem/providers/`:

```yaml
# src/elelem/providers/groq.yaml
provider:
  endpoint: https://api.groq.com/openai/v1

models:
  "groq:openai/gpt-oss-120b":
    metadata_ref: "gpt-oss-120b"  # References src/elelem/providers/_metadata.yaml
    provider: groq
    model_id: "openai/gpt-oss-120b"
    capabilities:
      supports_json_mode: true
      supports_temperature: true
      supports_system: true
    cost:
      input_cost_per_1m: 0.05
      output_cost_per_1m: 0.15
      currency: USD
```

### Virtual Models

Define failover chains in `virtual-models.yaml` (project root):

```yaml
models:
  "virtual:fast-and-reliable":
    candidates:
      - model: "groq:openai/gpt-oss-120b"
        timeout: 10
      - model: "openai:gpt-4.1-mini"
        timeout: 20
```

### Dynamic Models

Create failover chains at runtime:

```python
# Simple list
model="dynamic:[groq:openai/gpt-oss-120b, openai:gpt-4.1]"

# With timeout
model="dynamic:{candidates: [groq:openai/gpt-oss-120b, openai:gpt-4.1], timeout: 30}"

# With individual candidate timeouts
model="dynamic:{candidates: [{model: groq:openai/gpt-oss-120b, timeout: 10}, {model: openai:gpt-4.1, timeout: 20}]}"
```

## Supported Providers & Models

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | gpt-4.1, gpt-4.1-mini, o3, o3-mini | Native reasoning support |
| **Groq** | gpt-oss-120b, gpt-oss-20b, llama-4, kimi-k2, deepseek-r1 | Fast inference |
| **Fireworks** | deepseek-3.1, qwen-coder, llama-3.3 | Reasoning parameters |
| **DeepInfra** | gpt-oss-120b, deepseek-3.1, llama-3.1 | Cost-effective |
| **Parasail** | deepseek-3.1, deepseek-3.1-think, gpt-oss-120b | Thinking mode |
| **Scaleway** | gpt-oss-120b, gemma-3, mistral-small | EU-based |
| **OpenRouter** | All OpenRouter models | Meta-provider |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | Direct access |

Full model list: Run `elelem.list_models()` or `curl http://localhost:8000/v1/models`

## Development

### Running Tests

```bash
# All tests
uv run pytest

# Specific test categories
uv run pytest tests/test_config_validation.py    # Config validation
uv run pytest tests/test_elelem_with_faker.py    # No API keys needed
uv run pytest tests/test_real_providers.py       # Requires API keys

# With coverage
uv run pytest --cov=elelem --cov-report=html
```

### Adding Providers

1. Create `src/elelem/providers/yourprovider.yaml`
2. Define models with `metadata_ref` to `_metadata.yaml`
3. Add API key environment variable
4. Run tests: `uv run pytest tests/test_config_validation.py`

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run test suite: `uv run pytest`
5. Submit a pull request

## Support

- Issues: [GitHub Issues](https://github.com/guybrush1984/Elelem/issues)
- Documentation: [SPECIFICATION.md](SPECIFICATION.md)
