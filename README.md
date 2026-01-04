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
# Automatically tries providers in optimal order (see Smart Routing below)
model="virtual:gpt-oss-120b-reliable"
```

**Smart Routing:** Virtual models automatically reorder candidates based on observed performance and cost:

1. **Dynamic observations**: Real performance stats from recent requests (last 5 per provider, 4-hour window)
2. **Value score**: `value = tps^speed_weight / cost_per_1m` - balances speed vs cost
3. **Adaptive exploration**: 100% shuffle at cold start → 10% at steady state (scales with data coverage)
4. **Failure cooldown**: Failed providers are excluded for 15 minutes

**Value Score Examples** (real provider pricing for GPT-OSS-120B):

| Provider | Cost ($/M) | Observed Speed | speed_weight=0.5 | speed_weight=1.0 | speed_weight=1.5 |
|----------|------------|----------------|------------------|------------------|------------------|
| novita   | $0.25      | 50 t/s         | **28** ⭐        | 200              | 1,414            |
| fireworks| $0.60      | 120 t/s        | 18               | 200              | 2,191            |
| cerebras | $0.75      | 400 t/s        | 27               | **533** ⭐       | **10,667** ⭐    |

- `speed_weight=0.5` → novita wins (prioritize cost savings)
- `speed_weight=1.0` → cerebras wins (balanced speed/cost)
- `speed_weight=1.5` (default) → cerebras wins decisively (prioritize speed)

Example log showing routing decision:
```
🚀 virtual:gpt-oss-120b → [cerebras(3x, 400t/s, 10667v), fireworks(2x, 120t/s, 2191v), novita(5x, 50t/s, 1414v)]
```
- `3x` = 3 samples observed
- `400t/s` = average tokens/sec
- `10667v` = value score (higher = better)

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

### 2. Output Formats: JSON, YAML, CSV

Elelem supports three structured output formats with schema validation, auto-repair, and LLM-based error correction.

#### JSON Format

```python
response = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Generate user data"}],
    response_format={"type": "json_object"},
    json_schema={
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"]
    }
)
```

#### YAML Format

```python
response = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Generate a story outline"}],
    yaml_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "chapters": {"type": "array", "items": {"type": "object"}}
        }
    }
)
```

#### CSV Format (Multi-Table)

```python
response = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Extract characters and locations"}],
    csv_schema={
        "tables": {
            "characters": {
                "required": True,
                "columns": {
                    "id": {"type": "string", "required": True},
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["hero", "villain"]}
                }
            }
        }
    }
)
# Output: ###TABLE:characters\nid;name;role\nc1;Aragorn;hero
```

**Error handling strategy:**

**Terminology:**
- **Provider**: Inference infrastructure (baseten, novita, fireworks, cerebras...)
- **Model ID**: How the provider names the model (e.g., `deepseek-ai/DeepSeek-V3.2`)
- **Model reference**: Canonical ID for the underlying model, shared across providers hosting the same model (e.g., `deepseek_v32`)
- **Candidate**: A specific provider + model combination to try
- **Virtual model**: A routing rule with multiple candidates to try in order

**Example: Virtual model with 3 candidates**
```yaml
virtual:deepseek-cheap:
  candidates:
    - baseten:deepseek/deepseek-3.2    # model_reference: deepseek_v32
    - novita:deepseek/deepseek-3.2     # model_reference: deepseek_v32
    - parasail:deepseek/deepseek-3.1   # model_reference: deepseek_v31
```

**Two types of errors, two failover behaviors:**

| Error Type | Examples | Failover |
|------------|----------|----------|
| **Infrastructure error** | Timeout, connection error, truncated response | Try same model on different provider |
| **Model error** | Schema validation fails after all retries | Skip same model, try different model |

**Example: Infrastructure error (timeout)**
```
Request to virtual:deepseek-cheap
  → baseten: timeout after 120s → InfrastructureError
  → baseten enters 15-minute cooldown
  → novita: SUCCESS ✓
```

**Example: Model error (schema validation)**
```
Request to virtual:deepseek-cheap with json_schema
  → baseten: response fails schema → fixer can't fix → retries exhausted → ModelError
  → novita: SKIPPED (same model_reference "deepseek_v32" would fail same way)
  → parasail: try deepseek-3.1 (different model_reference "deepseek_v31")
```

**LLM fixer for schema errors:**
When schema validation fails, Elelem calls a secondary LLM to attempt repair:
- Fixer returns `fixable: false` (truncated/empty) → treat as infrastructure error → failover to next provider
- Fixer returns fixed content → use it, success
- Fixer fails to fix → reduce temperature → retry → eventually ModelError

**Auto-repair features:**
- Strips markdown code blocks (` ```json ... ``` `)
- Fixes trailing commas, single quotes, unquoted keys
- CSV uses tilde (`~`) for null/empty values

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

**Access raw data:**
```python
data = elelem.get_metrics_data(tags=["env:production"])
# Returns List[Dict] with all request details

# Filter by time and tags
from datetime import datetime, timedelta
start = datetime.now() - timedelta(hours=24)
data = elelem.get_metrics_data(start_time=start, tags=["user:123"])
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

### 5. Response Caching

Elelem includes a built-in PostgreSQL-based response cache that reduces costs and latency by reusing identical responses.

**Why PostgreSQL instead of alternatives?**

- **Not in-memory (memcached/local dict):** Serverless containers are ephemeral - cache would be lost on every deployment/scale-down
- **Not Redis:** Adds infrastructure complexity and cost. For LLM caching (large payloads, infrequent writes, simple TTL), PostgreSQL is sufficient
- **Not pg_cache extension:** Requires PostgreSQL superuser privileges, not available in managed databases (RDS, Cloud SQL, Supabase)
- **PostgreSQL native:** Reuses existing metrics database, no extra infrastructure. Advisory locks handle multi-worker coordination without Redis/etcd

**Configuration (Server Mode):**

```bash
# Enable cache
export ELELEM_CACHE_ENABLED=true
export ELELEM_CACHE_TTL=300              # TTL in seconds (default: 5 min)
export ELELEM_CACHE_MAX_SIZE=50000       # Max response size in bytes
export ELELEM_CACHE_CLEANUP_INTERVAL=600 # Cleanup every N seconds
```

**Configuration (Library Mode):**

```python
elelem = Elelem(
    cache_enabled=True,
    cache_ttl=300,        # 5 minutes
    cache_max_size=50000  # 50KB max response
)
```

**Cache behavior:**
- **Cache key includes:** model, messages, temperature, max_tokens, response_format, json_schema
- **Cache key excludes:** tags (metadata), stream (format), user (identifier)
- **TTL is fixed:** Based on entry creation time, not last access (simple time-based expiration)
- **Temperature matters:** Different temperatures create different cache keys (temp=0 and temp=0.5 are separate)
- **Bypass cache:** Use `cache=False` parameter to skip caching for specific requests

**Example:**

```python
# First request - cache MISS, hits API
response1 = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
    tags=["test:cache"]
)
# response1.elelem_metrics['cached'] = False
# response1.elelem_metrics['cost_usd'] > 0

# Second identical request - cache HIT, free!
response2 = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
    tags=["different:tags"]  # Tags don't affect cache key
)
# response2.elelem_metrics['cached'] = True
# response2.elelem_metrics['cost_usd'] = 0.0
# response2.elelem_metrics['cache_age_seconds'] = 2.5

# Bypass cache
response3 = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
    cache=False  # Explicitly bypass cache
)
# response3.elelem_metrics['cached'] = False
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

**Smart Routing configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ELELEM_EXPLORATION_EPSILON` | `0.1` | Minimum exploration rate (steady state). When all providers have been tested, 10% of requests shuffle randomly to detect performance changes. |
| `ELELEM_EXPLORATION_EPSILON_MAX` | `1.0` | Maximum exploration rate (cold start). When no providers have been tested, 100% of requests shuffle randomly to gather data quickly. Scales linearly with coverage. |
| `ELELEM_DYNAMIC_ROUTING_ENABLED` | `true` | When enabled, Elelem reorders candidates based on observed performance. Disable to use YAML definition order. |
| `ELELEM_DYNAMIC_ROUTING_CACHE_TTL` | `30` | How long (seconds) to cache aggregated performance stats before re-querying the database. Lower = more responsive to changes, higher = less DB load. |
| `ELELEM_DYNAMIC_ROUTING_WINDOW_MINUTES` | `240` | Time window for performance stats (default: 4 hours). Only requests from this window are considered. |
| `ELELEM_DYNAMIC_ROUTING_MAX_SAMPLES` | `5` | Max samples per model for averaging. Uses only the N most recent requests per model within the window. Recent-biased to reflect current performance. |
| `ELELEM_DYNAMIC_ROUTING_COOLDOWN_MINUTES` | `15` | Cooldown period for failed providers. After a provider fails, it's excluded from routing for this duration. |

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
