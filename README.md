# Elelem 🎪 The "It Just Works™" Multi-Provider LLM Gateway

> *Because managing 47 models across 8 providers shouldn't require a PhD in API archaeology*

## Why Elelem Exists (A Love Story)

Once upon a time, I just wanted my LLMs to return JSON. Simple, right? **WRONG.**

- OpenAI wants `response_format: {type: "json_object"}`
- Groq wants it but sometimes doesn't support it
- DeepInfra laughs at your JSON dreams
- Fireworks needs `stream: true` even when you don't want streaming
- Don't even get me started on reasoning tokens...

So Elelem was born - not to pick the best model (that's your job), but to make sure when sh*t hits the fan with Provider A, you can seamlessly fail over to Provider B, C, or even D. It's about **infrastructure redundancy**, not model selection.

Think of it as:
- 🎯 A very tiny [LiteLLM](https://github.com/BerriAI/litellm), but obsessed with JSON
- 🌐 Your own local [OpenRouter](https://openrouter.ai/) (which btw, is also an Elelem provider!)
- 🤹 A circus juggler that never drops the JSON ball

### The Never-Ending Quest

Started as a pet project to figure out why one could return JSON but the other would rather write poetry about JSON. Now it's a never-ending quest to tame the wild west of LLM APIs. Very, very far from complete - new providers break things weekly, and so do I! 🎢

## Key Features (The Good Stuff)

### 🎭 Virtual Models - Your Failover Safety Net

Define virtual models that iterate through different providers until one works:

```yaml
# In your virtual-models.yaml
models:
  "virtual:gpt-oss-120b-reliable":
    candidates:
      - model: "groq:openai/gpt-oss-120b"      # Try Groq first (fast!)
        timeout: 10s
      - model: "fireworks:openai/gpt-oss-120b" # Fall back to Fireworks
        timeout: 15s
      - model: "deepinfra:openai/gpt-oss-120b" # Last resort
        timeout: 30s
```

```python
# Just use it like any other model
response = await elelem.create_chat_completion(
    model="virtual:gpt-oss-120b-reliable",
    messages=[{"role": "user", "content": "Never fail me!"}]
)
# Elelem handles the provider dance for you
```

### 🚀 Dynamic Models - Create Virtual Models on the Fly

```python
# When you need a custom failover RIGHT NOW
response = await elelem.create_chat_completion(
    model="dynamic:{candidates: [groq:openai/gpt-oss-120b, openai:gpt-4.1], timeout: 30s}",
    messages=[{"role": "user", "content": "I'm dynamically redundant!"}]
)
```

### 🎯 JSON Mode That Actually Works

Elelem is paranoid about JSON. It will:
1. Try the native JSON mode if supported
2. Strip markdown code blocks if the model gets creative
3. Fix common JSON errors (trailing commas, single quotes)
4. Retry with lower temperature if all else fails
5. Validate against your schema (if provided)

```python
# This WILL return valid JSON, or die trying (it won't die, it'll failover)
response = await elelem.create_chat_completion(
    model="any-model-really",
    messages=[{"role": "user", "content": "Return a JSON with name and age"}],
    response_format={"type": "json_object"},  # Elelem handles model quirks
    json_schema={  # Optional but recommended
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
)
```

### 🔌 Use as Library or OpenAI-Compatible Server

**As a Library:**
```python
from elelem import Elelem

elelem = Elelem()
response = await elelem.create_chat_completion(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**As an OpenAI-Compatible Server:**
```bash
# Start the server
docker-compose -f src/elelem/server/docker-compose.yml up -d

# Use with OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
response = client.chat.completions.create(
    model="groq:openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 📊 Built-in Pandas Metrics Store

Every call is tracked, tagged, and ready for analysis:

```python
# Tag your calls for later analysis
response = await elelem.create_chat_completion(
    model="openai:gpt-4.1",
    messages=[{"role": "user", "content": "Analyze this"}],
    tags=["experiment-42", "production"]
)

# Later, get your metrics
df = elelem.get_metrics_dataframe(tags=["experiment-42"])
# Ready for S3, BigQuery, or your favorite data warehouse
```

### 🎮 OpenRouter as a Provider (Meta!)

Yes, OpenRouter can be an Elelem provider. It's providers all the way down:

```python
# Use OpenRouter's infrastructure through Elelem
response = await elelem.create_chat_completion(
    model="openrouter:anthropic/claude-sonnet-4",  # OpenRouter's Claude
    messages=[{"role": "user", "content": "Inception!"}]
)

# Or their optimized routing
response = await elelem.create_chat_completion(
    model="cost@openrouter:openai/gpt-oss-120b",  # Cost-optimized routing
    messages=[{"role": "user", "content": "Cheap and cheerful!"}]
)

### 🥊 The OpenAI SDK Override Championship

Elelem completely takes over retry logic from the OpenAI SDK. Why? Because:
- OpenAI SDK: "Let me retry 3 times with the same provider that's down"
- Elelem: "Let me try 3 different providers that actually work"

Built on OpenAI SDK, but with trust issues and commitment to redundancy.

### 🎨 Response Harmonization

Every provider returns data differently. Elelem makes them all look like OpenAI:
- Reasoning tokens? ✓ (even when hidden in weird places)
- Usage stats? ✓ (standardized across all providers)
- Cost calculation? ✓ (because money matters)
- Model metadata? ✓ (who made this model anyway?)

## What Elelem Is NOT

- ❌ Not a model picker (that's your job)
- ❌ Not trying to be LangChain (too many abstractions)
- ❌ Not competing with OpenRouter (we're friends!)
- ❌ Not complete (new providers break things weekly)
- ❌ Not using Pydantic yet (yes, I know, I'm sorry!)

## Installation

### Local Development
```bash
pip install -e /path/to/Elelem/
```

### From Git Repository
```bash
pip install git+https://github.com/yourorg/elelem.git
```

### From Private PyPI
```bash
pip install elelem --index-url https://your-private-pypi.com
```

## Quick Start - The 60-Second Tour

```python
import asyncio
from elelem import Elelem

async def main():
    # Initialize Elelem (it loads 47 models and hopes for the best)
    elelem = Elelem()

    # Method 1: The "I have a favorite provider" approach
    response = await elelem.create_chat_completion(
        model="groq:openai/gpt-oss-120b",  # Fast and furious
        messages=[{"role": "user", "content": "Say hello in JSON"}],
        response_format={"type": "json_object"}  # Will work even if model doesn't support it
    )

    # Method 2: The "I don't care who answers" approach (recommended!)
    response = await elelem.create_chat_completion(
        model="virtual:gpt-oss-120b-quick",  # Tries multiple providers
        messages=[{"role": "user", "content": "What's 2+2? In JSON please."}],
        response_format={"type": "json_object"}
    )

    # Method 3: The "I'm feeling lucky" dynamic approach
    response = await elelem.create_chat_completion(
        model="dynamic:{candidates: [groq:openai/gpt-oss-120b, openai:gpt-4.1]}",
        messages=[{"role": "user", "content": "Make me a sandwich... in JSON"}],
        tags=["lunch", "experimental"]  # Track your experiments
    )

    # It's all OpenAI-compatible (because standards are nice)
    print(response.choices[0].message.content)  # Your JSON
    print(f"Cost: ${response.usage.cost_usd:.4f}")  # Your bill

    # Check the damage
    stats = elelem.get_stats_by_tag("lunch")
    print(f"Lunch experiments cost: ${stats['total_cost_usd']:.4f}")
    print(f"Retries: {stats['total_retries']}")  # How many providers failed you

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GROQ_API_KEY="your-groq-key" 
export DEEPINFRA_API_KEY="your-deepinfra-key"
export SCALEWAY_ACCESS_KEY="your-scaleway-access-key"
export SCALEWAY_SECRET_KEY="your-scaleway-secret-key"
export FIREWORKS_API_KEY="your-fireworks-key"
export OPENROUTER_API_KEY="your-openrouter-key"
export PARASAIL_API_KEY="your-parasail-key"
```

## Virtual Models & Candidate System

Elelem now supports virtual models that automatically fall back across multiple providers:

```python
# Virtual model with automatic provider fallback
response = await elelem.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    model="virtual:gpt-oss-120b",  # Tries multiple providers automatically
    tags=["fallback-test"]
)
```

Virtual models try candidates in order with configurable timeouts:
1. Primary provider (e.g., GROQ) - 120s timeout
2. Fallback provider (e.g., Scaleway) - 240s timeout  
3. Additional providers as configured

## Reasoning Token Analytics

Elelem provides detailed analytics for reasoning models like o3, DeepSeek-R1, and thinking mode:

```python
# Use reasoning model
response = await elelem.create_chat_completion(
    messages=[{"role": "user", "content": "What's 2+2? Think step by step."}],
    model="openai:o3-mini",  # or "parasail:deepseek-3.1-think"
    tags=["reasoning"]
)

# Get reasoning analytics
stats = elelem.get_stats_by_tag("reasoning")
print(f"Reasoning tokens: {stats['reasoning_tokens']}")
print(f"Output tokens: {stats['total_output_tokens']}")
print(f"Reasoning cost: ${stats['reasoning_cost_usd']:.4f}")
print(f"Output speed: {stats['total_output_tokens'] / stats['total_duration']:.1f} tokens/s")
```

**Key Features:**
- **Dual Token Rates**: Total generation speed vs actual output speed
- **Cost per Actual Token**: More accurate cost analysis for reasoning models
- **Thinking Mode Support**: Automatic `<think>` tag extraction for DeepSeek
- **Provider Compatibility**: Works across OpenAI, GROQ, DeepInfra, Parasail

## Request Flow (Candidate-Based Architecture)

```mermaid
graph TD
    A[create_chat_completion] --> B[Get Model Config]
    B --> C[Parse Candidates]
    C --> D[Candidate Loop Start<br/>candidate=1/N]
    
    D --> E[Setup Candidate]
    E --> F[Get Timeout for Candidate]
    F --> G[Setup Provider Client]
    G --> H[Add Provider/Model Defaults]
    H --> I[Cleanup Unsupported Params]
    
    I --> J[Retry Loop Start<br/>attempt=1/4]
    J --> K[Make API Call with Timeout]
    
    K --> L{API Result}
    
    L -->|Timeout| M[Infrastructure Error<br/>Try Next Candidate]
    L -->|503/502/401| M
    L -->|429 Rate Limit| N[Rate Limit Retry<br/>Exponential Backoff]
    L -->|JSON Validation Error| O[Handle JSON Error]
    L -->|Other API Error| P[Model Error<br/>Don't Iterate]
    L -->|200 Success| Q[Extract Usage Stats]
    
    M --> R{More Candidates?}
    R -->|Yes| S[Next Candidate<br/>+candidate_iterations]
    R -->|No| T[All Candidates Failed]
    S --> D
    
    N --> U{Max Rate Retries?}
    U -->|No| V[Wait & Retry]
    U -->|Yes| M
    V --> J
    
    O --> W[Reduce Temperature<br/>+temperature_reductions]
    W --> X{Max JSON Retries?}
    X -->|No| J
    X -->|Yes| Y[Remove response_format<br/>+response_format_removals]
    Y --> J
    
    Q --> Z[Extract Reasoning Tokens]
    Z --> AA[Process Response Content]
    AA --> BB[Remove Think Tags]
    BB --> CC[Extract from Markdown]
    CC --> DD{JSON Mode?}
    
    DD -->|No| SUCCESS[Return Response<br/>Update Statistics]
    DD -->|Yes| EE[Parse & Validate JSON]
    
    EE --> FF{Valid JSON & Schema?}
    FF -->|Yes| SUCCESS
    FF -->|No| O
    
    P --> T
    T --> GG[Throw Final Error]
    
    style SUCCESS fill:#90EE90
    style GG fill:#FFB6C1
    style M fill:#FFE4B5
    style O fill:#FFE4B5
    style S fill:#E0E0E0
    
```

**Key Improvements:**
- **🔄 Candidate Iteration**: Automatic fallback across providers for infrastructure failures
- **⏱️ Timeout Hierarchy**: Per-candidate, per-model, and global timeout settings
- **🧠 Reasoning Token Extraction**: Handles OpenAI, GROQ, and Parasail formats
- **🏷️ Think Tag Processing**: Automatic removal with proper content extraction
- **📊 Enhanced Analytics**: Tracks candidate iterations and infrastructure failures

## Telelem Batch Testing

Comprehensive benchmarking tool for testing multiple models with rich metadata:

```bash
# Run batch test with multiple models
uv run python telelem.py --batch tests/telelem/batch.json --output results.json

# Single model test
uv run python telelem.py --model parasail:deepseek-3.1-think --prompt tests/telelem/small.yaml --output test.json
```

**Features:**
- **Self-Contained Results**: JSON output includes complete model metadata
- **Dashboard Ready**: Zero external dependencies for frontend consumption
- **Reasoning Analytics**: Dual token rates and cost per actual output token
- **Rich Metadata**: Display names, owners, reasoning capabilities, licenses
- **Provider Coverage**: Tests across all 8 providers and 47+ models

## JSON Schema Validation

Elelem supports automatic JSON schema validation with intelligent retry strategies:

```python
response = await elelem.create_chat_completion(
    messages=[{"role": "user", "content": "Generate user data"}],
    model="groq:openai/gpt-oss-120b",
    response_format={"type": "json_object"},
    json_schema=your_schema_dict,  # Your JSON Schema definition
    temperature=1.5
)
```

**Benefits:**
- **Guaranteed Structure**: Response will match your schema or fail gracefully
- **Automatic Retries**: Temperature reduction → response_format removal → candidate iteration
- **Detailed Error Logging**: Shows exactly what failed validation
- **Production Ready**: Handles edge cases like malformed JSON and API rejections

## Supported Models

### OpenAI Models
- `openai:gpt-4.1` - Latest GPT-4.1 model  
- `openai:gpt-4.1-mini` - Cost-effective GPT-4.1 variant
- `openai:o3` - Reasoning model (no temperature support)
- `openai:o3-mini` - Cost-effective reasoning model

### GROQ Models  
- `groq:openai/gpt-oss-120b` - Large open-source GPT model
- `groq:openai/gpt-oss-20b` - Medium open-source GPT model
- `groq:meta-llama/llama-4-maverick-17b-128e-instruct` - Llama 4 Maverick
- `groq:moonshotai/kimi-k2-instruct` - Kimi K2 instruction model

### Scaleway Models
- `scaleway:gpt-oss-120b` - Large open-source GPT (€0.15/€0.60 per 1M tokens)
- `scaleway:gemma-3-27b-it` - Google Gemma 3 27B (€0.25/€0.50 per 1M tokens)
- `scaleway:mistral-small-3.2-24b-instruct-2506` - Mistral Small (€0.15/€0.35 per 1M tokens)

### Parasail Models
- `parasail:deepseek-3.1` - DeepSeek 3.1 standard mode
- `parasail:deepseek-3.1-think` - DeepSeek 3.1 with thinking mode enabled
- `parasail:gpt-oss-120b` - GPT OSS 120B via Parasail

### Fireworks Models
- `fireworks:deepseek-v3p1` - DeepSeek V3.1 via Fireworks  
- `fireworks:qwen2.5-coder-32b-instruct` - Qwen 2.5 Coder 32B
- `fireworks:llama-v3p3-70b-instruct` - Llama 3.3 70B

### Virtual Models (Multi-Provider Fallback)
- `virtual:gpt-oss-120b` - Tries GROQ → Scaleway → DeepInfra
- `virtual:deepseek-v3p1` - Tries Fireworks → DeepInfra with different timeout strategies

**Provider Coverage**: 8 providers, 47+ models with full metadata and cost tracking

## Retry Analytics

Elelem tracks detailed metrics on retry events and failure patterns:

```python
stats = elelem.get_stats_by_tag("your_tag")
retry_analytics = stats["retry_analytics"]

# Available metrics:
retry_analytics["json_parse_retries"]           # Malformed JSON syntax
retry_analytics["json_schema_retries"]          # Valid JSON, wrong structure  
retry_analytics["api_json_validation_retries"]  # Provider rejected request
retry_analytics["rate_limit_retries"]           # Rate limit backoff events
retry_analytics["temperature_reductions"]       # Temperature adjustment events
retry_analytics["response_format_removals"]     # Fallback strategy usage
retry_analytics["candidate_iterations"]         # Provider fallback events
retry_analytics["final_failures"]              # Requests that never succeeded
retry_analytics["total_retries"]               # Sum of all retry events
```

**Use Cases**: Production monitoring, cost optimization, provider reliability analysis, temperature tuning

## API Reference

### Main Methods

#### `create_chat_completion(messages, model, tags=[], **kwargs)`

Creates a chat completion with automatic candidate iteration and fallback.

**Parameters:**
- `messages` (List[Dict]): List of message dictionaries
- `model` (str): Model string in "provider:model" or "virtual:model" format
- `tags` (Union[str, List[str]]): Tags for cost tracking
- `**kwargs`: Additional OpenAI API parameters

**Returns:**
- OpenAI-compatible response with `choices[0]["message"]["content"]` access

#### `get_stats()`

Returns overall usage statistics including reasoning tokens.

**Returns:**
- Dictionary with token counts, costs, call counts, reasoning analytics

#### `get_stats_by_tag(tag)`

Returns usage statistics filtered by a specific tag.

**Parameters:**
- `tag` (str): Tag to filter statistics by

**Returns:**
- Dictionary with tag-specific statistics

#### `list_models()`

Returns all available models with rich metadata.

**Returns:**
- Dictionary with model information including display names, owners, capabilities

## Special Features

### Reasoning Token Analytics
- **OpenAI o3/o3-mini**: Native reasoning_tokens field extraction
- **GROQ**: Output tokens details parsing
- **Parasail DeepSeek**: `<think>` tag content analysis with character ratio estimation
- **Cost Analysis**: Separate reasoning and output cost tracking

### Thinking Mode (DeepSeek)
```python
# Enable thinking mode
response = await elelem.create_chat_completion(
    messages=[{"role": "user", "content": "What's 2+2? Think step by step."}],
    model="parasail:deepseek-3.1-think",  # Automatically includes thinking: true
    tags=["thinking"]
)
```

- Automatically removes `<think>...</think>` tags from response
- Estimates reasoning tokens using character count ratios
- Provides clean output while preserving reasoning analytics

### Virtual Model Fallback
Virtual models automatically try multiple providers:
```python
# This model tries GROQ first, then falls back to Scaleway
model="virtual:gpt-oss-120b"
```

**Timeout Hierarchy:**
1. Candidate-level timeout (if specified)
2. Model-level timeout (if specified)  
3. Global timeout (default: 120s)

### Cost Tracking
- **Reasoning Token Costs**: Separate tracking for reasoning vs output tokens
- **Provider Comparison**: Runtime costs from OpenRouter when available
- **Tag-Based Analytics**: Project and category-based cost allocation

## Architecture

Elelem follows a modular, provider-agnostic architecture:

- **elelem.py**: Main Elelem class with candidate-based iteration
- **config.py**: Unified configuration system with metadata resolution
- **providers/**: Provider-specific YAML files with model definitions
- **providers/_metadata.yaml**: Centralized metadata definitions
- **telelem.py**: Comprehensive batch testing and benchmarking tool

### Configuration System
- **Provider Files**: `src/elelem/providers/openai.yaml`, `parasail.yaml`, etc.
- **Metadata References**: DRY system using `metadata_ref` to shared definitions
- **Auto-Discovery**: Automatically loads all provider YAML files
- **Backward Compatibility**: Maintains existing API while adding new features

## Development

### Running Tests

```bash
# Run all tests (simple - works out of the box!)
uv run pytest

# Run with more details
uv run pytest -v

# Run specific test categories
uv run pytest tests/test_config_validation.py    # Config validation only
uv run pytest tests/test_real_providers.py       # Real provider tests (needs API keys)
uv run pytest tests/test_elelem_with_faker.py    # Faker tests (no API keys needed)

# Run with coverage report
uv run pytest --cov=elelem --cov-report=html
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Add models to provider YAML files with metadata_ref
5. Run the test suite (`pytest`)
6. Submit a pull request

## Support

For issues and questions:
- Check the [documentation](SPECIFICATION.md)
- Review existing [issues](https://github.com/yourorg/elelem/issues)
- Submit a [new issue](https://github.com/yourorg/elelem/issues/new)