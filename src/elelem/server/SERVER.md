# Elelem OpenAI API Server

A FastAPI server that provides OpenAI-compatible endpoints using Elelem's multi-provider backend for resilient AI API access.

## Features

- **🔄 Multi-Provider Resilience**: Automatically fails over between AI providers
- **🔌 OpenAI Compatible**: Drop-in replacement for OpenAI API
- **🎯 Dynamic Models**: Support for on-demand virtual model creation
- **📊 Cost Tracking**: Built-in usage and cost analytics
- **🐳 Docker Ready**: Easy deployment with Docker/Docker Compose

## Quick Start

### Using Docker Compose (Recommended)

1. **Set up environment variables:**
   ```bash
   # From project root
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start the server:**
   ```bash
   # From this directory (src/elelem/server/)
   docker-compose up -d

   # Or from project root:
   docker-compose -f src/elelem/server/docker-compose.yml up -d
   ```

3. **Test the server:**
   ```bash
   curl http://localhost:8000/health
   ```

### Using Docker

```bash
# Build the image
docker build -t elelem-server .

# Run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  elelem-server
```

### Local Development

```bash
# Install dependencies
pip install -r requirements-server.txt

# Run the server
uvicorn elelem.server.main:app --reload --port 8000
```

## Usage

### With OpenAI Python Library

```python
import openai

# Point to your Elelem server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Elelem handles provider keys
)

# Use any Elelem model!
response = client.chat.completions.create(
    model="groq:openai/gpt-oss-120b",  # Multi-provider model
    messages=[{"role": "user", "content": "Hello!"}]
)

# Or use dynamic virtual models
response = client.chat.completions.create(
    model="dynamic:{candidates: ['groq:openai/gpt-oss-120b', 'deepinfra:openai/gpt-oss-120b']}",
    messages=[{"role": "user", "content": "Hello with failover!"}]
)
```

### With curl

```bash
# Chat completion
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq:openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

# List models
curl "http://localhost:8000/v1/models"
```

## API Endpoints

- **`POST /v1/chat/completions`** - OpenAI-compatible chat completions
- **`GET /v1/models`** - List available models
- **`GET /health`** - Health check
- **`GET /docs`** - Interactive API documentation

## Configuration

Set environment variables for the AI providers you want to use:

```bash
# Provider API Keys
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
DEEPINFRA_API_KEY=...
FIREWORKS_API_KEY=...
PARASAIL_API_KEY=...
SCALEWAY_API_KEY=...
OPENROUTER_API_KEY=...
```

## Benefits Over Direct OpenAI API

1. **🛡️ Resilience**: Automatic failover if a provider is down
2. **💰 Cost Optimization**: Route to cheaper providers automatically
3. **🚀 Performance**: Use faster providers when available
4. **🔄 Load Balancing**: Distribute requests across providers
5. **📈 Analytics**: Built-in usage tracking and cost monitoring

## Model Examples

```python
# Single provider models
"openai:openai/gpt-4.1"
"groq:openai/gpt-oss-120b"
"deepinfra:deepseek/deepseek-3.1?nothink"

# Static virtual models (defined in config)
"virtual:gpt-oss-120b-quick"
"virtual:deepseek-v3.1-cheap"

# Dynamic virtual models (runtime definition)
"dynamic:{candidates: ['groq:openai/gpt-oss-120b', 'fireworks:openai/gpt-oss-120b'], timeout: 60s}"
```

## Monitoring

- **Health endpoint**: `GET /health`
- **Logs**: Docker logs show request/response details
- **Metrics**: Built-in cost and usage tracking via Elelem

## Development

The server is a thin FastAPI wrapper around Elelem's core functionality:

```
src/elelem/server/
├── __init__.py
├── main.py          # FastAPI app and routes
└── models.py        # Pydantic request/response models
```

For development, see the main [Elelem documentation](./README.md).