# Faker Scenario Syntax

## Overview

Scenarios define how the faker server responds to requests. Each scenario is a YAML file that specifies models, response behaviors, and error conditions.

## Basic Structure

```yaml
scenario: "scenario_name"
description: "Human readable description"
type: "fixed|sequence|conditional"

models:
  "model_name":
    provider: faker
    capabilities:
      supports_json_mode: true
      supports_temperature: true
      supports_system: true

# Response definition (varies by type)
```

## Scenario Types

### 1. Fixed Scenarios
Always return the same response:

```yaml
scenario: "happy_path"
type: "fixed"

models:
  "faker:basic":
    provider: faker

response:
  type: "success"
  content: "Always returns this message"
  tokens:
    input: 50
    output: 15
  json_data:
    status: "success"
    message: "JSON mode response"
```

### 2. Sequence Scenarios
Return different responses in order (great for testing retries):

```yaml
scenario: "rate_limits"
type: "sequence"

models:
  "faker:rate-limited":
    provider: faker

response_sequence:
  - type: "rate_limit"        # First request → 429
    retry_after: 5
  - type: "rate_limit"        # Second request → 429
    retry_after: 5
  - type: "success"           # Third request → 200
    content: "Success after retries"
```

### 3. Conditional Scenarios
Return different responses based on request content:

```yaml
scenario: "json_temperature_reduction"
type: "conditional"

models:
  "faker:json-temp-test":
    provider: faker

conditions:
  - check: "temperature_above"    # If temperature > 0.8
    value: 0.8
    response:
      type: "json_error"         # Return malformed JSON

  - check: "json_mode"           # If JSON mode requested
    response:
      type: "success"
      json_data:
        status: "success"
        message: "Valid JSON"

default_response:
  type: "success"
  content: "Default text response"
```

## Response Types

### Success Response
```yaml
response:
  type: "success"
  content: "Response text"
  tokens:
    input: 50
    output: 20
  json_data:                    # Used when JSON mode requested
    key: "value"
```

### Error Responses
```yaml
# Rate limit (429)
response:
  type: "rate_limit"
  retry_after: 5

# Authentication error (401)
response:
  type: "auth_error"

# Server timeout (408)
response:
  type: "timeout"

# Server overloaded (503)
response:
  type: "overloaded"

# Malformed JSON (for testing JSON retry logic)
response:
  type: "json_error"
```

## Condition Types

### Temperature-based
```yaml
- check: "temperature_above"
  value: 0.8
  response: ...

- check: "temperature_below"
  value: 0.3
  response: ...
```

### Content-based
```yaml
- check: "message_contains"
  value: "timeout"              # If any message contains "timeout"
  response:
    type: "timeout"

- check: "model_contains"
  value: "gpt-4"               # If model name contains "gpt-4"
  response: ...
```

### Request format
```yaml
- check: "json_mode"           # If response_format.type == "json_object"
  response: ...
```

## Complete Example

```yaml
scenario: "comprehensive_test"
description: "Tests multiple error conditions and recovery"
type: "conditional"

models:
  "faker:comprehensive":
    provider: faker
    capabilities:
      supports_json_mode: true
      supports_temperature: true
      supports_system: true

conditions:
  # High temperature + JSON mode → malformed JSON
  - check: "temperature_above"
    value: 0.8
    response:
      type: "json_error"

  # Message contains "error" → timeout
  - check: "message_contains"
    value: "error"
    response:
      type: "timeout"

  # JSON mode → valid JSON
  - check: "json_mode"
    response:
      type: "success"
      json_data:
        result: "valid"
        temperature: "acceptable"

# Default for non-matching requests
default_response:
  type: "success"
  content: "Standard response"
  tokens:
    input: 40
    output: 12
```

## Testing Your Scenarios

```bash
# Test with the CLI
uv run python tests/faker/cli.py --scenario your_scenario.yaml --port 8899

# Test with curl
curl -X POST http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "faker:your-model", "messages": [{"role": "user", "content": "test"}]}'
```