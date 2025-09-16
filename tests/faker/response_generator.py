"""Response generation system for the faker."""

import json
import uuid
import time
from typing import Dict, Any, Optional, List
from flask import jsonify


class ResponseGenerator:
    """Generates OpenAI-compatible responses for the faker system."""

    def generate_openai_response(
        self,
        content: str,
        tokens: Optional[Dict[str, int]] = None,
        model: str = "faker:default",
        finish_reason: str = "stop"
    ) -> Dict[str, Any]:
        """Generate standard OpenAI chat completion response."""

        if tokens is None:
            # Estimate tokens from content
            tokens = {
                'input': len(content.split()) * 2,  # Rough estimate
                'output': len(content.split())
            }

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": tokens['input'],
                "completion_tokens": tokens['output'],
                "total_tokens": tokens['input'] + tokens['output']
            }
        }

        return jsonify(response)

    def generate_json_response(
        self,
        json_data: Dict[str, Any],
        tokens: Optional[Dict[str, int]] = None,
        model: str = "faker:default",
        valid: bool = True
    ):
        """Generate JSON mode response."""

        if valid:
            content = json.dumps(json_data)
        else:
            # Generate malformed JSON for testing error handling
            content = json.dumps(json_data)[:-1]  # Remove closing brace

        return self.generate_openai_response(
            content=content,
            tokens=tokens,
            model=model
        )

    def generate_error_response(
        self,
        error_type: str,
        message: str,
        code: str,
        status_code: int = 400
    ):
        """Generate OpenAI-compatible error response."""

        error_response = {
            "error": {
                "message": message,
                "type": error_type,
                "code": code
            }
        }

        return jsonify(error_response), status_code

    def generate_rate_limit_error(self, retry_after: int = 5):
        """Generate 429 rate limit error with retry-after header."""
        response = jsonify({
            "error": {
                "message": "Rate limit exceeded. Please retry after a few seconds.",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        })
        response.status_code = 429
        response.headers['Retry-After'] = str(retry_after)
        return response

    def generate_timeout_response(self):
        """Generate timeout response (simulated by long delay)."""
        time.sleep(30)  # Simulate timeout
        return self.generate_error_response(
            error_type="timeout_error",
            message="Request timed out",
            code="timeout",
            status_code=408
        )

    def generate_auth_error(self):
        """Generate authentication error."""
        return self.generate_error_response(
            error_type="authentication_error",
            message="Invalid API key provided",
            code="invalid_api_key",
            status_code=401
        )

    def generate_overloaded_error(self):
        """Generate server overloaded error."""
        return self.generate_error_response(
            error_type="server_error",
            message="The server is currently overloaded. Please try again later.",
            code="server_overloaded",
            status_code=503
        )

    def generate_json_schema_response(self, schema: Dict[str, Any], valid: bool = True):
        """Generate response matching a JSON schema."""

        if valid:
            # Generate valid data matching the schema
            fake_data = self._generate_from_schema(schema)
            return self.generate_json_response(fake_data, valid=True)
        else:
            # Generate invalid data for testing error handling
            fake_data = {"invalid": "schema", "wrong": "format"}
            return self.generate_json_response(fake_data, valid=False)

    def _generate_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fake data that matches a JSON schema."""

        if schema.get('type') == 'object':
            result = {}
            properties = schema.get('properties', {})
            required = schema.get('required', [])

            for prop_name, prop_schema in properties.items():
                if prop_name in required or len(properties) <= 3:  # Generate most properties
                    result[prop_name] = self._generate_value_from_schema(prop_schema)

            return result

        return self._generate_value_from_schema(schema)

    def _generate_value_from_schema(self, schema: Dict[str, Any]) -> Any:
        """Generate a single value matching a schema type."""

        schema_type = schema.get('type', 'string')

        if schema_type == 'string':
            return schema.get('example', 'fake_string_value')
        elif schema_type == 'number' or schema_type == 'integer':
            return schema.get('example', 42)
        elif schema_type == 'boolean':
            return schema.get('example', True)
        elif schema_type == 'array':
            item_schema = schema.get('items', {'type': 'string'})
            return [self._generate_value_from_schema(item_schema)]
        elif schema_type == 'object':
            return self._generate_from_schema(schema)
        else:
            return f"fake_{schema_type}_value"

    def generate_reasoning_response(
        self,
        content: str,
        reasoning_content: str,
        model: str = "faker:reasoning"
    ):
        """Generate response with reasoning tokens (for models that support it)."""

        # Some models return reasoning in a specific format
        full_content = f"<thinking>\n{reasoning_content}\n</thinking>\n\n{content}"

        # Token calculation with reasoning
        reasoning_tokens = len(reasoning_content.split())
        content_tokens = len(content.split())
        input_tokens = 50  # Estimated

        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": content_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": input_tokens + content_tokens + reasoning_tokens
            }
        }

        return jsonify(response_data)

    def generate_reasoning_response_with_format(
        self,
        content: str,
        tokens: Dict[str, int],
        model: str = "faker:reasoning",
        reasoning_format: str = "default"
    ):
        """Generate response with different reasoning token formats."""
        import uuid
        import time

        input_tokens = tokens.get('input', 50)
        output_tokens = tokens.get('output', 20)
        reasoning_tokens = tokens.get('reasoning', 10)

        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens + reasoning_tokens
            }
        }

        # Add reasoning tokens in different formats based on provider
        if reasoning_format == "openai":
            # OpenAI format: nested in completion_tokens_details
            response_data["usage"]["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens
            }
        elif reasoning_format == "groq":
            # GROQ format: nested in output_tokens_details
            response_data["usage"]["output_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens
            }
        else:
            # Default/DeepSeek format: direct field
            response_data["usage"]["reasoning_tokens"] = reasoning_tokens

        return jsonify(response_data)

    def generate_streaming_response(self, chunks: List[Dict], tokens: Dict[str, int], model: str = "faker:stream"):
        """Generate streaming response with SSE format."""
        import uuid
        import time
        import json
        from flask import Response

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        def generate_chunks():
            # Send initial chunk with metadata
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            # Send content chunks
            for chunk_data in chunks:
                if "delta" in chunk_data:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": chunk_data["delta"],
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif "finish_reason" in chunk_data:
                    # Final chunk with usage info
                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": chunk_data["finish_reason"]
                        }],
                        "usage": {
                            "prompt_tokens": tokens.get('input', 50),
                            "completion_tokens": tokens.get('output', 20),
                            "total_tokens": tokens.get('input', 50) + tokens.get('output', 20)
                        }
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

            # End stream
            yield "data: [DONE]\n\n"

        return Response(
            generate_chunks(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )

    def generate_moderation_response(self, flagged: bool = False):
        """Generate content moderation response."""
        return jsonify({
            "id": f"modr-{uuid.uuid4().hex[:8]}",
            "model": "text-moderation-007",
            "results": [{
                "flagged": flagged,
                "categories": {
                    "hate": False,
                    "hate/threatening": False,
                    "harassment": False,
                    "harassment/threatening": False,
                    "self-harm": False,
                    "self-harm/intent": False,
                    "self-harm/instructions": False,
                    "sexual": False,
                    "sexual/minors": False,
                    "violence": False,
                    "violence/graphic": False
                },
                "category_scores": {
                    "hate": 0.001,
                    "hate/threatening": 0.001,
                    "harassment": 0.001,
                    "harassment/threatening": 0.001,
                    "self-harm": 0.001,
                    "self-harm/intent": 0.001,
                    "self-harm/instructions": 0.001,
                    "sexual": 0.001,
                    "sexual/minors": 0.001,
                    "violence": 0.001,
                    "violence/graphic": 0.001
                }
            }]
        })