"""Request validation and analysis for the faker system."""

import time
from typing import Dict, Any, List, Optional


class RequestAnalyzer:
    """Captures and validates incoming requests to verify Elelem's behavior."""

    def __init__(self):
        self.captured_requests: List[Dict[str, Any]] = []
        self.expectations: Dict[str, Any] = {}

    def analyze_request(self, request) -> Dict[str, Any]:
        """Capture and validate incoming Flask request."""
        captured = {
            'timestamp': time.time(),
            'method': request.method,
            'path': request.path,
            'headers': dict(request.headers),
            'body': request.get_json() or {},
            'query_params': dict(request.args)
        }

        self.captured_requests.append(captured)

        # Perform validations
        self._validate_headers(captured['headers'])
        self._validate_body(captured['body'])

        return captured

    def _validate_headers(self, headers: Dict[str, str]):
        """Validate request headers."""
        # Check required headers
        required_headers = self.expectations.get('required_headers', [])
        for header in required_headers:
            header_lower = header.lower()
            found = any(h.lower() == header_lower for h in headers.keys())
            assert found, f"Required header '{header}' not found in request"

        # Check forbidden headers
        forbidden_headers = self.expectations.get('forbidden_headers', [])
        for header in forbidden_headers:
            header_lower = header.lower()
            found = any(h.lower() == header_lower for h in headers.keys())
            assert not found, f"Forbidden header '{header}' found in request"

        # Check authentication
        if 'authorization' in [h.lower() for h in headers.keys()]:
            auth_header = next(v for k, v in headers.items() if k.lower() == 'authorization')
            assert auth_header.startswith('Bearer '), "Authorization should be Bearer token"

    def _validate_body(self, body: Dict[str, Any]):
        """Validate request body parameters."""
        if not body:
            return

        # Check required parameters
        required_params = self.expectations.get('required_params', [])
        for param in required_params:
            assert param in body, f"Required parameter '{param}' not found in request body"

        # Check forbidden parameters (cleaned up by Elelem)
        forbidden_params = self.expectations.get('forbidden_params', [])
        for param in forbidden_params:
            assert param not in body, f"Parameter '{param}' should have been removed by Elelem"

        # Validate specific parameter formats
        self._validate_messages_format(body.get('messages', []))
        self._validate_response_format(body.get('response_format'))

    def _validate_messages_format(self, messages: List[Dict[str, Any]]):
        """Validate messages array format."""
        if not messages:
            return

        for i, message in enumerate(messages):
            assert 'role' in message, f"Message {i} missing 'role' field"
            assert 'content' in message, f"Message {i} missing 'content' field"

            valid_roles = ['system', 'user', 'assistant']
            assert message['role'] in valid_roles, f"Message {i} has invalid role: {message['role']}"

            assert isinstance(message['content'], str), f"Message {i} content must be string"

    def _validate_response_format(self, response_format: Optional[Dict[str, Any]]):
        """Validate response_format parameter."""
        if response_format is None:
            return

        assert isinstance(response_format, dict), "response_format must be dict"

        if 'type' in response_format:
            valid_types = ['json_object', 'text']
            assert response_format['type'] in valid_types, f"Invalid response_format type: {response_format['type']}"

    def set_expectations(self, expectations: Dict[str, Any]):
        """Set validation expectations for requests."""
        self.expectations = expectations

    def get_captured_requests(self) -> List[Dict[str, Any]]:
        """Get all captured requests."""
        return self.captured_requests.copy()

    def clear_requests(self):
        """Clear captured request log."""
        self.captured_requests.clear()

    def get_request_count(self) -> int:
        """Get total number of captured requests."""
        return len(self.captured_requests)

    def get_last_request(self) -> Optional[Dict[str, Any]]:
        """Get the most recent captured request."""
        return self.captured_requests[-1] if self.captured_requests else None

    def get_requests_for_model(self, model: str) -> List[Dict[str, Any]]:
        """Get all requests for a specific model."""
        return [
            req for req in self.captured_requests
            if req.get('body', {}).get('model') == model
        ]

    def validate_parameter_cleanup(self, model_capabilities: Dict[str, bool]):
        """Validate that Elelem properly cleaned up parameters based on model capabilities."""
        for request in self.captured_requests:
            body = request.get('body', {})

            # If model doesn't support JSON, response_format should be removed
            if not model_capabilities.get('supports_json_mode', True):
                assert 'response_format' not in body, \
                    "response_format should be removed for non-JSON models"

            # If model doesn't support system messages, system messages should be converted
            if not model_capabilities.get('supports_system', True):
                messages = body.get('messages', [])
                system_messages = [m for m in messages if m.get('role') == 'system']
                assert len(system_messages) == 0, \
                    "System messages should be converted for models that don't support them"

    def validate_no_forbidden_headers(self):
        """Validate that no forbidden headers were sent."""
        forbidden = ['x-debug', 'x-test-mode']  # Example forbidden headers

        for request in self.captured_requests:
            headers = request.get('headers', {})
            for forbidden_header in forbidden:
                found = any(h.lower() == forbidden_header.lower() for h in headers.keys())
                assert not found, f"Forbidden header '{forbidden_header}' found in request"

    def assert_retry_count(self, expected_count: int):
        """Assert that exactly N requests were made (useful for retry testing)."""
        actual_count = len(self.captured_requests)
        assert actual_count == expected_count, \
            f"Expected {expected_count} requests (retries), got {actual_count}"

    def assert_temperature_sequence(self, expected_temps: List[float], tolerance: float = 0.01):
        """Assert temperature reduction pattern in requests."""
        actual_temps = []
        for request in self.captured_requests:
            body = request.get('body', {})
            temp = body.get('temperature')
            if temp is not None:
                actual_temps.append(temp)

        assert len(actual_temps) == len(expected_temps), \
            f"Expected {len(expected_temps)} temperatures, got {len(actual_temps)}"

        for i, (expected, actual) in enumerate(zip(expected_temps, actual_temps)):
            assert abs(expected - actual) <= tolerance, \
                f"Temperature {i}: expected {expected}, got {actual} (tolerance {tolerance})"

    def assert_json_mode_consistency(self):
        """Assert that JSON mode is consistently applied across retries."""
        json_requests = []
        for request in self.captured_requests:
            body = request.get('body', {})
            response_format = body.get('response_format')
            if response_format and response_format.get('type') == 'json_object':
                json_requests.append(request)

        # Either all requests have JSON mode, or none do
        assert len(json_requests) == 0 or len(json_requests) == len(self.captured_requests), \
            "JSON mode should be consistent across all retry attempts"