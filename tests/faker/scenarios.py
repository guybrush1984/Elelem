"""Scenario management for the faker system."""

import os
import yaml
from typing import Dict, Any, Optional
from .response_generator import ResponseGenerator


class ScenarioManager:
    """Manages test scenarios and executes them."""

    def __init__(self):
        self.response_generator = ResponseGenerator()
        self.scenarios_dir = os.path.join(os.path.dirname(__file__), 'configs', 'scenarios')

    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load a scenario configuration from YAML file."""
        scenario_path = os.path.join(self.scenarios_dir, f"{scenario_name}.yaml")

        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario '{scenario_name}' not found at {scenario_path}")

        with open(scenario_path, 'r') as f:
            scenario = yaml.safe_load(f)

        scenario['name'] = scenario_name
        return scenario

    def execute_scenario(
        self,
        scenario: Dict[str, Any],
        request_data: Dict[str, Any],
        sequence_position: int = 0
    ):
        """Execute a scenario based on the request and current state."""

        scenario_type = scenario.get('type', 'fixed')

        if scenario_type == 'sequence':
            return self._execute_sequence_scenario(scenario, request_data, sequence_position)
        elif scenario_type == 'conditional':
            return self._execute_conditional_scenario(scenario, request_data)
        elif scenario_type == 'probabilistic':
            return self._execute_probabilistic_scenario(scenario, request_data)
        else:
            return self._execute_fixed_scenario(scenario, request_data)

    def _execute_sequence_scenario(
        self,
        scenario: Dict[str, Any],
        request_data: Dict[str, Any],
        position: int
    ):
        """Execute sequence scenario (e.g., fail, fail, succeed)."""

        responses = scenario.get('response_sequence', [])

        if position >= len(responses):
            # Default to last response if we exceed sequence
            position = len(responses) - 1

        response_config = responses[position]
        return self._generate_response_from_config(response_config, request_data)

    def _execute_conditional_scenario(
        self,
        scenario: Dict[str, Any],
        request_data: Dict[str, Any]
    ):
        """Execute conditional scenario based on request content."""

        conditions = scenario.get('conditions', [])
        default_response = scenario.get('default_response', {})

        # Check conditions in order
        for condition in conditions:
            if self._check_condition(condition, request_data):
                return self._generate_response_from_config(condition['response'], request_data)

        # Use default response if no conditions match
        return self._generate_response_from_config(default_response, request_data)

    def _execute_fixed_scenario(
        self,
        scenario: Dict[str, Any],
        request_data: Dict[str, Any]
    ):
        """Execute fixed scenario (always same response)."""

        response_config = scenario.get('response', {})
        return self._generate_response_from_config(response_config, request_data)

    def _check_condition(self, condition: Dict[str, Any], request_data: Dict[str, Any]) -> bool:
        """Check if a condition matches the request."""

        check_type = condition.get('check')

        if check_type == 'json_mode':
            body = request_data.get('body', {})
            response_format = body.get('response_format', {})
            return response_format.get('type') == 'json_object'

        elif check_type == 'temperature_above':
            body = request_data.get('body', {})
            temperature = body.get('temperature', 1.0)
            threshold = condition.get('value', 0.5)
            return temperature > threshold

        elif check_type == 'temperature_equal':
            body = request_data.get('body', {})
            temperature = body.get('temperature', 1.0)
            target = condition.get('value', 1.0)
            return abs(temperature - target) < 0.01  # Allow small floating point differences

        elif check_type == 'model_contains':
            body = request_data.get('body', {})
            model = body.get('model', '')
            text = condition.get('value', '')
            return text in model

        elif check_type == 'message_contains':
            body = request_data.get('body', {})
            messages = body.get('messages', [])
            text = condition.get('value', '').lower()
            for message in messages:
                if text in message.get('content', '').lower():
                    return True
            return False

        return False

    def _generate_response_from_config(
        self,
        response_config: Dict[str, Any],
        request_data: Dict[str, Any]
    ):
        """Generate response based on configuration."""

        response_type = response_config.get('type', 'success')
        body = request_data.get('body', {})
        model = body.get('model', 'faker:default')

        if response_type == 'success':
            content = response_config.get('content', 'Fake successful response')
            tokens = response_config.get('tokens', {'input': 50, 'output': 20})

            # Check if JSON mode requested - either via response_format OR prompt instructions
            response_format = body.get('response_format', {})
            json_requested_via_format = response_format.get('type') == 'json_object'

            # Check if JSON requested via prompt instructions (when response_format is removed)
            json_requested_via_prompt = False
            messages = body.get('messages', [])
            for message in messages:
                content_text = message.get('content', '')
                if 'CRITICAL: You must respond with ONLY a clean JSON object' in content_text:
                    json_requested_via_prompt = True
                    break

            if json_requested_via_format or json_requested_via_prompt:
                # Return JSON response - either use configured json_data or convert content to JSON
                if 'json_data' in response_config:
                    json_data = response_config['json_data']
                else:
                    # Convert text content to a simple JSON object
                    json_data = {"message": content, "test": "parameter_cleanup_success"}

                return self.response_generator.generate_json_response(
                    json_data=json_data,
                    tokens=tokens,
                    model=model,
                    valid=response_config.get('valid_json', True)
                )

            return self.response_generator.generate_openai_response(
                content=content,
                tokens=tokens,
                model=model
            )

        elif response_type == 'rate_limit':
            retry_after = response_config.get('retry_after', 5)
            return self.response_generator.generate_rate_limit_error(retry_after)

        elif response_type == 'auth_error':
            return self.response_generator.generate_auth_error()

        elif response_type == 'timeout':
            return self.response_generator.generate_timeout_response()

        elif response_type == 'overloaded':
            return self.response_generator.generate_overloaded_error()

        elif response_type == 'json_error':
            # Return malformed JSON to test error handling
            return self.response_generator.generate_json_response(
                json_data={"test": "data"},
                model=model,
                valid=False
            )

        elif response_type == 'reasoning':
            content = response_config.get('content', 'Response with reasoning')
            tokens = response_config.get('tokens', {'input': 50, 'output': 20, 'reasoning': 10})
            reasoning_format = response_config.get('reasoning_format', 'default')

            return self.response_generator.generate_reasoning_response_with_format(
                content=content,
                tokens=tokens,
                model=model,
                reasoning_format=reasoning_format
            )

        elif response_type == 'streaming':
            chunks = response_config.get('chunks', [{"delta": {"content": "Streaming response"}}])
            tokens = response_config.get('tokens', {'input': 50, 'output': 20})

            return self.response_generator.generate_streaming_response(
                chunks=chunks,
                tokens=tokens,
                model=model
            )

        else:
            # Default to success response
            return self.response_generator.generate_openai_response(
                content="Unknown response type, defaulting to success",
                model=model
            )