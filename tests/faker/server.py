"""HTTP mock server for Elelem testing."""

import json
import time
import uuid
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify
import threading
import yaml

from .validator import RequestAnalyzer
from .response_generator import ResponseGenerator
from .scenarios import ScenarioManager


class ModelFaker:
    """Main faker server that provides OpenAI-compatible API endpoints."""

    def __init__(self, port: int = 8899, host: str = "localhost"):
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        self.server_thread = None
        self.running = False

        # Core components
        self.request_analyzer = RequestAnalyzer()
        self.response_generator = ResponseGenerator()
        self.scenario_manager = ScenarioManager()

        # State
        self.current_scenario = None
        self.request_count = 0
        self.sequence_position = 0

        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes for OpenAI compatibility."""
        self.app.route('/v1/chat/completions', methods=['POST'])(self.chat_completions)
        self.app.route('/v1/models', methods=['GET'])(self.list_models)
        self.app.route('/health', methods=['GET'])(self.health)

        # Faker control endpoints
        self.app.route('/faker/scenario', methods=['POST'])(self.set_scenario)
        self.app.route('/faker/requests', methods=['GET'])(self.get_requests)
        self.app.route('/faker/reset', methods=['POST'])(self.reset_state)
        self.app.route('/shutdown', methods=['POST'])(self.shutdown)

    def health(self):
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "faker": True,
            "scenario": self.current_scenario['name'] if self.current_scenario else None,
            "requests_received": self.request_count
        })

    def list_models(self):
        """List available fake models."""
        if self.current_scenario and 'models' in self.current_scenario:
            models = []
            for model_name, model_config in self.current_scenario['models'].items():
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": model_config.get('provider', 'faker')
                })

            return jsonify({
                "object": "list",
                "data": models
            })

        # Default models if no scenario loaded
        return jsonify({
            "object": "list",
            "data": [{
                "id": "faker:default",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "faker"
            }]
        })

    def chat_completions(self):
        """Handle chat completion requests."""
        try:
            # Increment request count
            self.request_count += 1

            # Analyze and capture request
            analyzed_request = self.request_analyzer.analyze_request(request)

            # Get request data
            data = request.get_json()
            model = data.get('model')

            # Execute current scenario
            if self.current_scenario:
                response = self.scenario_manager.execute_scenario(
                    self.current_scenario,
                    analyzed_request,
                    self.sequence_position
                )

                # Update sequence position for sequential scenarios
                if self.current_scenario.get('type') == 'sequence':
                    self.sequence_position += 1

                return response

            # Default response if no scenario
            return self.response_generator.generate_openai_response(
                content="Default fake response - no scenario configured",
                tokens={'input': 10, 'output': 8}
            )

        except Exception as e:
            return jsonify({
                "error": {
                    "message": f"Faker error: {str(e)}",
                    "type": "faker_error",
                    "code": "internal_error"
                }
            }), 500

    def set_scenario(self):
        """Set current test scenario."""
        try:
            scenario_name = request.json.get('scenario')
            self.current_scenario = self.scenario_manager.load_scenario(scenario_name)
            self.sequence_position = 0  # Reset sequence

            return jsonify({
                "status": "ok",
                "scenario": scenario_name,
                "loaded": True
            })
        except Exception as e:
            return jsonify({
                "error": str(e)
            }), 400

    def get_requests(self):
        """Get captured request log."""
        return jsonify({
            "total_requests": self.request_count,
            "requests": self.request_analyzer.get_captured_requests()
        })

    def reset_state(self):
        """Reset faker state."""
        self.request_count = 0
        self.sequence_position = 0
        self.request_analyzer.clear_requests()

        # Only return JSON response if called within Flask request context
        try:
            return jsonify({
                "status": "reset",
                "requests_cleared": True
            })
        except RuntimeError:
            # Not in request context, just return None
            return None

    def shutdown(self):
        """Shutdown endpoint for graceful server termination."""
        print(f"[FAKER] Shutdown endpoint called on port {self.port}")
        self.running = False

        # Use Werkzeug's shutdown functionality
        func = request.environ.get('werkzeug.server.shutdown')
        if func is not None:
            print(f"[FAKER] Calling werkzeug shutdown function")
            func()
            print(f"[FAKER] Werkzeug shutdown function called")
        else:
            print(f"[FAKER] No werkzeug shutdown function available")

        print(f"[FAKER] Shutdown endpoint returning response")
        return jsonify({"status": "shutting down"})

    def start(self):
        """Start the faker server in background thread."""
        if self.running:
            return

        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()

        # Wait for server to be ready
        import time
        time.sleep(0.5)

    def _run_server(self):
        """Run the Flask server."""
        print(f"[FAKER] Starting Flask server on {self.host}:{self.port}")
        try:
            # Use werkzeug's make_server for better control
            from werkzeug.serving import make_server
            self.server = make_server(
                self.host,
                self.port,
                self.app,
                threaded=True
            )
            self.server.serve_forever()
            print(f"[FAKER] Flask server stopped normally on port {self.port}")
        except Exception as e:
            print(f"[FAKER] Flask server error on port {self.port}: {e}")
        finally:
            print(f"[FAKER] Flask server thread exiting on port {self.port}")

    def stop(self):
        """Stop the faker server."""
        print(f"[FAKER] Stopping server on port {self.port}")
        self.running = False

        # Shutdown the Werkzeug server directly
        if hasattr(self, 'server'):
            print(f"[FAKER] Shutting down Werkzeug server directly")
            try:
                self.server.shutdown()
                print(f"[FAKER] Werkzeug server.shutdown() called")
            except Exception as e:
                print(f"[FAKER] Error calling server.shutdown(): {e}")

        # Wait for thread to finish
        if hasattr(self, 'server_thread') and self.server_thread.is_alive():
            print(f"[FAKER] Waiting for server thread to join (timeout=3s)")
            self.server_thread.join(timeout=3)

            # Check if still running
            if self.server_thread.is_alive():
                print(f"[FAKER] WARNING: Server thread still alive after 3s timeout on port {self.port}")
            else:
                print(f"[FAKER] Server thread joined successfully on port {self.port}")

        # Reset state
        if hasattr(self, 'server_thread'):
            del self.server_thread
            print(f"[FAKER] Server thread deleted")
        if hasattr(self, 'server'):
            del self.server
            print(f"[FAKER] Server object deleted")

        print(f"[FAKER] Stop completed for port {self.port}")

    def configure_scenario(self, scenario_name: str):
        """Configure faker with a specific scenario."""
        self.current_scenario = self.scenario_manager.load_scenario(scenario_name)
        self.sequence_position = 0

    # Test assertion methods
    def assert_request_count(self, expected_count: int):
        """Assert specific number of requests received."""
        actual = len(self.request_analyzer.captured_requests)
        assert actual == expected_count, f"Expected {expected_count} requests, got {actual}"

    def assert_retry_pattern(self, expected_temperatures: List[float]):
        """Verify retry behavior with temperature reduction."""
        temps = []
        for req in self.request_analyzer.captured_requests:
            body = req.get('body', {})
            temps.append(body.get('temperature'))

        assert temps == expected_temperatures, f"Expected temperature pattern {expected_temperatures}, got {temps}"

    def assert_parameter_present(self, param_name: str):
        """Assert parameter is present in requests."""
        for req in self.request_analyzer.captured_requests:
            body = req.get('body', {})
            assert param_name in body, f"Parameter {param_name} not found in request"

    def assert_no_forbidden_headers(self):
        """Assert no forbidden headers were sent."""
        self.request_analyzer.validate_no_forbidden_headers()

    def assert_temperature_reduction_occurred(self):
        """Assert temperature reduction pattern in requests."""
        temps = []
        for req in self.request_analyzer.captured_requests:
            body = req.get('body', {})
            if 'temperature' in body:
                temps.append(body['temperature'])

        if len(temps) > 1:
            # Check that temperatures are decreasing
            for i in range(1, len(temps)):
                assert temps[i] < temps[i-1], f"Temperature should decrease in retries: {temps}"