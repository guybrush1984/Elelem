#!/usr/bin/env python3
"""
Telelem with Dual Mode Support - Direct Elelem library or API server

This version preserves ALL existing Telelem capabilities and adds:
- --elelem-server flag for using Elelem through API server
- Automatic detection of direct vs API mode
- Full compatibility with all existing features

Usage:
    # Direct mode (default - uses Elelem library)
    python telelem_dual_mode.py --prompt prompt.prompt --response output.json --llm groq:openai/gpt-oss-120b

    # API mode (via Elelem server)
    python telelem_dual_mode.py --elelem-server localhost:8000 --prompt prompt.prompt --response output.json --llm groq:openai/gpt-oss-120b

    # Batch mode with API server
    python telelem_dual_mode.py --elelem-server localhost:8000 --batch batch.json --output results.json
"""

import argparse
import asyncio
import json
import logging
import os
import requests
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the src directory to the path so we can import elelem
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import Elelem for direct mode
from elelem import Elelem


# Setup logging
def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('telelem.log')
        ]
    )
    return logging.getLogger('telelem')


def parse_prompt_file(file_path: str) -> tuple[str, str]:
    """Parse a YAML prompt file into system and user messages.

    Expected format:
    system: |
      <system message content>
    user: |
      <user message content>
    """
    import yaml
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    system_content = data.get('system', '').strip()
    user_content = data.get('user', '').strip()

    if not system_content and not user_content:
        raise ValueError("No valid system/user sections found in YAML file")

    return system_content, user_content


def format_duration(seconds: float) -> str:
    """Format duration in seconds only."""
    return f"{seconds:.2f}s"


class ElememClient:
    """Unified client that supports both direct and API modes."""

    def __init__(self, server_url: Optional[str] = None):
        """
        Initialize the Elelem client.

        Args:
            server_url: If provided, use API mode. Format: "localhost:8000" or "http://localhost:8000"
        """
        self.server_url = server_url
        self.mode = "api" if server_url else "direct"
        self.logger = logging.getLogger('telelem.client')
        self._direct_client = None
        self._api_client = None

        if self.mode == "direct":
            # Direct mode - use Elelem library
            self._direct_client = Elelem()
            self.logger.info("Using direct Elelem library mode")
        else:
            # API mode - use OpenAI SDK pointed at Elelem server
            try:
                import openai
            except ImportError:
                raise ImportError("OpenAI SDK required for API mode. Install with: pip install openai")

            # Format the server URL
            if not server_url.startswith(('http://', 'https://')):
                server_url = f"http://{server_url}"
            if not server_url.endswith('/v1'):
                server_url = f"{server_url}/v1"

            self._api_client = openai.AsyncOpenAI(
                base_url=server_url,
                api_key="not-needed"  # Elelem server handles provider keys
            )
            self.logger.info(f"Using API mode with server: {server_url}")

    @property
    def config(self):
        """Access config for direct mode compatibility."""
        if self.mode == "direct":
            return self._direct_client.config
        else:
            # Create a mock config object for API mode
            class MockConfig:
                def get_model_config(self, model):
                    # Basic implementation for API mode
                    return {'candidates': []}

                def get_candidate_timeout(self, candidate, config):
                    return 120  # Default timeout

            return MockConfig()

    async def create_chat_completion(self, messages: list, model: str, **kwargs):
        """
        Create a chat completion using either direct or API mode.

        Returns a response object compatible with both modes.
        """
        if self.mode == "direct":
            # Direct mode - use Elelem library
            response = await self._direct_client.create_chat_completion(
                messages=messages,
                model=model,
                **kwargs
            )
            return response
        else:
            # API mode - use OpenAI SDK
            # Remove Elelem-specific parameters that OpenAI SDK doesn't support
            api_kwargs = kwargs.copy()
            api_kwargs.pop('tags', None)  # Remove tags parameter

            response = await self._api_client.chat.completions.create(
                messages=messages,
                model=model,
                **api_kwargs
            )
            return response

    def get_stats_by_tag(self, tag: str):
        """Get statistics by tag (only available in direct mode)."""
        if self.mode == "direct":
            return self._direct_client.get_stats_by_tag(tag)
        else:
            # Return empty stats for API mode
            return {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'total_input_cost_usd': 0.0,
                'total_output_cost_usd': 0.0,
                'total_cost_usd': 0.0,
                'reasoning_tokens': 0,
                'retry_analytics': {}
            }


async def run_telelem_test(prompt_file: str, response_file: str, model: str,
                          elelem_server: Optional[str] = None, logger=None, **kwargs):
    """Run a telelem test with the specified parameters - now with dual mode support."""

    if logger is None:
        logger = logging.getLogger('telelem')

    mode_str = f"API server ({elelem_server})" if elelem_server else "Direct library"
    logger.info(f"Starting telelem test - mode: {mode_str}, model: {model}, prompt: {prompt_file}")

    print(f"🚀 Telelem Test Starting")
    if elelem_server:
        print(f"   Mode: API server ({elelem_server})")
    print(f"   Prompt: {prompt_file}")
    print(f"   Model: {model}")
    print(f"   Output: {response_file}")

    # Parse temperature and other parameters
    temperature = kwargs.get('temperature', 1.0)
    json_mode = kwargs.get('json', False)

    if temperature != 1.0:
        print(f"   Temperature: {temperature}")
    if json_mode:
        print(f"   JSON mode: enabled")

    print("-" * 50)

    # Parse the prompt file
    try:
        system_content, user_content = parse_prompt_file(prompt_file)
        print(f"✅ Parsed prompt file:")
        print(f"   System message: {len(system_content)} characters")
        print(f"   User message: {len(user_content)} characters")
    except Exception as e:
        print(f"❌ Failed to parse prompt file: {e}")
        return 1

    # Initialize Elelem client (now with dual mode support)
    try:
        elelem = ElememClient(server_url=elelem_server)
        print(f"✅ Initialized Elelem ({elelem.mode} mode)")

        # Check if model is available (only works fully in direct mode)
        if elelem.mode == "direct":
            try:
                config = elelem.config.get_model_config(model)
                candidates = config['candidates']
                print(f"✅ Model '{model}' found with {len(candidates)} candidate(s)")

                # Show candidates for virtual models
                if len(candidates) > 1:
                    print("   Candidates:")
                    for i, candidate in enumerate(candidates):
                        provider = candidate['provider']
                        timeout = elelem.config.get_candidate_timeout(candidate, config)
                        print(f"   {i+1}. {provider} (timeout={timeout}s)")
            except Exception as e:
                raise e
        else:
            # In API mode, model config details are not available
            print(f"✅ Model '{model}' (config details not available in API mode)")

    except Exception as e:
        print(f"❌ Failed to initialize Elelem or find model: {e}")
        return 1

    # Prepare messages
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if user_content:
        messages.append({"role": "user", "content": user_content})

    # Prepare API parameters
    api_params = {
        "temperature": temperature
    }
    if json_mode:
        api_params["response_format"] = {"type": "json_object"}

    # Add tags for direct mode
    if elelem.mode == "direct":
        api_params["tags"] = ["telelem_test"]

    print(f"\n🔄 Making request...")
    start_time = time.time()

    # Make the request
    try:
        response = await elelem.create_chat_completion(
            messages=messages,
            model=model,
            **api_params
        )

        duration = time.time() - start_time

        # Extract response details
        logger.debug(f"Response object type: {type(response)}")

        # Safely extract content - handle both dict (direct mode) and object (API mode) responses
        content = None
        if isinstance(response, dict):
            # Direct mode returns dict
            logger.debug(f"Response choices (dict): {response.get('choices', [])}")
            choices = response.get('choices', [])
            if choices and len(choices) > 0:
                first_choice = choices[0]
                logger.debug(f"First choice (dict): {first_choice}")
                if 'message' in first_choice and first_choice['message']:
                    content = first_choice['message'].get('content')
                    logger.debug(f"Message content type: {type(content)}, value: {content[:100] if content else 'None'}")
        elif response and hasattr(response, 'choices'):
            # API mode returns OpenAI object
            logger.debug(f"Response choices (object): {response.choices}")
            if response.choices:
                first_choice = response.choices[0]
                logger.debug(f"First choice (object): {first_choice}")
                if hasattr(first_choice, 'message') and first_choice.message:
                    content = first_choice.message.content
                    logger.debug(f"Message content type: {type(content)}, value: {content[:100] if content else 'None'}")

        if content is None:
            logger.warning("Response content is None - this might cause issues")
            content = ""  # Default to empty string to avoid NoneType errors

        # Extract usage information - handle both dict and object responses
        if isinstance(response, dict):
            # Direct mode returns dict
            usage = response.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
        else:
            # API mode returns object
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)

        # Get reasoning tokens from elelem's statistics (only in direct mode)
        stats = elelem.get_stats_by_tag("telelem_test")
        reasoning_tokens = stats.get('reasoning_tokens', 0)

        actual_output_tokens = output_tokens - reasoning_tokens
        total_tokens = input_tokens + output_tokens

        print(f"✅ Request completed in {format_duration(duration)}")
        print(f"   Input tokens: {input_tokens:,}")
        print(f"   Output tokens: {output_tokens:,}")
        if reasoning_tokens > 0:
            print(f"   Reasoning tokens: {reasoning_tokens:,}")
            print(f"   Actual output tokens: {actual_output_tokens:,}")
        print(f"   Total tokens: {total_tokens:,}")

        # Calculate and display tokens per second
        if duration > 0 and output_tokens > 0:
            total_generation_speed = output_tokens / duration
            if reasoning_tokens > 0 and actual_output_tokens > 0:
                # Calculate speed for actual output only (excluding reasoning)
                actual_generation_speed = actual_output_tokens / duration
                print(f"   Generation speed: {total_generation_speed:.1f} tokens/s (total), {actual_generation_speed:.1f} tokens/s (actual output)")
            else:
                print(f"   Generation speed: {total_generation_speed:.1f} tokens/s")

        # Calculate and display cost (only in direct mode)
        total_cost = stats.get('total_cost_usd', 0.0)
        if elelem.mode == "direct" and total_cost > 0:
            print(f"   Total cost: ${total_cost:.6f} USD")

            # Break down cost per 1k tokens
            if input_tokens > 0:
                input_cost_per_1k = (stats.get('total_input_cost_usd', 0.0) / input_tokens) * 1000
                print(f"   Cost per input: ${input_cost_per_1k:.4f}/1k tokens")
            if output_tokens > 0:
                output_cost_per_1k = (stats.get('total_output_cost_usd', 0.0) / output_tokens) * 1000
                print(f"   Cost per output: ${output_cost_per_1k:.4f}/1k tokens")

                # If we have reasoning tokens, also show cost for actual output
                if reasoning_tokens > 0 and actual_output_tokens > 0:
                    actual_output_cost_per_1k = (total_cost / actual_output_tokens) * 1000
                    print(f"   Cost per actual output: ${actual_output_cost_per_1k:.4f}/1k tokens")
        elif elelem.mode == "api":
            print(f"   ⚠️  Cost tracking not available in API mode")

        # Check for provider info (virtual models in direct mode)
        if isinstance(response, dict):
            provider = response.get('provider')
        else:
            provider = getattr(response, 'provider', None)

        if provider:
            print(f"   Provider used: {provider}")

        # Check for candidate iterations (direct mode only)
        retry_analytics = stats.get('retry_analytics', {})
        if retry_analytics.get('candidate_iterations', 0) > 0:
            print(f"   🔄 Candidate iterations: {retry_analytics['candidate_iterations']}")
        if retry_analytics.get('temperature_reductions', 0) > 0:
            print(f"   🌡️  Temperature reductions: {retry_analytics['temperature_reductions']}")

        # Validate JSON if in JSON mode
        if json_mode and content:
            try:
                json.loads(content)
                print(f"   ✅ Valid JSON response")
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON response: {e}")

        print(f"\n📄 Response preview (first 200 chars):")
        preview = content[:200] if content else "(empty)"
        print(f"   {preview}{'...' if len(content or '') > 200 else ''}")

        # Parse content as JSON if possible for prettier output
        parsed_content = content
        if content and (json_mode or content.strip().startswith('{')):
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                # Keep as string if not valid JSON
                parsed_content = content

        # Save response to file
        try:
            response_path = Path(response_file)
            response_path.parent.mkdir(parents=True, exist_ok=True)

            # Create comprehensive output
            output_data = {
                "request": {
                    "model": model,
                    "temperature": temperature,
                    "json_mode": json_mode,
                    "prompt_file": prompt_file,
                    "elelem_mode": elelem.mode,
                    "elelem_server": elelem_server
                },
                "response": {
                    "content": parsed_content,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                },
                "metrics": {
                    "duration_seconds": duration,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": total_cost,
                    "provider_used": getattr(response, 'provider', None)
                },
                "analytics": retry_analytics
            }

            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Response saved to: {response_file}")

        except Exception as e:
            print(f"❌ Failed to save response: {e}")
            return 1

        return 0

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        error_type = type(e).__name__

        # Log detailed error information
        logger.error(f"Request failed - Error type: {error_type}, Message: {error_msg}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        print(f"❌ Request failed after {format_duration(duration)}: {e}")

        # Still save error information
        try:
            response_path = Path(response_file)
            response_path.parent.mkdir(parents=True, exist_ok=True)

            error_data = {
                "request": {
                    "model": model,
                    "temperature": temperature,
                    "json_mode": json_mode,
                    "prompt_file": prompt_file,
                    "elelem_mode": elelem_server is not None,
                    "elelem_server": elelem_server
                },
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                },
                "metrics": {
                    "duration_seconds": duration,
                    "success": False
                }
            }

            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)

            print(f"🗂️  Error details saved to: {response_file}")

        except Exception as save_error:
            print(f"❌ Failed to save error details: {save_error}")

        return 1


def generate_csv_summary(summary, csv_file, elelem_server):
    """Generate CSV summary with model performance metrics."""
    import csv

    # Get model configuration data for static info (costs, metadata)
    if elelem_server:
        # API mode - get model list
        response = requests.get(f"http://{elelem_server}/v1/models")
        models_data = response.json() if response.status_code == 200 else {"data": []}
        models_config = {m["id"]: m for m in models_data.get("data", [])}
    else:
        # Direct mode - access config
        from src.elelem import Elelem
        elelem = Elelem()
        models_config = {}
        for model_key, model_config in elelem._models.items():
            cost_config = model_config.get("cost", {})
            cost_data = {}
            if isinstance(cost_config, dict) and cost_config:
                cost_data = {
                    "input_cost_per_1m": cost_config.get("input_cost_per_1m", 0),
                    "output_cost_per_1m": cost_config.get("output_cost_per_1m", 0),
                    "currency": cost_config.get("currency", "USD")
                }

            models_config[model_key] = {
                "nickname": model_config.get("display_metadata", {}).get("model_nickname", model_key),
                "owned_by": model_config.get("display_metadata", {}).get("model_owner", "unknown"),
                "provider": model_config.get("provider", "unknown"),
                "license": model_config.get("display_metadata", {}).get("license", "unknown"),
                "reasoning": model_config.get("display_metadata", {}).get("reasoning", "no"),
                "cost": cost_data
            }

    # Prepare CSV data
    csv_rows = []
    for model_key, model_stats in summary.get("models", {}).items():
        if not model_stats:
            continue

        # Get static model info
        model_info = models_config.get(model_key, {})

        # Extract performance metrics
        total_calls = model_stats.get("total_calls", 0)
        total_successes = model_stats.get("total_successes", 0)
        success_rate = (total_successes / total_calls * 100) if total_calls > 0 else 0

        output_tokens = model_stats.get("output_tokens", {}).get("total", 0)
        reasoning_tokens = model_stats.get("reasoning_tokens", {}).get("total", 0)
        total_duration = model_stats.get("duration_seconds", {}).get("total", 0)

        # Calculate token rates (tokens per second)
        output_token_rate = output_tokens / total_duration if total_duration > 0 else 0
        real_output_tokens = output_tokens - reasoning_tokens
        real_output_token_rate = real_output_tokens / total_duration if total_duration > 0 else 0

        # Get costs from static config and calculate real cost per output token
        cost_config = model_info.get("cost", {})
        input_cost_per_1m = cost_config.get("input_cost_per_1m", 0) if isinstance(cost_config, dict) else 0
        output_cost_per_1m = cost_config.get("output_cost_per_1m", 0) if isinstance(cost_config, dict) else 0

        # Calculate real cost per output token (output cost only / real output tokens)
        output_cost_usd = model_stats.get("output_cost_usd", {}).get("total", 0)
        real_cost_per_output_token = (output_cost_usd / real_output_tokens) if real_output_tokens > 0 else 0

        # Calculate reasoning token ratio (reasoning_tokens / output_tokens)
        reasoning_token_ratio = (reasoning_tokens / output_tokens) if output_tokens > 0 else 0

        csv_rows.append({
            "model_name": model_info.get("nickname", model_key),
            "model_owner": model_info.get("owned_by", "unknown"),
            "model_provider": model_info.get("provider", "unknown"),
            "license": model_info.get("license", "unknown"),
            "reasoning": model_info.get("reasoning", "no"),
            "cost_per_1m_input": input_cost_per_1m,
            "cost_per_1m_output": output_cost_per_1m,
            "success_rate_percent": round(success_rate, 1),
            "output_token_rate": round(output_token_rate, 1),
            "real_output_token_rate": round(real_output_token_rate, 1),
            "computed_cost_per_1m_actual_output_token": round(real_cost_per_output_token * 1000000, 6),  # Convert to cost per 1M tokens
            "reasoning_token_ratio": round(reasoning_token_ratio, 3)  # Ratio as percentage (0.0 to 1.0)
        })

    # Write CSV file
    if csv_rows:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "model_name", "model_owner", "model_provider", "license", "reasoning",
                "cost_per_1m_input", "cost_per_1m_output", "success_rate_percent",
                "output_token_rate", "real_output_token_rate", "computed_cost_per_1m_actual_output_token",
                "reasoning_token_ratio"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


async def run_batch_tests(batch_file: str, output_file: str, elelem_server: Optional[str] = None, **kwargs):
    """Run batch tests with multiple models and prompts - now with dual mode support."""

    print(f"🚀 Telelem Batch Test Starting")
    if elelem_server:
        print(f"   Mode: API server ({elelem_server})")
    print(f"   Batch config: {batch_file}")
    print(f"   Output: {output_file}")
    print("-" * 50)

    # Load batch configuration
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_config = json.load(f)

        models = batch_config.get('models', [])
        prompts = batch_config.get('prompts', [])

        if not models:
            print("❌ No models specified in batch config")
            return 1
        if not prompts:
            print("❌ No prompts specified in batch config")
            return 1

        print(f"✅ Loaded batch config: {len(models)} models × {len(prompts)} prompts = {len(models) * len(prompts)} tests")

    except Exception as e:
        print(f"❌ Failed to load batch config: {e}")
        return 1

    # Prepare output directory
    output_path = Path(output_file)
    output_dir = output_path.parent / output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track test start time and models for metrics
    from datetime import datetime
    test_start_time = datetime.now()
    tested_models = set()

    # Run all combinations
    results = []
    total_tests = len(models) * len(prompts)
    test_num = 0

    for model in models:
        tested_models.add(model)  # Track which models we test
        for prompt in prompts:
            test_num += 1
            print(f"\n📊 Test {test_num}/{total_tests}: {model} × {prompt}")
            print("-" * 30)

            # Generate output filename
            model_safe = model.replace(':', '_').replace('/', '_')
            prompt_safe = Path(prompt).stem
            response_file = output_dir / f"{model_safe}_{prompt_safe}.json"

            # Run the test - metrics are automatically tracked by elelem
            try:
                exit_code = await run_telelem_test(
                    prompt_file=prompt,
                    response_file=str(response_file),
                    model=model,
                    elelem_server=elelem_server,
                    **kwargs
                )
                success = exit_code == 0
            except Exception as e:
                success = False
                print(f"❌ Test failed: {e}")

            # Simple result tracking - no need for manual metrics
            results.append({
                "model": model,
                "prompt": prompt,
                "success": success,
                "response_file": str(response_file)
            })

    # Generate summary using metrics API
    summary = {
        "batch_config": batch_file,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elelem_mode": "api" if elelem_server else "direct",
        "elelem_server": elelem_server,
        "test_start_time": test_start_time.isoformat(),
        "models": {}
    }

    print(f"\n📊 Batch Test Complete")
    print("-" * 50)

    # Get overall summary (no tags = all metrics since test_start_time)
    if elelem_server:
        # API mode
        response = requests.get(
            f"http://{elelem_server}/v1/metrics/summary",
            params={"start": test_start_time.isoformat()}
        )
        if response.status_code == 200:
            overall_summary = response.json()
            print(f"[DEBUG] Overall summary response: {overall_summary}")
        else:
            print(f"[DEBUG] Failed to get overall summary: {response.status_code} - {response.text}")
            overall_summary = {}
    else:
        # Direct mode - need to initialize elelem to access metrics
        from src.elelem import Elelem
        elelem = Elelem()
        overall_summary = elelem._metrics_store.get_summary(start=test_start_time)
        print(f"[DEBUG] Overall summary (direct): {overall_summary}")

    # Get per-model summaries
    for model in sorted(tested_models):
        if elelem_server:
            response = requests.get(
                f"http://{elelem_server}/v1/metrics/summary",
                params={
                    "tags": f"model:{model}",
                    "start": test_start_time.isoformat()
                }
            )
            if response.status_code == 200:
                model_summary = response.json()
                print(f"[DEBUG] Model {model} summary: {model_summary}")
            else:
                print(f"[DEBUG] Failed to get {model} summary: {response.status_code} - {response.text}")
                model_summary = {}
        else:
            model_summary = elelem._metrics_store.get_summary(
                tags=[f"model:{model}"],
                start=test_start_time
            )
            print(f"[DEBUG] Model {model} summary (direct): {model_summary}")

        summary["models"][model] = model_summary

    # Store overall summary and print results
    summary["overall"] = overall_summary

    if overall_summary:
        print(f"✅ Overall: {overall_summary['total_successes']}/{overall_summary['total_calls']} successful")
        print(f"💵 Total cost: ${overall_summary['total_cost_usd']['total']:.6f}")

    # Print per-model breakdown
    for model in sorted(tested_models):
        model_stats = summary["models"].get(model, {})
        if model_stats:
            print(f"\n📊 {model}: {model_stats['total_successes']}/{model_stats['total_calls']} "
                  f"(${model_stats['total_cost_usd']['total']:.6f})")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Batch results saved to: {output_file}")

        # Generate CSV summary
        csv_file = output_file.replace('.json', '_summary.csv')
        generate_csv_summary(summary, csv_file, elelem_server)
        print(f"📊 CSV summary saved to: {csv_file}")

        return 0
    except Exception as e:
        print(f"❌ Failed to save batch results: {e}")
        return 1


def main():
    """Main function to parse arguments and run the test - now with dual mode support."""
    parser = argparse.ArgumentParser(
        description="Test Elelem with structured prompt files (dual mode: direct library or API server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single test - Direct mode (default)
  %(prog)s --prompt test.prompt --response test.json --llm groq:openai/gpt-oss-120b

  # Single test - API mode
  %(prog)s --elelem-server localhost:8000 --prompt test.prompt --response test.json --llm groq:openai/gpt-oss-120b

  # Batch test - Direct mode
  %(prog)s --batch batch.json --output results.json

  # Batch test - API mode
  %(prog)s --elelem-server localhost:8000 --batch batch.json --output results.json

  # With custom temperature and JSON mode
  %(prog)s --prompt test.prompt --response test.json --llm virtual:gpt-oss-120b --temperature 0.7 --json

  # Using remote Elelem server
  %(prog)s --elelem-server https://api.example.com --prompt test.prompt --response test.json --llm virtual:gpt-oss-120b-quick
        """
    )

    # NEW: Elelem server argument for dual mode
    parser.add_argument(
        "--elelem-server",
        help="Use Elelem API server instead of direct library. Format: localhost:8000 or https://api.example.com"
    )

    # Batch mode arguments
    parser.add_argument(
        "--batch",
        help="Path to JSON file with models and prompts lists for batch testing"
    )
    parser.add_argument(
        "--output",
        help="Path to save batch test results (used with --batch)"
    )

    # Single test mode arguments
    parser.add_argument(
        "--prompt",
        help="Path to .prompt file with Role: system/user sections"
    )
    parser.add_argument(
        "--response",
        help="Path to save the response JSON file"
    )
    parser.add_argument(
        "--llm",
        help="Model to use (e.g., virtual:gpt-oss-120b, groq:openai/gpt-oss-120b, openai:gpt-4.1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation (default: 1.0)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Enable JSON mode (response_format=json_object)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.debug)

    if args.debug:
        print("🔍 Debug mode enabled - check telelem.log for details")

    # Show mode information
    if args.elelem_server:
        print(f"🌐 Using Elelem API server: {args.elelem_server}")
    else:
        print(f"📚 Using direct Elelem library mode")

    # Determine which mode to run
    if args.batch and args.output:
        # Batch mode
        if not os.path.exists(args.batch):
            print(f"❌ Batch config file not found: {args.batch}")
            return 1

        try:
            exit_code = asyncio.run(run_batch_tests(
                batch_file=args.batch,
                output_file=args.output,
                elelem_server=args.elelem_server,
                temperature=args.temperature,
                json=args.json
            ))
            return exit_code
        except KeyboardInterrupt:
            print("\n❌ Batch test interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Batch test failed: {e}")
            logger.error(f"Batch test error: {e}")
            logger.debug(traceback.format_exc())
            return 1

    elif args.prompt and args.response and args.llm:
        # Single test mode
        if not os.path.exists(args.prompt):
            print(f"❌ Prompt file not found: {args.prompt}")
            return 1

        try:
            exit_code = asyncio.run(run_telelem_test(
                prompt_file=args.prompt,
                response_file=args.response,
                model=args.llm,
                elelem_server=args.elelem_server,
                logger=logger,
                temperature=args.temperature,
                json=args.json
            ))
            return exit_code
        except KeyboardInterrupt:
            print("\n❌ Test interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Test failed: {e}")
            logger.error(f"Test error: {e}")
            logger.debug(traceback.format_exc())
            return 1
    else:
        # Show help if no valid mode
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())