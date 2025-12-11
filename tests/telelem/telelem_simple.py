#!/usr/bin/env python3
"""
Simplified Telelem - Clean dual-mode test runner for Elelem

Usage:
    # Direct library mode (default)
    python telelem_simple.py --prompt test.prompt --model groq:openai/gpt-oss-120b

    # API server mode
    python telelem_simple.py --server localhost:8000 --prompt test.prompt --model groq:openai/gpt-oss-120b

    # Batch mode
    python telelem_simple.py --batch batch.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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


class TelelemClient:
    """Unified client for both direct library and API server modes."""

    def __init__(self, server_url: Optional[str] = None):
        """Initialize client in either direct or API mode."""
        self.server_url = server_url
        self.mode = "api" if server_url else "direct"
        self._direct_client = None
        self._api_client = None

        if self.mode == "direct":
            # Direct mode - import and initialize Elelem
            from elelem import Elelem
            self._direct_client = Elelem()
            print(f"✅ Initialized Elelem in direct library mode")
        else:
            # API mode - use OpenAI SDK
            try:
                import openai  # type: ignore
                import requests
            except ImportError:
                raise ImportError("OpenAI SDK and requests required for API mode. Install with: uv add openai requests")

            # Normalize server URL
            if not server_url.startswith(('http://', 'https://')):
                self.server_url = f"http://{server_url}"

            # Check health endpoint first
            try:
                health_response = requests.get(f"{self.server_url}/health", timeout=5)
                health_response.raise_for_status()
                health_data = health_response.json()

                if health_data.get("status") != "healthy":
                    raise Exception(f"Server unhealthy: {health_data}")

                print(f"✅ Server health check passed")

            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to connect to Elelem server at {self.server_url}")
                print(f"   Error: {e}")
                raise Exception(f"Cannot connect to Elelem server: {e}")

            # Initialize OpenAI client
            api_base_url = f"{self.server_url}/v1"
            self._api_client = openai.AsyncOpenAI(
                base_url=api_base_url,
                api_key="not-needed",  # Elelem server handles provider keys
                max_retries=0  # Let Elelem server handle all retries
            )

            print(f"✅ Connected to Elelem server at {self.server_url}")

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs):
        """Create a chat completion - unified interface for both modes."""

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
            # Remove Elelem-specific params that OpenAI SDK doesn't support
            api_kwargs = kwargs.copy()
            api_kwargs.pop('tags', None)  # Remove tags parameter

            response = await self._api_client.chat.completions.create(
                messages=messages,
                model=model,
                **api_kwargs
            )

            # Return response object directly to preserve elelem_metrics
            return response

    def list_models(self) -> List[Dict]:
        """Get list of available models."""

        if self.mode == "direct":
            # Direct mode - access internal models
            models = []
            for model_key, config in self._direct_client._models.items():
                models.append({
                    "id": model_key,
                    "nickname": config.get("display_metadata", {}).get("model_nickname", model_key),
                    "provider": config.get("provider", "unknown"),
                    "owned_by": config.get("display_metadata", {}).get("model_owner", "unknown")
                })
            return models
        else:
            # API mode - use OpenAI SDK
            import asyncio
            async def _get_models():
                response = await self._api_client.models.list()
                return [{"id": model.id, "owned_by": getattr(model, 'owned_by', 'unknown')} for model in response.data]

            return asyncio.run(_get_models())

    def get_metrics(self, tags: Optional[List[str]] = None, start_time: Optional[datetime] = None) -> Dict:
        """Get metrics - properly handles the same instance for direct mode."""

        if self.mode == "direct":
            # Direct mode - use the SAME instance's metrics
            return self._direct_client._metrics_store.get_stats(
                tags=tags,
                start_time=start_time
            )
        else:
            # API mode - call metrics endpoint with requests
            import requests
            params = {}
            if tags:
                params["tags"] = ",".join(tags)
            if start_time:
                params["start"] = start_time.isoformat()

            try:
                response = requests.get(
                    f"{self.server_url}/v1/metrics/summary",
                    params=params,
                    timeout=10
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"⚠️  Metrics not available from server: {response.status_code}")
                    return {}
            except Exception as e:
                print(f"⚠️  Failed to get metrics from server: {e}")
                return {}

    def get_model_config(self, model: str) -> Optional[Dict]:
        """Get configuration for a specific model."""

        if self.mode == "direct":
            # Direct mode - access config directly
            try:
                return self._direct_client.config.get_model_config(model)
            except Exception:
                return None
        else:
            # API mode - limited info from models list
            models = self.list_models()
            for m in models:
                if m["id"] == model:
                    return {"model": m}
            return None


def parse_prompt_file(file_path: str) -> tuple[str, str]:
    """Parse a YAML prompt file into system and user messages."""
    import yaml

    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    system_content = data.get('system', '').strip()
    user_content = data.get('user', '').strip()

    if not system_content and not user_content:
        raise ValueError("No valid system/user sections found in prompt file")

    return system_content, user_content


def format_duration(seconds: float) -> str:
    """Format duration in seconds."""
    return f"{seconds:.2f}s"


async def run_single_test(
    client: TelelemClient,
    prompt: Union[str, dict],
    model: str,
    output_file: Optional[str] = None,
    temperature: float = 1.0,
    json_mode: bool = False,
    quiet: bool = False
) -> int:
    """Run a single test with the given parameters.

    Args:
        prompt: Either a file path (str) or inline prompt dict with 'system' and 'user' keys
    """
    # Determine prompt name for display
    if isinstance(prompt, dict):
        prompt_name = prompt.get('name', 'inline')
    else:
        prompt_name = prompt

    if not quiet:
        print(f"\n🚀 Running test")
        print(f"   Mode: {client.mode}")
        print(f"   Model: {model}")
        print(f"   Prompt: {prompt_name}")
        if temperature != 1.0:
            print(f"   Temperature: {temperature}")
        if json_mode:
            print(f"   JSON mode: enabled")
        print("-" * 50)

    # Parse prompt - either from file or inline dict
    try:
        if isinstance(prompt, dict):
            # Inline prompt
            system_content = prompt.get('system', '').strip()
            user_content = prompt.get('user', '').strip()
            if not system_content and not user_content:
                raise ValueError("Inline prompt must have 'system' or 'user' key")
        else:
            # File path
            system_content, user_content = parse_prompt_file(prompt)
        if not quiet:
            print(f"✅ Parsed prompt")
    except Exception as e:
        if not quiet:
            print(f"❌ Failed to parse prompt: {e}")
        return 1

    # Build messages
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if user_content:
        messages.append({"role": "user", "content": user_content})

    # Prepare request parameters
    params = {"temperature": temperature}
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    # Add tags for metrics tracking in direct mode
    if client.mode == "direct":
        params["tags"] = ["telelem_test", f"model:{model}"]

    # Make request
    if not quiet:
        print(f"🔄 Making request...")
    start_time = time.time()

    try:
        response = await client.chat_completion(
            messages=messages,
            model=model,
            **params
        )
        duration = time.time() - start_time

        # Extract response content (consistent format for both modes)
        choices = getattr(response, "choices", [])
        content = choices[0].message.content if choices else ""

        # Use Elelem's processed metrics for accurate token counts (includes reasoning tokens)
        elelem_metrics = getattr(response, "elelem_metrics", {})
        if elelem_metrics and "tokens" in elelem_metrics:
            tokens = elelem_metrics["tokens"]
            input_tokens = tokens.get("input", 0)
            output_tokens = tokens.get("output", 0)
            reasoning_tokens = tokens.get("reasoning", 0)
            total_tokens = tokens.get("total", 0)

            # Extract reasoning content if available
            reasoning_content = elelem_metrics.get("reasoning_content")
        else:
            # Fallback to raw usage if elelem_metrics not available
            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)
            else:
                input_tokens = 0
                output_tokens = 0
            reasoning_tokens = 0
            total_tokens = input_tokens + output_tokens
            reasoning_content = None

        if not quiet:
            print(f"✅ Request completed in {format_duration(duration)}")
            print(f"   Input tokens: {input_tokens:,}")
            print(f"   Output tokens: {output_tokens:,}")
            if reasoning_tokens > 0:
                print(f"   Reasoning tokens: {reasoning_tokens:,}")
            print(f"   Total tokens: {total_tokens:,}")

            if duration > 0 and output_tokens > 0:
                tokens_per_sec = output_tokens / duration
                print(f"   Generation speed: {tokens_per_sec:.1f} tokens/s")

            # Display cost information if available
            if elelem_metrics and "costs_usd" in elelem_metrics:
                costs = elelem_metrics["costs_usd"]
                total_cost = costs.get("total_cost_usd", 0.0)
                input_cost = costs.get("input_cost_usd", 0.0)
                output_cost = costs.get("output_cost_usd", 0.0)
                reasoning_cost = costs.get("reasoning_cost_usd", 0.0)

                print(f"   Total cost: ${total_cost:.6f}")
                if reasoning_cost > 0:
                    print(f"   Input cost: ${input_cost:.6f}, Output cost: ${output_cost:.6f}, Reasoning cost: ${reasoning_cost:.6f}")
                else:
                    print(f"   Input cost: ${input_cost:.6f}, Output cost: ${output_cost:.6f}")

            # Validate JSON if needed
            if json_mode:
                try:
                    json.loads(content)
                    print(f"   ✅ Valid JSON response")
                except json.JSONDecodeError as e:
                    print(f"   ❌ Invalid JSON: {e}")

            # Show reasoning content if available
            if reasoning_content:
                print(f"\n🧠 Reasoning content:")
                reasoning_preview = reasoning_content[:300] if reasoning_content else "(empty)"
                print(f"   {reasoning_preview}{'...' if len(reasoning_content) > 300 else ''}")

            # Show preview
            print(f"\n📄 Response preview:")
            preview = content[:200] if content else "(empty)"
            print(f"   {preview}{'...' if len(content) > 200 else ''}")

        # Save output if requested
        if output_file:
            output_data = {
                "request": {
                    "model": model,
                    "temperature": temperature,
                    "json_mode": json_mode,
                    "prompt": prompt_name,
                    "mode": client.mode
                },
                "response": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "timestamp": datetime.now().isoformat()
                },
                "metrics": {
                    "duration_seconds": duration,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens
                }
            }

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            if not quiet:
                print(f"✅ Response saved to: {output_file}")

        return 0

    except Exception as e:
        duration = time.time() - start_time
        if not quiet:
            print(f"❌ Request failed after {format_duration(duration)}: {e}")

        # Save error if output file specified
        if output_file:
            error_data = {
                "request": {
                    "model": model,
                    "temperature": temperature,
                    "json_mode": json_mode,
                    "prompt": prompt_name,
                    "mode": client.mode
                },
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            }

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(error_data, f, indent=2)

            if not quiet:
                print(f"❌ Error saved to: {output_file}")

        return 1


async def run_batch_tests(
    client: TelelemClient,
    batch_file: str,
    output_dir: str = "results"
) -> int:
    """Run batch tests with multiple models and prompts."""

    print(f"\n🚀 Running batch tests")
    print(f"   Mode: {client.mode}")
    print(f"   Config: {batch_file}")
    print(f"   Output: {output_dir}/")
    print("-" * 50)

    # Load batch configuration
    try:
        with open(batch_file, 'r') as f:
            config = json.load(f)

        models = config.get("models", [])
        prompts = config.get("prompts", [])

        if not models or not prompts:
            print("❌ Batch config must have 'models' and 'prompts' arrays")
            return 1

        total_tests = len(models) * len(prompts)
        print(f"✅ Loaded config: {len(models)} models × {len(prompts)} prompts = {total_tests} tests")

    except Exception as e:
        print(f"❌ Failed to load batch config: {e}")
        return 1

    # Track metrics
    test_start_time = datetime.utcnow()
    results = []

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run all combinations - parallel execution per prompt
    test_num = 0
    for prompt_idx, prompt in enumerate(prompts, 1):
        # Get prompt name for display
        if isinstance(prompt, dict):
            prompt_name = prompt.get('name', 'inline')
        else:
            prompt_name = Path(prompt).stem

        print(f"\n📊 Prompt {prompt_idx}/{len(prompts)}: {prompt_name}")
        print(f"   Running {len(models)} models in parallel...")

        # Create tasks for all models with this prompt
        tasks = []
        for model in models:
            test_num += 1

            # Generate output filename
            model_safe = model.replace(':', '_').replace('/', '_')
            prompt_safe = prompt_name
            output_file = output_path / f"{model_safe}_{prompt_safe}.json"

            # Create task for this model/prompt combination
            task = run_single_test(
                client=client,
                prompt=prompt,
                model=model,
                output_file=str(output_file),
                temperature=config.get("temperature", 1.0),
                json_mode=config.get("json", False),
                quiet=True  # Suppress verbose output in parallel mode
            )
            tasks.append((model, prompt_name, output_file, task))

        # Run all models for this prompt in parallel
        print(f"⚡ Executing {len(tasks)} requests in parallel...")
        start_time = time.time()

        task_results = await asyncio.gather(
            *[task for _, _, _, task in tasks],
            return_exceptions=True
        )

        elapsed = time.time() - start_time
        avg_time = elapsed / len(tasks)
        print(f"✅ Completed {len(tasks)} requests in {elapsed:.1f}s (avg {avg_time:.1f}s per request)")

        # If running sequentially, this would be close to elapsed time
        # If running in parallel, this would be much less
        if avg_time < elapsed * 0.6:  # Less than 60% means good parallelism
            print(f"   ⚡ Good parallelism detected!")

        # Process results
        for (model, prompt, output_file, _), result in zip(tasks, task_results):
            if isinstance(result, Exception):
                print(f"   ❌ {model}: Failed with error: {result}")
                success = False
            else:
                success = result == 0
                status = "✅" if success else "❌"
                print(f"   {status} {model}: {'Success' if success else 'Failed'}")

            results.append({
                "model": model,
                "prompt": prompt,
                "success": success,
                "output_file": str(output_file)
            })

    # Generate summary using metrics API - exactly like original
    print(f"\n📊 Batch Test Complete")
    print("-" * 50)

    # Create summary structure matching original
    summary = {
        "batch_config": batch_file,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elelem_mode": client.mode,
        "elelem_server": client.server_url if client.mode == "api" else None,
        "test_start_time": test_start_time.isoformat(),
        "models": {}
    }

    # Get overall summary (no tags = all metrics since test_start_time)
    overall_summary = client.get_metrics(start_time=test_start_time)

    # Get per-model summaries exactly like original
    tested_models = set(result["model"] for result in results)
    for model in sorted(tested_models):
        model_summary = client.get_metrics(
            tags=[f"model:{model}"],
            start_time=test_start_time
        )
        summary["models"][model] = model_summary

    # Store overall summary and print results
    summary["overall"] = overall_summary

    if overall_summary:
        requests = overall_summary.get("requests", {})
        total_calls = requests.get("total", 0)
        total_successes = requests.get("successful", 0)
        total_cost = overall_summary.get("costs", {}).get("total", 0)

        print(f"✅ Overall: {total_successes}/{total_calls} successful")
        if total_cost > 0:
            print(f"💵 Total cost: ${total_cost:.6f}")

    # Print per-model breakdown like original
    for model in sorted(tested_models):
        model_stats = summary["models"].get(model, {})
        if model_stats:
            requests = model_stats.get("requests", {})
            total_calls = requests.get("total", 0)
            total_successes = requests.get("successful", 0)
            total_cost = model_stats.get("costs", {}).get("total", 0)
            print(f"\n📊 {model}: {total_successes}/{total_calls} "
                  f"(${total_cost:.6f})")

    summary_file = output_path / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to: {summary_file}")

    # Generate CSV summary if we have metrics (works for both modes now)
    if overall_summary:
        csv_file = output_path / "batch_summary.csv"
        generate_csv_summary(client, list(tested_models), test_start_time, csv_file)
        print(f"📊 CSV summary saved to: {csv_file}")

    return 0


def generate_csv_summary(
    client: TelelemClient,
    models: List[str],
    start_time: datetime,
    csv_file: Path
):
    """Generate CSV summary with model performance metrics - matches original format."""
    import csv
    import requests

    # Get model configuration data for static info (costs, metadata)
    if client.mode == "api":
        # API mode - get model list
        try:
            response = requests.get(f"{client.server_url}/v1/models")
            models_data = response.json() if response.status_code == 200 else {"data": []}
            models_config = {m["id"]: m for m in models_data.get("data", [])}
        except:
            models_config = {}
    else:
        # Direct mode - access config exactly like original
        models_config = {}
        for model_key, model_config in client._direct_client._models.items():
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
                "model_configuration": model_config.get("display_metadata", {}).get("model_configuration", "none"),
                "cost": cost_data
            }

    # Prepare CSV data exactly like original
    csv_rows = []
    for model_key in models:
        # Get model-specific metrics
        model_stats = client.get_metrics(
            tags=[f"model:{model_key}"],
            start_time=start_time
        )

        if not model_stats:
            continue

        # Get static model info
        model_info = models_config.get(model_key, {})

        # Extract performance metrics from get_stats() format
        requests_stats = model_stats.get("requests", {})
        total_calls = requests_stats.get("total", 0)
        total_successes = requests_stats.get("successful", 0)
        success_rate = (total_successes / total_calls * 100) if total_calls > 0 else 0

        tokens_stats = model_stats.get("tokens", {})
        output_tokens = tokens_stats.get("output", {}).get("total", 0)
        reasoning_tokens = tokens_stats.get("reasoning", {}).get("total", 0)
        total_duration = model_stats.get("duration", {}).get("total", 0)

        # Calculate token rates (tokens per second)
        output_token_rate = output_tokens / total_duration if total_duration > 0 else 0
        real_output_tokens = output_tokens - reasoning_tokens
        real_output_token_rate = real_output_tokens / total_duration if total_duration > 0 else 0

        # Get costs from static config and calculate real cost per output token
        cost_config = model_info.get("cost", {})
        input_cost_per_1m = cost_config.get("input_cost_per_1m", 0) if isinstance(cost_config, dict) else 0
        output_cost_per_1m = cost_config.get("output_cost_per_1m", 0) if isinstance(cost_config, dict) else 0

        # Calculate real cost per output token (total cost / real output tokens)
        total_cost_usd = model_stats.get("costs", {}).get("total", 0)
        real_cost_per_output_token = (total_cost_usd / real_output_tokens) if real_output_tokens > 0 else 0

        # Calculate reasoning token ratio (reasoning_tokens / output_tokens)
        reasoning_token_ratio = (reasoning_tokens / output_tokens) if output_tokens > 0 else 0

        csv_rows.append({
            "model_name": model_info.get("nickname", model_key),
            "model_owner": model_info.get("owned_by", "unknown"),
            "model_provider": model_info.get("provider", "unknown"),
            "license": model_info.get("license", "unknown"),
            "model_configuration": model_info.get("model_configuration", "none"),
            "model_page": model_info.get("model_page", ""),
            "cost_per_1m_input": input_cost_per_1m,
            "cost_per_1m_output": output_cost_per_1m,
            "success_rate_percent": round(success_rate, 1),
            "output_token_rate": round(output_token_rate, 1),
            "real_output_token_rate": round(real_output_token_rate, 1),
            "computed_cost_per_1m_actual_output_token": round(real_cost_per_output_token * 1000000, 6),  # Convert to cost per 1M tokens
            "reasoning_token_ratio": round(reasoning_token_ratio, 3)  # Ratio as percentage (0.0 to 1.0)
        })

    # Write CSV file with exact same fieldnames as original
    if csv_rows:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "model_name", "model_owner", "model_provider", "license", "model_configuration", "model_page",
                "cost_per_1m_input", "cost_per_1m_output", "success_rate_percent",
                "output_token_rate", "real_output_token_rate", "computed_cost_per_1m_actual_output_token",
                "reasoning_token_ratio"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simplified Telelem - Test runner for Elelem",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Connection mode
    parser.add_argument(
        "--server",
        help="Use API server instead of direct library (e.g., localhost:8000)"
    )

    # Test mode
    parser.add_argument(
        "--batch",
        help="Batch config file for multiple tests"
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt file to test"
    )
    parser.add_argument(
        "--model",
        help="Model to use for single test"
    )

    # Options
    parser.add_argument(
        "--output",
        help="Output file/directory for results"
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
        help="Enable JSON mode"
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

    # Initialize client once
    try:
        client = TelelemClient(server_url=args.server)
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return 1

    # Run appropriate mode
    if args.batch:
        # Batch mode
        if not os.path.exists(args.batch):
            print(f"❌ Batch config not found: {args.batch}")
            return 1

        output_dir = args.output or "results"
        return asyncio.run(run_batch_tests(client, args.batch, output_dir))

    elif args.prompt and args.model:
        # Single test mode
        if not os.path.exists(args.prompt):
            print(f"❌ Prompt file not found: {args.prompt}")
            return 1

        output_file = args.output
        if not output_file:
            # Generate default output filename
            model_safe = args.model.replace(':', '_').replace('/', '_')
            prompt_safe = Path(args.prompt).stem
            output_file = f"{model_safe}_{prompt_safe}.json"

        return asyncio.run(run_single_test(
            client=client,
            prompt=args.prompt,
            model=args.model,
            output_file=output_file,
            temperature=args.temperature,
            json_mode=args.json
        ))

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Single test (direct mode)")
        print("  python telelem_simple.py --prompt test.prompt --model groq:openai/gpt-oss-120b")
        print("")
        print("  # Single test (API mode)")
        print("  python telelem_simple.py --server localhost:8000 --prompt test.prompt --model groq:openai/gpt-oss-120b")
        print("")
        print("  # Batch test")
        print("  python telelem_simple.py --batch batch.json --output results/")
        return 1


if __name__ == "__main__":
    sys.exit(main())