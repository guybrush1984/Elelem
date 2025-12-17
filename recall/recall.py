# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "elelem",
#     "python-dotenv",
# ]
# ///
"""
Extract prompts from debug dumps and replay requests.

Usage:
    # Extract prompt to edit
    uv run recall/recall.py debug_dumps/xxx.json --extract
    # → Creates xxx_prompt.json (edit this file)

    # Replay with original prompt
    uv run recall/recall.py debug_dumps/xxx.json --replay

    # Replay with modified prompt
    uv run recall/recall.py debug_dumps/xxx.json --replay --prompt xxx_prompt.json

    # Override model
    uv run recall/recall.py debug_dumps/xxx.json --replay --model deepinfra:deepseek/deepseek-3.2
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Auto-enable debug dumps for failed validations
os.environ.setdefault("ELELEM_DEBUG_VALIDATION", "1")
os.environ.setdefault("ELELEM_DEBUG_DIR", str(Path(__file__).parent.parent / "debug_dumps"))


def setup_logging(verbose: bool):
    """Configure logging based on verbosity."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    # Ensure elelem logger is set
    logging.getLogger("elelem").setLevel(level)


def extract_prompt(dump_path: str) -> str:
    """Extract messages from dump to a new file for editing."""
    with open(dump_path) as f:
        dump = json.load(f)

    # Create output path: xxx.json → xxx_prompt.json
    out_path = Path(dump_path).stem + "_prompt.json"
    out_path = Path(dump_path).parent / out_path

    prompt_data = {
        "messages": dump["request"]["messages"],
        "api_kwargs": dump["request"]["api_kwargs"],
        "model_info": dump["model_info"]
    }

    with open(out_path, "w") as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)

    print(f"Extracted to: {out_path}")
    return str(out_path)


def extract_schema(messages: list) -> dict | None:
    """Extract JSON schema from system message."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if "=== REQUIRED OUTPUT FORMAT ===" in content:
                try:
                    start = content.find("{", content.find("=== REQUIRED OUTPUT FORMAT ==="))
                    depth = 0
                    for i, c in enumerate(content[start:], start):
                        if c == "{":
                            depth += 1
                        elif c == "}":
                            depth -= 1
                            if depth == 0:
                                return json.loads(content[start:i+1])
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


async def single_request(elelem, messages, model, schema, api_kwargs, idx: int):
    """Execute a single request, return (idx, success, result_or_error, cost)."""
    try:
        response = await elelem.create_chat_completion(
            messages=messages,
            model=model,
            cache=False,
            response_format={"type": "json_object"},
            json_schema=schema,
            **api_kwargs
        )
        metrics = response.elelem_metrics
        cost = metrics.get('costs_usd', {}).get('total_cost_usd', 0)
        provider = metrics.get('actual_provider')
        return (idx, True, provider, cost)
    except Exception as e:
        # Failed responses are auto-dumped to debug_dumps/ via ELELEM_DEBUG_VALIDATION
        return (idx, False, f"{type(e).__name__}: {e}", 0)


async def replay(dump_path: str, prompt_path: str | None = None, model: str | None = None, repeat: int = 1, parallel: int = 0, allow_retry: bool = False):
    """Replay the request and validate against schema."""
    from elelem import Elelem

    # Load dump
    with open(dump_path) as f:
        dump = json.load(f)

    # Load prompt (from separate file or dump)
    if prompt_path:
        with open(prompt_path) as f:
            prompt = json.load(f)
        messages = prompt["messages"]
        api_kwargs = prompt.get("api_kwargs", dump["request"]["api_kwargs"])
    else:
        messages = dump["request"]["messages"]
        api_kwargs = dump["request"]["api_kwargs"]

    # Determine model
    model = model or dump["model_info"]["original_model"]

    # Extract schema for validation
    schema = extract_schema(messages)

    print(f"Model: {model}")
    print(f"Messages: {len(messages)}")
    print(f"Schema: {'found' if schema else 'not found'}")
    if repeat > 1:
        if parallel > 0:
            mode = f"parallel (concurrency={parallel})"
        else:
            mode = "sequential (stop on first failure)"
        print(f"Repeat: {repeat}x ({mode})")
    print()

    elelem = Elelem()

    # Disable JSON retries by default (fail immediately on validation error)
    if not allow_retry:
        elelem.config.retry_settings["max_json_retries"] = 0

    # Load benchmark data if configured
    from elelem._benchmark_store import get_benchmark_store
    benchmark_store = get_benchmark_store()
    if benchmark_store.enabled:
        await benchmark_store.fetch_once()
        if benchmark_store.get_all_benchmarks():
            print(f"Benchmark: loaded {len(benchmark_store.get_all_benchmarks())} entries")

    # Parallel execution with concurrency limit
    if parallel > 0 and repeat > 1:
        print(f"Launching {repeat} requests ({parallel} concurrent)...")
        semaphore = asyncio.Semaphore(parallel)
        failed = asyncio.Event()
        results = []
        results_lock = asyncio.Lock()
        all_tasks = []

        async def limited_request(idx):
            # Skip if already failed
            if failed.is_set():
                return (idx, None, "skipped", 0)

            async with semaphore:
                # Check again after acquiring semaphore
                if failed.is_set():
                    return (idx, None, "skipped", 0)

                result = await single_request(elelem, messages, model, schema, api_kwargs, idx)

                # On failure, signal to stop and cancel other tasks
                if not result[1]:
                    failed.set()
                    # Cancel all other pending tasks
                    for t in all_tasks:
                        if not t.done():
                            t.cancel()

                # Print result immediately
                async with results_lock:
                    idx, success, info, cost = result
                    if success:
                        print(f"[{idx+1}/{repeat}] OK ({info}, ${cost:.4f})")
                    else:
                        print(f"[{idx+1}/{repeat}] FAILED: {info}")

                return result

        all_tasks = [asyncio.create_task(limited_request(i)) for i in range(repeat)]

        # Wait for all tasks, catching cancellation
        results = []
        for i, task in enumerate(all_tasks):
            try:
                result = await task
                results.append(result)
            except asyncio.CancelledError:
                results.append((i, None, "cancelled", 0))

        # Filter out skipped/cancelled
        completed = [r for r in results if r[1] is not None]
        skipped = len([r for r in results if r[1] is None])
        successes = sum(1 for r in completed if r[1])
        failures = sum(1 for r in completed if not r[1])
        total_cost = sum(r[3] for r in completed)

        print(f"\n{successes} passed, {failures} failed, {skipped} skipped/cancelled (total cost: ${total_cost:.4f})")
        if failures > 0:
            sys.exit(1)
        return

    # Sequential execution
    successes = 0
    total_cost = 0.0

    for i in range(repeat):
        if repeat > 1:
            print(f"[{i+1}/{repeat}] ", end="", flush=True)

        try:
            response = await elelem.create_chat_completion(
                messages=messages,
                model=model,
                cache=False,  # Always disable caching for replay
                response_format={"type": "json_object"},
                json_schema=schema,
                **api_kwargs
            )

            content = response.choices[0].message.content
            metrics = response.elelem_metrics
            cost = metrics.get('costs_usd', {}).get('total_cost_usd', 0)
            total_cost += cost
            successes += 1

            if repeat > 1:
                print(f"OK ({metrics.get('actual_provider')}, ${cost:.4f})")
            else:
                print(f"SUCCESS")
                print(f"Provider: {metrics.get('actual_provider')}")
                print(f"Model: {metrics.get('actual_model')}")
                print(f"Tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
                print(f"Cost: ${cost:.4f}")

                # Save successful response
                out_path = Path(dump_path).stem + "_response.json"
                out_path = Path(dump_path).parent / out_path
                with open(out_path, "w") as f:
                    json.dump(json.loads(content), f, indent=2, ensure_ascii=False)
                print(f"Response saved: {out_path}")

        except Exception as e:
            if repeat > 1:
                print(f"FAILED: {type(e).__name__}: {e}")
                print(f"\nStopped after {successes}/{i+1} successes (total cost: ${total_cost:.4f})")
            else:
                print(f"\nFAILED: {type(e).__name__}: {e}")
            sys.exit(1)

    if repeat > 1:
        print(f"\nAll {successes}/{repeat} passed (total cost: ${total_cost:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Extract and replay debug dumps")
    parser.add_argument("dump", help="Path to debug dump JSON file")
    parser.add_argument("--extract", "-e", action="store_true", help="Extract prompt to editable file")
    parser.add_argument("--replay", "-r", action="store_true", help="Replay the request")
    parser.add_argument("--prompt", "-p", help="Use modified prompt file (with --replay)")
    parser.add_argument("--model", "-m", help="Override model")
    parser.add_argument("--repeat", "-n", type=int, default=1, help="Repeat N times (sequential, stop on first failure)")
    parser.add_argument("--parallel", "-P", type=int, default=0, metavar="N", help="Run in parallel with N concurrent requests")
    parser.add_argument("--allow-retry", action="store_true", help="Allow Elelem to retry on JSON errors (default: fail immediately)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show elelem debug traces")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.extract and not args.replay:
        parser.print_help()
        sys.exit(1)

    if args.extract:
        extract_prompt(args.dump)

    if args.replay:
        asyncio.run(replay(args.dump, args.prompt, args.model, args.repeat, args.parallel, args.allow_retry))


if __name__ == "__main__":
    main()
