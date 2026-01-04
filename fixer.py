# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx",
#     "pyyaml",
# ]
# ///
"""Standalone fixer CLI for testing fixer prompts on debug dumps.

Usage:
    uv run fixer.py <debug_dump.json> [--model MODEL] [--dry-run]

Examples:
    # Run fixer on a debug dump
    uv run fixer.py debug_dumps/elelem_debug_csv_yelp-1128_20251230_082154.json

    # Use a specific fixer model
    uv run fixer.py debug_dump.json --model cerebras:openai/gpt-oss-120b

    # Just show the fixer prompt without calling the LLM
    uv run fixer.py debug_dump.json --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
import yaml

# Default fixer model
DEFAULT_FIXER_MODEL = "cerebras:openai/gpt-oss-120b?reasoning=medium"
DEFAULT_ELELEM_URL = "http://localhost:8000"


def load_fixer_prompt(format_name: str) -> dict:
    """Load fixer prompt from YAML file."""
    prompt_dir = Path(__file__).parent / "src" / "elelem" / "fixer_prompts"
    prompt_file = prompt_dir / f"{format_name}.yaml"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Fixer prompt not found: {prompt_file}")

    with open(prompt_file) as f:
        return yaml.safe_load(f)


def build_messages(format_name: str, content: str, error: str, schema: dict) -> list:
    """Build fixer messages from template."""
    prompt = load_fixer_prompt(format_name)

    # Format schema based on format type
    if format_name == "json":
        schema_str = json.dumps(schema, indent=2)
    else:
        schema_str = yaml.dump(schema, default_flow_style=False)

    # Build messages with substituted variables
    system_content = prompt["system"]
    user_content = prompt["user"].format(
        error=error,
        schema=schema_str,
        content=content,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def call_fixer(messages: list, model: str, elelem_url: str) -> dict:
    """Call fixer LLM via Elelem API."""
    url = f"{elelem_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    print(f"Calling fixer model: {model}")
    print(f"Elelem URL: {url}")
    print()

    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def extract_result(response_content: str, format_name: str) -> dict:
    """Extract fixer result from response."""
    content = response_content.strip()

    # Clean markdown if present
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1])
        else:
            content = "\n".join(lines[1:])

    # Find JSON boundaries
    start = content.find("{")
    end = content.rfind("}") + 1

    if start < 0 or end <= start:
        return {"error": "Could not find JSON in response", "raw": response_content}

    json_str = content[start:end]

    try:
        wrapper = json.loads(json_str)
        return wrapper
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": response_content}


def main():
    parser = argparse.ArgumentParser(
        description="Test fixer prompts on debug dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("debug_dump", help="Path to debug dump JSON file")
    parser.add_argument(
        "--model",
        default=DEFAULT_FIXER_MODEL,
        help=f"Fixer model to use (default: {DEFAULT_FIXER_MODEL})",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_ELELEM_URL,
        help=f"Elelem server URL (default: {DEFAULT_ELELEM_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the fixer prompt, don't call the LLM",
    )
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="Show the raw LLM response",
    )

    args = parser.parse_args()

    # Load debug dump
    dump_path = Path(args.debug_dump)
    if not dump_path.exists():
        print(f"Error: Debug dump not found: {dump_path}", file=sys.stderr)
        sys.exit(1)

    with open(dump_path) as f:
        dump = json.load(f)

    # Extract info from dump
    format_name = dump.get("validation_type", "json")
    error = dump.get("error", "Unknown error")
    schema = dump.get("schema", {})
    content = dump.get("response", {}).get("content", "")
    request_id = dump.get("request_id", "unknown")
    model_info = dump.get("model_info", {})

    print("=" * 60)
    print(f"Debug Dump: {dump_path.name}")
    print(f"Request ID: {request_id}")
    print(f"Format: {format_name}")
    print(f"Original Model: {model_info.get('original_model', 'unknown')}")
    print(f"Provider: {model_info.get('provider', 'unknown')}")
    print("=" * 60)
    print()

    print("Error:")
    print(f"  {error}")
    print()

    print(f"Content ({len(content)} chars):")
    if len(content) > 500:
        print(f"  {content[:500]}...")
    else:
        print(f"  {content}")
    print()

    # Build fixer messages
    messages = build_messages(format_name, content, error, schema)

    if args.dry_run:
        print("=" * 60)
        print("FIXER PROMPT (dry-run mode)")
        print("=" * 60)
        print()
        print("SYSTEM MESSAGE:")
        print(messages[0]["content"])
        print()
        print("USER MESSAGE:")
        print(messages[1]["content"])
        return

    # Call fixer
    print("=" * 60)
    print("CALLING FIXER")
    print("=" * 60)
    print()

    try:
        response = call_fixer(messages, args.model, args.url)
    except httpx.HTTPStatusError as e:
        print(f"Error calling fixer: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Could not connect to Elelem at {args.url}", file=sys.stderr)
        print("Is the server running?", file=sys.stderr)
        sys.exit(1)

    # Extract response content
    response_content = response["choices"][0]["message"]["content"]

    if args.show_response:
        print("RAW RESPONSE:")
        print(response_content)
        print()

    # Parse result
    result = extract_result(response_content, format_name)

    print("=" * 60)
    print("FIXER RESULT")
    print("=" * 60)
    print()

    if "error" in result:
        print(f"Parse Error: {result['error']}")
        print()
        print("Raw response:")
        print(result.get("raw", response_content))
    else:
        fixable = result.get("fixable", True)
        changes = result.get("changes", "")
        fixed = result.get("fixed")

        print(f"Fixable: {fixable}")
        print(f"Changes: {changes}")
        print()

        if fixed:
            print("Fixed Content:")
            if isinstance(fixed, str):
                print(fixed)
            else:
                print(json.dumps(fixed, indent=2, ensure_ascii=False))
        else:
            print("Fixed Content: null (unfixable)")


if __name__ == "__main__":
    main()
