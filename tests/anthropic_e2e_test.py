# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "anthropic>=0.40.0",
#     "python-dotenv>=1.0.1",
# ]
# ///
"""
End-to-end test: Anthropic provider through Elelem.

Tests the full adapter path: Elelem → AnthropicAdapter → Anthropic API.
"""

import asyncio
import json
import os
import sys

# Add src to path so we can import elelem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

from elelem import Elelem


async def test_basic_call():
    """Test basic non-thinking call through Elelem."""
    print("\n>>> Test: Basic call (Sonnet 4.6)")
    e = Elelem()

    response = await e.create_chat_completion(
        model="anthropic:anthropic/claude-sonnet-4.6",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2? Answer in one word."},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    metrics = response.elelem_metrics
    print(f"  Content: {content}")
    print(f"  Model: {metrics['model_used']}")
    print(f"  Tokens: in={metrics['tokens']['input']}, out={metrics['tokens']['output']}")
    print(f"  Cost: ${metrics['costs_usd']['total_cost_usd']:.6f}")
    return True


async def test_json_schema():
    """Test JSON schema validation through Elelem."""
    print("\n>>> Test: JSON schema validation (Sonnet 4.6)")
    e = Elelem()

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "integer"},
            "explanation": {"type": "string"},
        },
        "required": ["answer", "explanation"],
    }

    response = await e.create_chat_completion(
        model="anthropic:anthropic/claude-sonnet-4.6",
        messages=[
            {"role": "user", "content": "What is 15 * 7? Return JSON with 'answer' and 'explanation'."},
        ],
        response_format={"type": "json_object"},
        json_schema=schema,
        temperature=0.0,
    )

    content = response.choices[0].message.content
    print(f"  Content: {content}")
    parsed = json.loads(content)
    print(f"  Parsed: {parsed}")
    assert parsed["answer"] == 105, f"Expected 105, got {parsed['answer']}"
    print("  Schema validation: PASS")
    return True


async def test_thinking():
    """Test extended thinking through Elelem."""
    print("\n>>> Test: Extended thinking (Sonnet 4.6?thinking)")
    e = Elelem()

    response = await e.create_chat_completion(
        model="anthropic:anthropic/claude-sonnet-4.6?thinking",
        messages=[
            {"role": "user", "content": "What is 17 * 23?"},
        ],
    )

    content = response.choices[0].message.content
    metrics = response.elelem_metrics
    reasoning = metrics.get("reasoning_content")
    print(f"  Content: {content}")
    print(f"  Reasoning: {reasoning[:200] if reasoning else 'None'}...")
    print(f"  Tokens: in={metrics['tokens']['input']}, out={metrics['tokens']['output']}")
    return True


async def test_streaming():
    """Test streaming (internal) through Elelem."""
    print("\n>>> Test: Streaming (Sonnet 4.6, stream=true in provider config)")
    e = Elelem()

    response = await e.create_chat_completion(
        model="anthropic:anthropic/claude-sonnet-4.6",
        messages=[
            {"role": "user", "content": "Count from 1 to 5, one per line."},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    metrics = response.elelem_metrics
    print(f"  Content: {content}")
    print(f"  Tokens: in={metrics['tokens']['input']}, out={metrics['tokens']['output']}")
    return True


async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found")
        sys.exit(1)

    tests = [
        ("Basic call", test_basic_call),
        ("JSON schema", test_json_schema),
        ("Thinking", test_thinking),
        ("Streaming", test_streaming),
    ]

    results = {}
    for name, fn in tests:
        try:
            ok = await fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            import traceback
            print(f"\n  EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = f"ERROR: {type(e).__name__}"

    print("\n\n" + "=" * 60)
    print("  E2E SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {result:12s}  {name}")


if __name__ == "__main__":
    asyncio.run(main())
