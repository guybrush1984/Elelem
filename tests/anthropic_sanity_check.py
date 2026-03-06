# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "anthropic>=0.40.0",
#     "python-dotenv>=1.0.1",
# ]
# ///
"""
Sanity check script for Anthropic API behavior.

Validates assumptions needed for the Elelem adapter:
1. Non-streaming response structure (content blocks, usage, stop_reason)
2. Streaming event types and structure
3. Extended thinking block format
4. Error exception types

Run: uv run tests/anthropic_sanity_check.py
"""

import asyncio
import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

import anthropic


def pp(label, obj):
    """Pretty-print an object with a label."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if hasattr(obj, "model_dump"):
        print(json.dumps(obj.model_dump(), indent=2, default=str))
    else:
        print(obj)


async def test_non_streaming():
    """Test 1: Non-streaming call — inspect Message structure."""
    print("\n\n>>> TEST 1: Non-streaming call")
    client = anthropic.AsyncAnthropic()

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
    )

    pp("Full response object", response)

    # Key fields we need for the adapter
    print("\n--- Key fields ---")
    print(f"response.id          = {response.id}")
    print(f"response.model       = {response.model}")
    print(f"response.stop_reason = {response.stop_reason}")
    print(f"response.type        = {response.type}")

    print(f"\nresponse.usage.input_tokens  = {response.usage.input_tokens}")
    print(f"response.usage.output_tokens = {response.usage.output_tokens}")

    print(f"\nContent blocks ({len(response.content)}):")
    for i, block in enumerate(response.content):
        print(f"  [{i}] type={block.type}, text={repr(block.text[:100]) if hasattr(block, 'text') else 'N/A'}")

    return True


async def test_non_streaming_with_system():
    """Test 1b: Non-streaming with system message."""
    print("\n\n>>> TEST 1b: Non-streaming with system message")
    client = anthropic.AsyncAnthropic()

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system="You are a pirate. Always respond in pirate speak.",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
    )

    print(f"Response: {response.content[0].text}")
    print(f"stop_reason: {response.stop_reason}")
    return True


async def test_streaming():
    """Test 2: Streaming call — inspect raw event types and structure."""
    print("\n\n>>> TEST 2: Streaming call")
    client = anthropic.AsyncAnthropic()

    stream = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        stream=True,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    )

    print("\n--- Raw streaming events ---")
    event_count = 0
    async for event in stream:
        event_count += 1
        event_type = event.type

        if event_type == "message_start":
            msg = event.message
            print(f"  [{event_count}] message_start: id={msg.id}, model={msg.model}, usage.input_tokens={msg.usage.input_tokens}")

        elif event_type == "content_block_start":
            print(f"  [{event_count}] content_block_start: index={event.index}, type={event.content_block.type}")

        elif event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                print(f"  [{event_count}] content_block_delta[text_delta]: text={repr(delta.text[:50])}")
            elif delta.type == "thinking_delta":
                print(f"  [{event_count}] content_block_delta[thinking_delta]: thinking={repr(delta.thinking[:50])}")
            elif delta.type == "signature_delta":
                print(f"  [{event_count}] content_block_delta[signature_delta]: sig={delta.signature[:30]}...")
            else:
                print(f"  [{event_count}] content_block_delta[{delta.type}]: {delta}")

        elif event_type == "content_block_stop":
            print(f"  [{event_count}] content_block_stop: index={event.index}")

        elif event_type == "message_delta":
            print(f"  [{event_count}] message_delta: stop_reason={event.delta.stop_reason}, usage.output_tokens={event.usage.output_tokens}")

        elif event_type == "message_stop":
            print(f"  [{event_count}] message_stop")

        elif event_type == "ping":
            print(f"  [{event_count}] ping")

        else:
            print(f"  [{event_count}] UNKNOWN: {event_type} — {event}")

    print(f"\nTotal events: {event_count}")
    return True


async def test_thinking():
    """Test 3: Extended thinking — verify thinking block format."""
    print("\n\n>>> TEST 3: Extended thinking (Haiku 4.5)")
    client = anthropic.AsyncAnthropic()

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 5000},
        messages=[{"role": "user", "content": "What is 17 * 23?"}],
    )

    print(f"\nContent blocks ({len(response.content)}):")
    for i, block in enumerate(response.content):
        if block.type == "thinking":
            print(f"  [{i}] type=thinking, thinking={repr(block.thinking[:200])}...")
            if hasattr(block, "signature"):
                print(f"       signature={block.signature[:30]}..." if block.signature else "       signature=None")
        elif block.type == "text":
            print(f"  [{i}] type=text, text={repr(block.text[:200])}")
        else:
            print(f"  [{i}] type={block.type}")

    print(f"\nUsage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
    # Check if there's a separate thinking token count
    usage_dict = response.usage.model_dump() if hasattr(response.usage, "model_dump") else vars(response.usage)
    print(f"Full usage fields: {list(usage_dict.keys())}")
    print(f"Full usage: {json.dumps(usage_dict, default=str)}")

    return True


async def test_thinking_streaming():
    """Test 3b: Extended thinking with streaming."""
    print("\n\n>>> TEST 3b: Extended thinking + streaming (Haiku 4.5)")
    client = anthropic.AsyncAnthropic()

    stream = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        stream=True,
        thinking={"type": "enabled", "budget_tokens": 5000},
        messages=[{"role": "user", "content": "What is 17 * 23?"}],
    )

    print("\n--- Streaming events with thinking ---")
    event_count = 0
    async for event in stream:
        event_count += 1
        event_type = event.type

        if event_type == "message_start":
            msg = event.message
            print(f"  [{event_count}] message_start: id={msg.id}, input_tokens={msg.usage.input_tokens}")
        elif event_type == "content_block_start":
            print(f"  [{event_count}] content_block_start: index={event.index}, type={event.content_block.type}")
        elif event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "thinking_delta":
                text = delta.thinking[:80] if len(delta.thinking) > 80 else delta.thinking
                print(f"  [{event_count}] thinking_delta: {repr(text)}")
            elif delta.type == "text_delta":
                print(f"  [{event_count}] text_delta: {repr(delta.text[:80])}")
            elif delta.type == "signature_delta":
                print(f"  [{event_count}] signature_delta: (skipped)")
            else:
                print(f"  [{event_count}] {delta.type}")
        elif event_type == "message_delta":
            print(f"  [{event_count}] message_delta: stop_reason={event.delta.stop_reason}, output_tokens={event.usage.output_tokens}")
        elif event_type in ("content_block_stop", "message_stop", "ping"):
            print(f"  [{event_count}] {event_type}")
        else:
            print(f"  [{event_count}] UNKNOWN: {event_type}")

    print(f"\nTotal events: {event_count}")
    return True


async def test_errors():
    """Test 4: Error cases — verify exception types."""
    print("\n\n>>> TEST 4: Error cases")

    # 4a: Bad API key
    print("\n--- 4a: Bad API key ---")
    try:
        bad_client = anthropic.AsyncAnthropic(api_key="sk-bad-key-12345")
        await bad_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
    except Exception as e:
        print(f"  Exception type: {type(e).__module__}.{type(e).__name__}")
        print(f"  Status code: {getattr(e, 'status_code', 'N/A')}")
        print(f"  Message: {str(e)[:150]}")

    # 4b: Bad model name
    print("\n--- 4b: Bad model name ---")
    try:
        client = anthropic.AsyncAnthropic()
        await client.messages.create(
            model="claude-nonexistent-model",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
    except Exception as e:
        print(f"  Exception type: {type(e).__module__}.{type(e).__name__}")
        print(f"  Status code: {getattr(e, 'status_code', 'N/A')}")
        print(f"  Message: {str(e)[:150]}")

    # 4c: Missing max_tokens
    print("\n--- 4c: Missing max_tokens ---")
    try:
        client = anthropic.AsyncAnthropic()
        await client.messages.create(
            model="claude-haiku-4-5-20251001",
            # max_tokens intentionally omitted
            messages=[{"role": "user", "content": "Hi"}],
        )
    except Exception as e:
        print(f"  Exception type: {type(e).__module__}.{type(e).__name__}")
        print(f"  Message: {str(e)[:150]}")

    return True


async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment or .env file")
        sys.exit(1)
    print(f"API key loaded: {api_key[:10]}...{api_key[-4:]}")

    tests = [
        ("Non-streaming", test_non_streaming),
        ("Non-streaming with system", test_non_streaming_with_system),
        ("Streaming", test_streaming),
        ("Thinking", test_thinking),
        ("Thinking + Streaming", test_thinking_streaming),
        ("Errors", test_errors),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            ok = await test_fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"\n  EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = f"ERROR: {type(e).__name__}"

    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {result:12s}  {name}")


if __name__ == "__main__":
    asyncio.run(main())
