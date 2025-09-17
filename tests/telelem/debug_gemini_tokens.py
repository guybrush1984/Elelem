#!/usr/bin/env python3
"""
Debug Gemini token extraction issue.
"""

import asyncio
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from elelem import Elelem

async def debug_gemini():
    """Test Gemini to debug token extraction."""

    elelem = Elelem()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Calculate 17 * 23 step by step."}
    ]

    # Test Gemini with reasoning
    model = "google:gemini-2.5-flash?reasoning=low"

    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")

    try:
        response = await elelem.create_chat_completion(
            messages=messages,
            model=model,
            temperature=0.7
        )

        # Get raw response data
        response_dict = response if isinstance(response, dict) else (response.model_dump() if hasattr(response, 'model_dump') else response.__dict__)

        # Extract usage information
        usage = response_dict.get('usage', {})
        elelem_metrics = response_dict.get('elelem_metrics', {})

        print(f"\nRaw Usage from Provider:")
        print(f"  prompt_tokens: {usage.get('prompt_tokens', 0)}")
        print(f"  completion_tokens: {usage.get('completion_tokens', 0)}")
        print(f"  total_tokens: {usage.get('total_tokens', 0)}")
        print(f"  All usage fields: {list(usage.keys())}")

        print(f"\nCalculated reasoning:")
        prompt = usage.get('prompt_tokens', 0)
        completion = usage.get('completion_tokens', 0)
        total = usage.get('total_tokens', 0)
        calculated = total - prompt - completion
        print(f"  {total} - {prompt} - {completion} = {calculated}")

        print(f"\nElelem Metrics:")
        tokens = elelem_metrics.get('tokens', {})
        print(f"  input: {tokens.get('input', 0)}")
        print(f"  output: {tokens.get('output', 0)}")
        print(f"  reasoning: {tokens.get('reasoning', 0)}")
        print(f"  total: {tokens.get('total', 0)}")

        print(f"\nResponse content length: {len(response_dict.get('choices', [{}])[0].get('message', {}).get('content', ''))} chars")

        # Save full response for analysis
        with open('debug_gemini_response.json', 'w') as f:
            json.dump(response_dict, f, indent=2)
        print(f"\nFull response saved to debug_gemini_response.json")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_gemini())