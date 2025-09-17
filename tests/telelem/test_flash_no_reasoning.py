#!/usr/bin/env python3
"""
Test if Gemini Flash models support reasoning_effort: "none"
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv('../../.env')
except:
    load_dotenv('.env')  # Try local .env if relative path fails

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from elelem import Elelem

async def test_flash_no_reasoning():
    """Test if Flash and Flash-lite support reasoning_effort: none"""

    elelem = Elelem()

    models_to_test = [
        'google:gemini-2.5-flash',
        'google:gemini-2.5-flash-lite'
    ]

    print("=== TESTING FLASH MODELS WITH reasoning_effort: 'none' ===")

    for model in models_to_test:
        print(f"\n🧪 Testing {model} with reasoning_effort='none'")
        print("-" * 60)

        try:
            response = await elelem.create_chat_completion(
                model=model,
                messages=[
                    {'role': 'user', 'content': 'What is 2+2? Answer briefly.'}
                ],
                reasoning_effort='none'
            )

            if response and 'elelem_metadata' in response:
                metadata = response['elelem_metadata']
                tokens = metadata.get('tokens', {})

                print(f"✅ SUCCESS - {model} supports reasoning_effort='none'")
                print(f"   Input tokens: {tokens.get('input', 0)}")
                print(f"   Output tokens: {tokens.get('output', 0)}")
                print(f"   Reasoning tokens: {tokens.get('reasoning', 0)}")
                print(f"   Total tokens: {tokens.get('total', 0)}")

                reasoning_tokens = tokens.get('reasoning', 0)
                if reasoning_tokens == 0:
                    print(f"   🎯 PERFECT: No reasoning tokens used (as expected)")
                else:
                    print(f"   ⚠️  UNEXPECTED: {reasoning_tokens} reasoning tokens used")
            else:
                print(f"✅ SUCCESS - {model} responded but no metadata available")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ FAILED - {model}: {error_msg}")

            if "Budget 0 is invalid" in error_msg:
                print(f"   🔒 This model REQUIRES reasoning (like Pro)")
            elif "thinking mode" in error_msg.lower():
                print(f"   🔒 This model only works in thinking mode")
            else:
                print(f"   🤔 Unexpected error type")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- If models support 'none': We can create ?quick variants")
    print("- If models fail with 'Budget 0': They require reasoning like Pro")

if __name__ == "__main__":
    asyncio.run(test_flash_no_reasoning())