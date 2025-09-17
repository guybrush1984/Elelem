#!/usr/bin/env python3
"""
Test to understand token structures across all providers.
This will help us understand how each provider reports tokens.
"""

import asyncio
import json
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from elelem import Elelem

async def test_provider_tokens():
    """Test different providers to understand their token structures."""

    elelem = Elelem()

    # Simple prompt that should generate reasoning with reasoning models
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Calculate 25 * 17 step by step."}
    ]

    # Test models from different providers
    test_models = [
        # GROQ
        ("groq:openai/gpt-oss-20b?reasoning=low", "GROQ with reasoning"),

        # Gemini
        ("google:gemini-2.5-flash", "Gemini Flash without reasoning"),
        ("google:gemini-2.5-flash?reasoning=low", "Gemini Flash with reasoning"),

        # DeepInfra (for comparison)
        ("deepinfra:openai/gpt-oss-20b?reasoning=low", "DeepInfra with reasoning"),

        # Parasail (DeepSeek)
        ("parasail:deepseek-3.1", "DeepSeek without thinking"),
    ]

    results = []

    for model, description in test_models:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        try:
            response = await elelem.create_chat_completion(
                messages=messages,
                model=model,
                temperature=0.7
            )

            # Get raw response data
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__

            # Extract usage information
            usage = response_dict.get('usage', {})
            elelem_metrics = response_dict.get('elelem_metrics', {})

            result = {
                "model": model,
                "description": description,
                "raw_usage": {
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                    "total_tokens": usage.get('total_tokens', 0),
                    "completion_tokens_details": usage.get('completion_tokens_details'),
                    "all_fields": list(usage.keys()) if isinstance(usage, dict) else []
                },
                "elelem_metrics": elelem_metrics.get('tokens', {}),
                "analysis": {}
            }

            # Analyze the token relationships
            prompt = usage.get('prompt_tokens', 0)
            completion = usage.get('completion_tokens', 0)
            total = usage.get('total_tokens', 0)
            reasoning = elelem_metrics.get('tokens', {}).get('reasoning', 0)

            result['analysis'] = {
                "prompt_plus_completion": prompt + completion,
                "equals_total": (prompt + completion) == total,
                "difference_from_total": total - (prompt + completion),
                "reasoning_tokens": reasoning,
                "reasoning_in_completion": reasoning <= completion,
                "pure_output_if_reasoning_in_completion": completion - reasoning if reasoning <= completion else "N/A",
                "pure_output_if_reasoning_separate": completion,
                "response_content_length": len(response_dict.get('choices', [{}])[0].get('message', {}).get('content', ''))
            }

            results.append(result)

            # Print summary
            print(f"\nRaw Usage from Provider:")
            print(f"  prompt_tokens: {prompt}")
            print(f"  completion_tokens: {completion}")
            print(f"  total_tokens: {total}")
            print(f"  All fields: {result['raw_usage']['all_fields']}")

            print(f"\nElelem Metrics:")
            print(f"  input: {elelem_metrics.get('tokens', {}).get('input', 0)}")
            print(f"  output: {elelem_metrics.get('tokens', {}).get('output', 0)}")
            print(f"  reasoning: {reasoning}")
            print(f"  total: {elelem_metrics.get('tokens', {}).get('total', 0)}")

            print(f"\nAnalysis:")
            print(f"  prompt + completion = {prompt + completion}")
            print(f"  total_tokens = {total}")
            print(f"  Difference: {total - (prompt + completion)}")
            print(f"  Reasoning tokens: {reasoning}")
            print(f"  Response content length: {result['analysis']['response_content_length']} chars")

            if reasoning > 0:
                if reasoning <= completion:
                    print(f"  → Reasoning appears to be INCLUDED in completion_tokens")
                    print(f"  → Pure output = {completion - reasoning} tokens")
                else:
                    print(f"  → Reasoning appears to be SEPARATE from completion_tokens")
                    print(f"  → Pure output = {completion} tokens")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "model": model,
                "description": description,
                "error": str(e)
            })

    # Save results
    with open('token_structure_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Results saved to token_structure_analysis.json")

    return results

if __name__ == "__main__":
    asyncio.run(test_provider_tokens())