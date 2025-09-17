#!/usr/bin/env python3
"""
Systematic test of Gemini reasoning tokens across different configurations.
Tests: 2 reasoning_effort levels × 2 thinking_config options × 3 models = 12 combinations
"""

import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../../.env')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from elelem import Elelem

async def test_combination(elelem, model, reasoning_effort, include_thoughts):
    """Test a specific combination and return the raw response."""

    extra_body = {}
    if include_thoughts:
        extra_body = {
            'google': {
                'thinking_config': {
                    'include_thoughts': True
                }
            }
        }

    kwargs = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': 'What is 17 * 23? Show your calculation step by step.'}
        ]
    }

    if reasoning_effort:
        kwargs['reasoning_effort'] = reasoning_effort

    if extra_body:
        kwargs['extra_body'] = extra_body

    try:
        response = await elelem.create_chat_completion(**kwargs)
        return {
            'success': True,
            'response': response,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'response': None,
            'error': str(e)
        }

async def main():
    """Run all 12 combinations systematically."""

    elelem = Elelem()

    # Test configurations
    models = [
        'google:gemini-2.5-pro',
        'google:gemini-2.5-flash',
        'google:gemini-2.5-flash-lite'
    ]

    reasoning_efforts = [None, 'low']  # None = no reasoning_effort param, 'low' = explicit low
    include_thoughts_options = [False, True]  # False = no thinking_config, True = include_thoughts: true

    results = {}
    test_num = 0

    print(f"Running {len(models)} × {len(reasoning_efforts)} × {len(include_thoughts_options)} = {len(models) * len(reasoning_efforts) * len(include_thoughts_options)} tests")
    print("=" * 80)

    for model in models:
        for reasoning_effort in reasoning_efforts:
            for include_thoughts in include_thoughts_options:
                test_num += 1

                # Generate descriptive key
                effort_str = reasoning_effort if reasoning_effort else 'none'
                thoughts_str = 'with_thoughts' if include_thoughts else 'no_thoughts'
                model_short = model.split(':')[1]  # gemini-2.5-pro

                test_key = f"{model_short}_{effort_str}_{thoughts_str}"

                print(f"[{test_num:2d}/12] Testing {test_key}")
                print(f"         Model: {model}")
                print(f"         Reasoning effort: {reasoning_effort}")
                print(f"         Include thoughts: {include_thoughts}")

                # Run test
                result = await test_combination(elelem, model, reasoning_effort, include_thoughts)

                if result['success']:
                    print(f"         ✅ SUCCESS")

                    # Extract key information for summary
                    response = result['response']
                    usage_info = {}

                    if isinstance(response, dict) and 'usage' in response:
                        usage = response['usage']
                        if isinstance(usage, dict):
                            usage_info = {
                                'prompt_tokens': usage.get('prompt_tokens', 0),
                                'completion_tokens': usage.get('completion_tokens', 0),
                                'total_tokens': usage.get('total_tokens', 0),
                                'reasoning_tokens': usage.get('reasoning_tokens', 0),
                                'completion_tokens_details': usage.get('completion_tokens_details'),
                                'all_keys': list(usage.keys())
                            }

                    # Elelem metadata
                    elelem_tokens = {}
                    if isinstance(response, dict) and 'elelem_metadata' in response:
                        metadata = response['elelem_metadata']
                        if 'tokens' in metadata:
                            elelem_tokens = metadata['tokens']

                    print(f"         Input tokens: {usage_info.get('prompt_tokens', 'N/A')}")
                    print(f"         Output tokens: {usage_info.get('completion_tokens', 'N/A')}")
                    print(f"         Reasoning tokens: {usage_info.get('reasoning_tokens', 'N/A')}")
                    print(f"         Elelem reasoning: {elelem_tokens.get('reasoning', 'N/A')}")

                else:
                    print(f"         ❌ FAILED: {result['error']}")

                # Store result
                results[test_key] = {
                    'test_config': {
                        'model': model,
                        'reasoning_effort': reasoning_effort,
                        'include_thoughts': include_thoughts
                    },
                    'result': result
                }

                # Save individual result to file
                output_file = f"gemini_test_{test_key}.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        'test_config': results[test_key]['test_config'],
                        'raw_response': result['response'],
                        'success': result['success'],
                        'error': result['error']
                    }, f, indent=2, default=str)

                print(f"         💾 Saved to: {output_file}")
                print()

    # Save summary
    summary = {
        'total_tests': test_num,
        'test_results': {}
    }

    for test_key, data in results.items():
        result = data['result']
        config = data['test_config']

        summary_item = {
            'model': config['model'],
            'reasoning_effort': config['reasoning_effort'],
            'include_thoughts': config['include_thoughts'],
            'success': result['success'],
            'error': result['error']
        }

        if result['success'] and result['response']:
            response = result['response']

            # Extract usage summary
            if isinstance(response, dict) and 'usage' in response:
                usage = response['usage']
                if isinstance(usage, dict):
                    summary_item['usage'] = {
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'completion_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0),
                        'reasoning_tokens': usage.get('reasoning_tokens', 0),
                        'has_completion_tokens_details': 'completion_tokens_details' in usage,
                        'all_usage_keys': list(usage.keys())
                    }

            # Extract elelem metadata summary
            if isinstance(response, dict) and 'elelem_metadata' in response:
                metadata = response['elelem_metadata']
                if 'tokens' in metadata:
                    summary_item['elelem_tokens'] = metadata['tokens']

        summary['test_results'][test_key] = summary_item

    with open('gemini_reasoning_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 80)
    print("✅ All tests completed!")
    print(f"📊 Summary saved to: gemini_reasoning_summary.json")
    print(f"📄 Individual results saved to: gemini_test_*.json")

if __name__ == "__main__":
    asyncio.run(main())