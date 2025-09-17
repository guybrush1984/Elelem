#!/usr/bin/env python3
"""
Comprehensive test of Gemini reasoning tokens and thinking capabilities.
Tests all reasoning_effort levels and thinking_config separately.
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

async def test_reasoning_effort(elelem, model, reasoning_effort):
    """Test reasoning_effort parameter alone."""

    kwargs = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': 'What is 17 * 23? Show your calculation step by step.'}
        ]
    }

    if reasoning_effort != "default":
        kwargs['reasoning_effort'] = reasoning_effort

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

async def test_thinking_config(elelem, model):
    """Test thinking_config with include_thoughts=true."""

    kwargs = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': 'What is 17 * 23? Show your calculation step by step.'}
        ],
        'extra_body': {
            'google': {
                'thinking_config': {
                    'include_thoughts': True
                }
            }
        }
    }

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

def analyze_response(response):
    """Extract key information from response for analysis."""
    if not response or not isinstance(response, dict):
        return {}

    analysis = {}

    # Usage information
    if 'usage' in response:
        usage = response['usage']
        if isinstance(usage, dict):
            analysis['usage'] = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'reasoning_tokens': usage.get('reasoning_tokens', 0),
                'completion_tokens_details': usage.get('completion_tokens_details'),
                'prompt_tokens_details': usage.get('prompt_tokens_details'),
                'all_usage_keys': list(usage.keys())
            }

            # Calculate potential hidden reasoning tokens
            prompt_plus_completion = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
            total = usage.get('total_tokens', 0)
            analysis['hidden_reasoning_tokens'] = max(0, total - prompt_plus_completion)

    # Content analysis
    if 'choices' in response and response['choices']:
        choice = response['choices'][0]
        if 'message' in choice and 'content' in choice['message']:
            content = choice['message']['content']
            analysis['content_length'] = len(content)
            analysis['has_thinking_tags'] = '<think>' in content or '</think>' in content
            analysis['content_preview'] = content[:200] + '...' if len(content) > 200 else content

    # Elelem metadata
    if 'elelem_metadata' in response or 'elelem_metrics' in response:
        metadata_key = 'elelem_metadata' if 'elelem_metadata' in response else 'elelem_metrics'
        metadata = response[metadata_key]
        if 'tokens' in metadata:
            analysis['elelem_tokens'] = metadata['tokens']

    return analysis

async def main():
    """Run comprehensive reasoning tests."""

    elelem = Elelem()

    # Test configurations
    models = [
        'google:gemini-2.5-pro',
        'google:gemini-2.5-flash',
        'google:gemini-2.5-flash-lite'
    ]

    # Test 1: reasoning_effort parameter
    reasoning_efforts = ["default", "none", "low", "medium", "high"]

    # Test 2: thinking_config
    thinking_tests = ["thinking_config"]

    all_results = {}
    test_num = 0
    total_tests = len(models) * (len(reasoning_efforts) + len(thinking_tests))

    print(f"Running comprehensive Gemini reasoning tests")
    print(f"Models: {len(models)}, Reasoning efforts: {len(reasoning_efforts)}, Thinking config: {len(thinking_tests)}")
    print(f"Total tests: {total_tests}")
    print("=" * 80)

    for model in models:
        model_short = model.split(':')[1]  # gemini-2.5-pro

        # Test reasoning_effort levels
        for reasoning_effort in reasoning_efforts:
            test_num += 1
            test_key = f"{model_short}_reasoning_{reasoning_effort}"

            print(f"[{test_num:2d}/{total_tests}] Testing {test_key}")
            print(f"         Model: {model}")
            print(f"         Reasoning effort: {reasoning_effort}")

            result = await test_reasoning_effort(elelem, model, reasoning_effort)

            if result['success']:
                analysis = analyze_response(result['response'])
                print(f"         ✅ SUCCESS")

                if 'usage' in analysis:
                    usage = analysis['usage']
                    print(f"         Tokens: {usage['prompt_tokens']} input + {usage['completion_tokens']} output = {usage['total_tokens']} total")
                    if analysis['hidden_reasoning_tokens'] > 0:
                        print(f"         Hidden reasoning: {analysis['hidden_reasoning_tokens']} tokens")
                    if usage['reasoning_tokens'] > 0:
                        print(f"         Explicit reasoning: {usage['reasoning_tokens']} tokens")

                if analysis.get('has_thinking_tags'):
                    print(f"         🧠 Contains thinking tags!")

            else:
                print(f"         ❌ FAILED: {result['error']}")

            # Store complete result
            all_results[test_key] = {
                'test_type': 'reasoning_effort',
                'test_config': {
                    'model': model,
                    'reasoning_effort': reasoning_effort if reasoning_effort != "default" else None
                },
                'result': result,
                'analysis': analyze_response(result['response']) if result['success'] else {}
            }

            # Save individual file with COMPLETE response
            output_file = f"comprehensive_{test_key}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'test_type': 'reasoning_effort',
                    'test_config': all_results[test_key]['test_config'],
                    'complete_raw_response': result['response'],  # COMPLETE response
                    'success': result['success'],
                    'error': result['error'],
                    'analysis': all_results[test_key]['analysis']
                }, f, indent=2, default=str)

            print(f"         💾 Saved complete response to: {output_file}")
            print()

        # Test thinking_config
        test_num += 1
        test_key = f"{model_short}_thinking_config"

        print(f"[{test_num:2d}/{total_tests}] Testing {test_key}")
        print(f"         Model: {model}")
        print(f"         Thinking config: include_thoughts=true")

        result = await test_thinking_config(elelem, model)

        if result['success']:
            analysis = analyze_response(result['response'])
            print(f"         ✅ SUCCESS")

            if 'usage' in analysis:
                usage = analysis['usage']
                print(f"         Tokens: {usage['prompt_tokens']} input + {usage['completion_tokens']} output = {usage['total_tokens']} total")
                if analysis['hidden_reasoning_tokens'] > 0:
                    print(f"         Hidden reasoning: {analysis['hidden_reasoning_tokens']} tokens")
                if usage['reasoning_tokens'] > 0:
                    print(f"         Explicit reasoning: {usage['reasoning_tokens']} tokens")

            if analysis.get('has_thinking_tags'):
                print(f"         🧠 Contains thinking tags!")

        else:
            print(f"         ❌ FAILED: {result['error']}")

        # Store complete result
        all_results[test_key] = {
            'test_type': 'thinking_config',
            'test_config': {
                'model': model,
                'thinking_config': {'include_thoughts': True}
            },
            'result': result,
            'analysis': analyze_response(result['response']) if result['success'] else {}
        }

        # Save individual file with COMPLETE response
        output_file = f"comprehensive_{test_key}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'test_type': 'thinking_config',
                'test_config': all_results[test_key]['test_config'],
                'complete_raw_response': result['response'],  # COMPLETE response
                'success': result['success'],
                'error': result['error'],
                'analysis': all_results[test_key]['analysis']
            }, f, indent=2, default=str)

        print(f"         💾 Saved complete response to: {output_file}")
        print()

    # Save comprehensive summary
    summary = {
        'total_tests': test_num,
        'models_tested': models,
        'reasoning_efforts_tested': reasoning_efforts,
        'thinking_config_tested': True,
        'results': {}
    }

    for test_key, data in all_results.items():
        summary['results'][test_key] = {
            'test_type': data['test_type'],
            'test_config': data['test_config'],
            'success': data['result']['success'],
            'error': data['result']['error'],
            'analysis': data['analysis']
        }

    with open('comprehensive_gemini_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 80)
    print("✅ Comprehensive testing completed!")
    print(f"📊 Summary: comprehensive_gemini_summary.json")
    print(f"📄 Individual complete responses: comprehensive_*.json")
    print()
    print("🔍 ANALYSIS SUMMARY:")

    # Quick summary of key findings
    for test_key, data in all_results.items():
        if data['result']['success'] and 'usage' in data['analysis']:
            usage = data['analysis']['usage']
            hidden = data['analysis']['hidden_reasoning_tokens']
            thinking_tags = data['analysis'].get('has_thinking_tags', False)

            status = []
            if hidden > 0:
                status.append(f"{hidden} hidden reasoning tokens")
            if usage['reasoning_tokens'] > 0:
                status.append(f"{usage['reasoning_tokens']} explicit reasoning tokens")
            if thinking_tags:
                status.append("thinking tags present")

            status_str = ", ".join(status) if status else "no reasoning detected"
            print(f"   {test_key:35s}: {status_str}")

if __name__ == "__main__":
    asyncio.run(main())