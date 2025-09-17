#!/usr/bin/env python3
"""
Analyze existing test results to understand token structures across providers.
"""

import json
import glob
from pathlib import Path

def analyze_file(filename):
    """Analyze a single test result file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return None

    result = {'file': Path(filename).name}

    # Handle different file formats
    if 'complete_raw_response' in data:
        # Comprehensive test format
        response = data['complete_raw_response']
        usage = response.get('usage', {})
        elelem_metrics = response.get('elelem_metrics', {}).get('tokens', {})
        result['model'] = data.get('test_config', {}).get('model', 'unknown')
        result['format'] = 'comprehensive'

    elif 'metrics' in data and 'request' in data:
        # Telelem simple format
        result['model'] = data.get('request', {}).get('model', 'unknown')
        result['format'] = 'telelem'

        # In telelem format, metrics are at top level
        metrics = data.get('metrics', {})
        usage = {
            'prompt_tokens': metrics.get('input_tokens', 0),
            'completion_tokens': metrics.get('output_tokens', 0),
            'total_tokens': metrics.get('total_tokens', 0)
        }
        elelem_metrics = {
            'input': metrics.get('input_tokens', 0),
            'output': metrics.get('output_tokens', 0),
            'reasoning': metrics.get('reasoning_tokens', 0),
            'total': metrics.get('total_tokens', 0)
        }
    else:
        return None

    # Extract token data
    result['raw_usage'] = {
        'prompt_tokens': usage.get('prompt_tokens', 0),
        'completion_tokens': usage.get('completion_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0)
    }

    result['elelem_metrics'] = elelem_metrics

    # Calculate relationships
    prompt = result['raw_usage']['prompt_tokens']
    completion = result['raw_usage']['completion_tokens']
    total = result['raw_usage']['total_tokens']
    reasoning = elelem_metrics.get('reasoning', 0)

    result['analysis'] = {
        'prompt_plus_completion': prompt + completion,
        'total_matches_p_plus_c': (prompt + completion) == total,
        'hidden_tokens': total - (prompt + completion),
        'reasoning_tokens': reasoning,
        'reasoning_vs_hidden': reasoning - (total - prompt - completion) if total > (prompt + completion) else 'N/A',
        'reasoning_vs_completion': 'in_completion' if reasoning > 0 and reasoning < completion else 'separate' if reasoning > 0 else 'none'
    }

    return result

def main():
    """Analyze all relevant test files."""

    # Pattern groups to analyze
    patterns = [
        ('GEMINI', 'comprehensive_gemini*.json'),
        ('GEMINI', '*gemini*.json'),
        ('GROQ', '*groq*.json'),
        ('DEEPINFRA', '*deepinfra*.json'),
        ('PARASAIL', '*parasail*.json'),
        ('FIREWORKS', '*fireworks*.json'),
        ('SCALEWAY', '*scaleway*.json')
    ]

    for provider_name, pattern in patterns:
        files = glob.glob(pattern)
        if not files:
            continue

        print(f"\n{'='*80}")
        print(f"PROVIDER: {provider_name}")
        print(f"{'='*80}")

        for file in sorted(files)[:5]:  # Limit to 5 files per provider
            result = analyze_file(file)
            if not result:
                continue

            print(f"\nFile: {result['file']}")
            print(f"Model: {result['model']}")
            print(f"Format: {result['format']}")

            raw = result['raw_usage']
            print(f"\nRaw Provider Response:")
            print(f"  prompt_tokens:     {raw['prompt_tokens']:>6}")
            print(f"  completion_tokens: {raw['completion_tokens']:>6}")
            print(f"  total_tokens:      {raw['total_tokens']:>6}")

            elelem = result['elelem_metrics']
            if elelem:
                print(f"\nElelem Metrics:")
                print(f"  input:     {elelem.get('input', 0):>6}")
                print(f"  output:    {elelem.get('output', 0):>6}")
                print(f"  reasoning: {elelem.get('reasoning', 0):>6}")
                print(f"  total:     {elelem.get('total', 0):>6}")

            analysis = result['analysis']
            print(f"\nAnalysis:")
            print(f"  prompt + completion = {analysis['prompt_plus_completion']}")
            print(f"  Matches total? {analysis['total_matches_p_plus_c']}")
            print(f"  Hidden tokens (total - p - c): {analysis['hidden_tokens']}")
            print(f"  Reasoning tokens: {analysis['reasoning_tokens']}")

            # Key insight
            if analysis['hidden_tokens'] > 0:
                print(f"  ⚠️ Provider includes {analysis['hidden_tokens']} hidden tokens in total")
                if analysis['reasoning_tokens'] > 0:
                    if analysis['reasoning_tokens'] == analysis['hidden_tokens']:
                        print(f"  ✅ Hidden tokens match reasoning tokens exactly")
                    else:
                        print(f"  ❌ Hidden tokens ({analysis['hidden_tokens']}) != reasoning tokens ({analysis['reasoning_tokens']})")

            if analysis['reasoning_tokens'] > 0:
                if analysis['reasoning_tokens'] <= raw['completion_tokens']:
                    actual_output = raw['completion_tokens'] - analysis['reasoning_tokens']
                    print(f"  → Reasoning appears INCLUDED in completion_tokens")
                    print(f"  → Pure output tokens: {actual_output}")
                else:
                    print(f"  → Reasoning appears SEPARATE from completion_tokens")
                    print(f"  → Pure output tokens: {raw['completion_tokens']}")

if __name__ == "__main__":
    main()