#!/usr/bin/env python3
"""
Telelem - Test Elelem with structured prompt files

Usage:
    python telelem.py --prompt prompt.prompt --response prompt.response --llm virtual:gpt-oss-120b
    python telelem.py --prompt tests/telelem/prompt.prompt --response output.json --llm groq:openai/gpt-oss-120b --temperature 0.8
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the src directory to the path so we can import elelem
sys.path.insert(0, str(Path(__file__).parent / "src"))

from elelem import Elelem

# Setup logging
def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('telelem.log')
        ]
    )
    return logging.getLogger('telelem')


def parse_prompt_file(file_path: str) -> tuple[str, str]:
    """Parse a YAML prompt file into system and user messages.
    
    Expected format:
    system: |
      <system message content>
    user: |
      <user message content>
    """
    import yaml
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    system_content = data.get('system', '').strip()
    user_content = data.get('user', '').strip()
    
    if not system_content and not user_content:
        raise ValueError("No valid system/user sections found in YAML file")
    
    return system_content, user_content


def format_duration(seconds: float) -> str:
    """Format duration in seconds only."""
    return f"{seconds:.2f}s"


async def run_telelem_test(prompt_file: str, response_file: str, model: str, logger=None, **kwargs):
    """Run a telelem test with the specified parameters."""
    
    if logger is None:
        logger = logging.getLogger('telelem')
    
    logger.info(f"Starting telelem test - model: {model}, prompt: {prompt_file}")
    
    print(f"🚀 Telelem Test Starting")
    print(f"   Prompt: {prompt_file}")
    print(f"   Model: {model}")
    print(f"   Output: {response_file}")
    
    # Parse temperature and other parameters
    temperature = kwargs.get('temperature', 1.0)
    json_mode = kwargs.get('json', False)
    
    if temperature != 1.0:
        print(f"   Temperature: {temperature}")
    if json_mode:
        print(f"   JSON mode: enabled")
    
    print("-" * 50)
    
    # Parse the prompt file
    try:
        system_content, user_content = parse_prompt_file(prompt_file)
        print(f"✅ Parsed prompt file:")
        print(f"   System message: {len(system_content)} characters")
        print(f"   User message: {len(user_content)} characters")
    except Exception as e:
        print(f"❌ Failed to parse prompt file: {e}")
        return 1
    
    # Initialize Elelem
    try:
        elelem = Elelem()
        print(f"✅ Initialized Elelem")
        
        # Check if model is available
        config = elelem.config.get_model_config(model)
        candidates = config['candidates']
        print(f"✅ Model '{model}' found with {len(candidates)} candidate(s)")
        
        # Show candidates for virtual models
        if len(candidates) > 1:
            print("   Candidates:")
            for i, candidate in enumerate(candidates):
                provider = candidate['provider']
                timeout = elelem.config.get_candidate_timeout(candidate, config)
                print(f"   {i+1}. {provider} (timeout={timeout}s)")
        
    except Exception as e:
        print(f"❌ Failed to initialize Elelem or find model: {e}")
        return 1
    
    # Prepare messages
    messages = []
    if system_content:
        print("APPPENDING SYSTEM ROLE")
        messages.append({"role": "system", "content": system_content})
    if user_content:
        print("APPPENDING USER ROLE")
        messages.append({"role": "user", "content": user_content})
    
    # Prepare API parameters
    api_params = {
        "temperature": temperature
    }
    if json_mode:
        api_params["response_format"] = {"type": "json_object"}
    
    print(f"\n🔄 Making request...")
    start_time = time.time()
    
    # Make the request
    try:
        response = await elelem.create_chat_completion(
            messages=messages,
            model=model,
            tags=["telelem_test"],
            **api_params
        )
        
        duration = time.time() - start_time
        
        # Extract response details
        logger.debug(f"Response object type: {type(response)}")
        logger.debug(f"Response choices: {response.choices}")
        
        # Safely extract content
        content = None
        if response and hasattr(response, 'choices') and response.choices:
            first_choice = response.choices[0]
            logger.debug(f"First choice: {first_choice}")
            if hasattr(first_choice, 'message') and first_choice.message:
                content = first_choice.message.content
                logger.debug(f"Message content type: {type(content)}, value: {content[:100] if content else 'None'}")
        
        if content is None:
            logger.warning("Response content is None - this might cause issues")
            content = ""  # Default to empty string to avoid NoneType errors
        
        usage = response.usage
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = input_tokens + output_tokens
        
        print(f"✅ Request completed in {format_duration(duration)}")
        print(f"   Input tokens: {input_tokens:,}")
        print(f"   Output tokens: {output_tokens:,}")
        print(f"   Total tokens: {total_tokens:,}")
        
        # Calculate and display tokens per second
        if duration > 0 and output_tokens > 0:
            tokens_per_second = output_tokens / duration
            print(f"   Output speed: {tokens_per_second:.1f} tokens/s")
        
        # Get cost information
        stats = elelem.get_stats_by_tag("telelem_test")
        total_cost = stats.get('total_cost_usd', 0.0)
        input_cost = stats.get('total_input_cost_usd', 0.0)
        output_cost = stats.get('total_output_cost_usd', 0.0)
        
        # Always show cost info, even if 0
        print(f"   Cost: ${total_cost:.6f}")
        
        if total_cost > 0:
            # Calculate per-token costs if available
            if input_tokens > 0:
                input_cost_per_1k = (input_cost / input_tokens) * 1000
                print(f"   Input cost: ${input_cost_per_1k:.4f}/1k tokens")
            if output_tokens > 0:
                output_cost_per_1k = (output_cost / output_tokens) * 1000
                print(f"   Output cost: ${output_cost_per_1k:.4f}/1k tokens")
        else:
            print(f"   ⚠️  Cost calculation may not be available for this model/provider")
        
        # Check for provider info (virtual models)
        if hasattr(response, 'provider') and response.provider:
            print(f"   Provider used: {response.provider}")
        
        # Check for candidate iterations
        retry_analytics = stats.get('retry_analytics', {})
        if retry_analytics.get('candidate_iterations', 0) > 0:
            print(f"   🔄 Candidate iterations: {retry_analytics['candidate_iterations']}")
        if retry_analytics.get('temperature_reductions', 0) > 0:
            print(f"   🌡️  Temperature reductions: {retry_analytics['temperature_reductions']}")
        
        # Validate JSON if in JSON mode
        if json_mode and content:
            try:
                json.loads(content)
                print(f"   ✅ Valid JSON response")
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON response: {e}")
        
        print(f"\n📄 Response preview (first 200 chars):")
        preview = content[:200] if content else "(empty)"
        print(f"   {preview}{'...' if len(content or '') > 200 else ''}")
        
        # Parse content as JSON if possible for prettier output
        parsed_content = content
        if content and (json_mode or content.strip().startswith('{')):
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                # Keep as string if not valid JSON
                parsed_content = content
        
        # Save response to file
        try:
            response_path = Path(response_file)
            response_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive output
            output_data = {
                "request": {
                    "model": model,
                    "temperature": temperature,
                    "json_mode": json_mode,
                    "prompt_file": prompt_file
                },
                "response": {
                    "content": parsed_content,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                },
                "metrics": {
                    "duration_seconds": duration,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": total_cost,
                    "provider_used": getattr(response, 'provider', None)
                },
                "analytics": retry_analytics
            }
            
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Response saved to: {response_file}")
            
        except Exception as e:
            print(f"❌ Failed to save response: {e}")
            return 1
            
        return 0
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Log detailed error information
        logger.error(f"Request failed - Error type: {error_type}, Message: {error_msg}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        print(f"❌ Request failed after {format_duration(duration)}: {e}")
        
        # Still save error information
        try:
            response_path = Path(response_file)
            response_path.parent.mkdir(parents=True, exist_ok=True)
            
            error_data = {
                "request": {
                    "model": model,
                    "temperature": temperature,
                    "json_mode": json_mode,
                    "prompt_file": prompt_file
                },
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                },
                "metrics": {
                    "duration_seconds": duration,
                    "success": False
                }
            }
            
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
                
            print(f"🗂️  Error details saved to: {response_file}")
            
        except Exception as save_error:
            print(f"❌ Failed to save error details: {save_error}")
        
        return 1


async def run_batch_tests(batch_file: str, output_file: str, **kwargs):
    """Run batch tests with multiple models and prompts."""
    
    print(f"🚀 Telelem Batch Test Starting")
    print(f"   Batch config: {batch_file}")
    print(f"   Output: {output_file}")
    print("-" * 50)
    
    # Load batch configuration
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_config = json.load(f)
        
        models = batch_config.get('models', [])
        prompts = batch_config.get('prompts', [])
        
        print(f"✅ Loaded batch config:")
        print(f"   Models: {len(models)}")
        print(f"   Prompts: {len(prompts)}")
        print(f"   Total tests: {len(models) * len(prompts)}")
        
    except Exception as e:
        print(f"❌ Failed to load batch config: {e}")
        return 1
    
    # Initialize Elelem once
    try:
        elelem = Elelem()
        print(f"✅ Initialized Elelem")
    except Exception as e:
        print(f"❌ Failed to initialize Elelem: {e}")
        return 1
    
    # Run all combinations
    results = {
        "metadata": {
            "batch_file": batch_file,
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_models": len(models),
            "total_prompts": len(prompts),
            "total_tests": len(models) * len(prompts)
        },
        "runs": [],
        "summary_by_model": {},
        "summary_by_prompt": {}
    }
    
    test_num = 0
    total_tests = len(models) * len(prompts)
    
    for model in models:
        model_summary = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0,
            "avg_tokens_per_second": 0,
            "total_duration_seconds": 0
        }
        
        for prompt_file in prompts:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] Testing {model} with {prompt_file}")
            
            # Initialize prompt summary if needed
            if prompt_file not in results["summary_by_prompt"]:
                results["summary_by_prompt"][prompt_file] = {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost_usd": 0,
                    "avg_tokens_per_second": 0,
                    "total_duration_seconds": 0
                }
            
            prompt_summary = results["summary_by_prompt"][prompt_file]
            
            # Parse prompt file
            try:
                system_content, user_content = parse_prompt_file(prompt_file)
                messages = []
                if system_content:
                    messages.append({"role": "system", "content": system_content})
                if user_content:
                    messages.append({"role": "user", "content": user_content})
            except Exception as e:
                print(f"   ❌ Failed to parse prompt: {e}")
                run_result = {
                    "model": model,
                    "prompt": prompt_file,
                    "status": "failed",
                    "error": str(e),
                    "duration_seconds": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tokens_per_second": 0,
                    "cost_usd": 0
                }
                results["runs"].append(run_result)
                model_summary["failed_runs"] += 1
                prompt_summary["failed_runs"] += 1
                continue
            
            # Prepare API parameters
            api_params = {
                "temperature": kwargs.get('temperature', 1.0)
            }
            if kwargs.get('json', False):
                api_params["response_format"] = {"type": "json_object"}
            
            # Run the test
            start_time = time.time()
            unique_tag = f"batch_{test_num}_{model.replace(':', '_')}_{Path(prompt_file).stem}"
            
            try:
                response = await elelem.create_chat_completion(
                    messages=messages,
                    model=model,
                    tags=[unique_tag],
                    **api_params
                )
                
                duration = time.time() - start_time
                
                # Extract metrics
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                tokens_per_second = output_tokens / duration if duration > 0 else 0
                
                # Get cost from stats
                stats = elelem.get_stats_by_tag(unique_tag)
                cost_usd = stats.get('total_cost_usd', 0.0)
                
                # Get response preview
                content = response.choices[0].message.content
                preview = content[:100] if content else "(empty)"
                
                run_result = {
                    "model": model,
                    "prompt": prompt_file,
                    "status": "success",
                    "duration_seconds": round(duration, 2),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_second": round(tokens_per_second, 1),
                    "cost_usd": cost_usd,
                    "response_preview": preview
                }
                
                print(f"   ✅ Success: {output_tokens} tokens in {duration:.1f}s ({tokens_per_second:.1f} tok/s), cost: ${cost_usd:.6f}")
                
                # Update summaries
                model_summary["successful_runs"] += 1
                model_summary["total_input_tokens"] += input_tokens
                model_summary["total_output_tokens"] += output_tokens
                model_summary["total_cost_usd"] += cost_usd
                model_summary["total_duration_seconds"] += duration
                
                prompt_summary["successful_runs"] += 1
                prompt_summary["total_input_tokens"] += input_tokens
                prompt_summary["total_output_tokens"] += output_tokens
                prompt_summary["total_cost_usd"] += cost_usd
                prompt_summary["total_duration_seconds"] += duration
                
            except Exception as e:
                duration = time.time() - start_time
                run_result = {
                    "model": model,
                    "prompt": prompt_file,
                    "status": "failed",
                    "error": str(e)[:200],
                    "duration_seconds": round(duration, 2),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tokens_per_second": 0,
                    "cost_usd": 0
                }
                
                print(f"   ❌ Failed: {str(e)[:60]}...")
                
                model_summary["failed_runs"] += 1
                prompt_summary["failed_runs"] += 1
            
            model_summary["total_runs"] += 1
            prompt_summary["total_runs"] += 1
            results["runs"].append(run_result)
        
        # Calculate model averages
        if model_summary["successful_runs"] > 0:
            model_summary["avg_tokens_per_second"] = round(
                model_summary["total_output_tokens"] / model_summary["total_duration_seconds"]
                if model_summary["total_duration_seconds"] > 0 else 0, 1
            )
        
        results["summary_by_model"][model] = model_summary
    
    # Calculate prompt averages
    for prompt_file, summary in results["summary_by_prompt"].items():
        if summary["successful_runs"] > 0:
            summary["avg_tokens_per_second"] = round(
                summary["total_output_tokens"] / summary["total_duration_seconds"]
                if summary["total_duration_seconds"] > 0 else 0, 1
            )
    
    # Calculate overall summary
    results["overall_summary"] = {
        "total_runs": test_num,
        "successful_runs": sum(1 for r in results["runs"] if r["status"] == "success"),
        "failed_runs": sum(1 for r in results["runs"] if r["status"] == "failed"),
        "total_cost_usd": sum(r["cost_usd"] for r in results["runs"]),
        "total_input_tokens": sum(r["input_tokens"] for r in results["runs"]),
        "total_output_tokens": sum(r["output_tokens"] for r in results["runs"]),
        "total_duration_seconds": sum(r["duration_seconds"] for r in results["runs"])
    }
    
    # Save results
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Batch test complete!")
        print(f"   Total runs: {results['overall_summary']['total_runs']}")
        print(f"   Successful: {results['overall_summary']['successful_runs']}")
        print(f"   Failed: {results['overall_summary']['failed_runs']}")
        print(f"   Total cost: ${results['overall_summary']['total_cost_usd']:.6f}")
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return 1
    
    return 0


def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(
        description="Test Elelem with structured prompt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single test
  python telelem.py --prompt tests/telelem/prompt.prompt --response output.json --llm virtual:gpt-oss-120b
  
  # Batch testing
  python telelem.py --batch batch_config.json --output batch_results.json
  
  # Batch config example:
  {
    "models": ["groq:gpt-oss-120b", "fireworks:gpt-oss-20b"],
    "prompts": ["story.prompt", "small.prompt"]
  }
        """
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--batch",
        help="Path to JSON file with models and prompts lists for batch testing"
    )
    parser.add_argument(
        "--output",
        help="Path to save batch test results (used with --batch)"
    )
    
    # Single test mode arguments
    parser.add_argument(
        "--prompt",
        help="Path to .prompt file with Role: system/user sections"
    )
    parser.add_argument(
        "--response",
        help="Path to save the response JSON file"
    )
    parser.add_argument(
        "--llm",
        help="Model to use (e.g., virtual:gpt-oss-120b, groq:openai/gpt-oss-120b, openai:gpt-4.1)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Temperature for generation (default: 1.0)"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Enable JSON mode (response_format=json_object)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to telelem.log"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(debug=args.debug)
    if args.debug:
        print("🔍 Debug mode enabled - check telelem.log for details")
    
    # Determine which mode to run
    if args.batch and args.output:
        # Batch mode
        if not os.path.exists(args.batch):
            print(f"❌ Batch config file not found: {args.batch}")
            return 1
        
        try:
            exit_code = asyncio.run(run_batch_tests(
                batch_file=args.batch,
                output_file=args.output,
                temperature=args.temperature,
                json=args.json
            ))
            return exit_code
        except KeyboardInterrupt:
            print("\n❌ Batch test interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Unexpected error in batch mode: {e}")
            return 1
            
    elif args.prompt and args.response and args.llm:
        # Single test mode (original behavior)
        if not os.path.exists(args.prompt):
            print(f"❌ Prompt file not found: {args.prompt}")
            return 1
        
        try:
            exit_code = asyncio.run(run_telelem_test(
                prompt_file=args.prompt,
                response_file=args.response, 
                model=args.llm,
                logger=logger,
                temperature=args.temperature,
                json=args.json
            ))
            return exit_code
        except KeyboardInterrupt:
            print("\n❌ Test interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1
    else:
        # Invalid arguments
        if args.batch and not args.output:
            print("❌ Error: --batch requires --output")
        elif args.output and not args.batch:
            print("❌ Error: --output requires --batch")
        else:
            print("❌ Error: Either provide --batch + --output for batch mode, or --prompt + --response + --llm for single test")
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)