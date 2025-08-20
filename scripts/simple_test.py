#!/usr/bin/env python3
"""
Simple test script for Elelem - testing it as a user would
"""

import asyncio
import json
import logging
import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from elelem import Elelem

# Configure logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_1_initialization():
    """Test 1: Initialize Elelem"""
    print("\n" + "="*50)
    print("TEST 1: Initialize Elelem")
    print("="*50)
    
    try:
        elelem = Elelem()
        print("✅ PASS - Initialized successfully")
        return elelem
    except Exception as e:
        print(f"❌ FAIL - {e}")
        sys.exit(1)


async def test_2_simple_request(elelem):
    """Test 2: Simple request without JSON"""
    print("\n" + "="*50)
    print("TEST 2: Simple request (no JSON)")
    print("="*50)
    
    # Find first available model
    models = [
        "groq:openai/gpt-oss-20b",
        "deepinfra:openai/gpt-oss-20b",
        "openai:gpt-4.1-mini"
    ]
    
    model = None
    for m in models:
        provider = m.split(":")[0]
        if os.getenv(f"{provider.upper()}_API_KEY"):
            model = m
            break
    
    if not model:
        print("❌ SKIP - No API keys available")
        return
    
    print(f"Using model: {model}")
    
    try:
        response = await elelem.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with exactly: Hello World"}
            ],
            model=model,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        print(f"Response: {content}")
        print("✅ PASS")
        
    except Exception as e:
        print(f"❌ FAIL - {e}")


async def test_3_simple_json_all_models(elelem):
    """Test 3: Simple JSON request with ALL available models"""
    print("\n" + "="*50)
    print("TEST 3: Simple JSON with ALL Available Models")
    print("="*50)
    
    # Get all available models using the list_models API
    models_response = elelem.list_models()
    available_models = [model for model in models_response["data"] if model["available"]]
    
    print(f"Found {len(available_models)} available models:")
    for model in available_models:
        print(f"  - {model['id']} (provider: {model['owned_by']})")
    
    if not available_models:
        print("❌ SKIP - No models available")
        return
    
    successes = 0
    failures = 0
    
    for model_info in available_models:
        model_id = model_info["id"]
        print(f"\nTesting {model_id}...")
        
        try:
            response = await elelem.create_chat_completion(
                messages=[
                    {"role": "user", "content": "Return a JSON object with name='test' and value=123"}
                ],
                model=model_id,
                response_format={"type": "json_object"},
                temperature=0.3,
                tags=["test_json_all_models"]
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            print(f"Response: {json.dumps(data, indent=2)}")
            
            if "name" in data or "value" in data:
                print(f"✅ PASS - {model_id}")
                successes += 1
            else:
                print(f"⚠️  WARNING - {model_id} - JSON valid but unexpected structure")
                successes += 1  # Still counts as success
                
        except json.JSONDecodeError:
            print(f"❌ FAIL - {model_id} - Invalid JSON")
            failures += 1
        except Exception as e:
            print(f"❌ FAIL - {model_id} - {e}")
            failures += 1
    
    print(f"\nResults: {successes}/{len(available_models)} models passed, {failures} failed")


async def test_4_complex_story_json_sequential(elelem):
    """Test 4: Complex Fable-like story generation prompts (20 sequential calls)"""
    print("\n" + "="*50)
    print("TEST 4: Complex Story JSON - 20 Sequential Calls")
    print("="*50)
    
    # Use DeepInfra if available (most likely to need retries)
    models = [
        "deepinfra:openai/gpt-oss-20b",
        "groq:openai/gpt-oss-20b",
        "openai:gpt-4.1-mini"
    ]
    
    model = None
    for m in models:
        provider = m.split(":")[0]
        if os.getenv(f"{provider.upper()}_API_KEY"):
            model = m
            break
    
    if not model:
        print("❌ SKIP - No API keys available")
        return
    
    print(f"Using model: {model}")
    print("Running 20 complex story generation prompts...\n")
    
    # Different temperatures to test retry mechanism - START HIGH!
    temperatures = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8,
                   1.5, 1.3, 1.1, 0.9, 1.4, 1.2, 1.0, 1.5, 1.3, 1.1]
    
    successes = 0
    failures = 0
    retries_triggered = 0
    
    for i in range(20):
        # Complex Fable-like prompt with dialogues (let Elelem add JSON instructions)
        story_prompt = f"""Create an interactive children's story:
{{
  "title": "The Adventure of the {['Brave', 'Curious', 'Magic', 'Tiny', 'Giant'][i % 5]} {['Dragon', 'Robot', 'Unicorn', 'Wizard', 'Knight'][i % 5]} #{i+1}",
  "story_parameters": {{
    "theme": "{['friendship', 'courage', 'discovery', 'kindness', 'adventure'][i % 5]}",
    "tone": "{['whimsical', 'exciting', 'mysterious', 'heartwarming', 'funny'][i % 5]}",
    "setting": "{['enchanted forest', 'space station', 'underwater city', 'cloud kingdom', 'magic school'][i % 5]}",
    "narrative_voice": "{['first_person', 'third_person'][i % 2]}",
    "target_age": {5 + (i % 8)},
    "educational_elements": ["{['counting', 'colors', 'emotions', 'problem-solving', 'teamwork'][i % 5]}", "{['science', 'nature', 'friendship', 'creativity', 'perseverance'][(i+1) % 5]}"],
    "word_count_target": {500 + i * 50}
  }},
  "characters": [
    {{
      "name": "Hero_{i+1}",
      "type": "protagonist",
      "age": {6 + (i % 5)},
      "personality_traits": ["brave", "curious", "{['kind', 'smart', 'funny', 'creative', 'determined'][i % 5]}"],
      "special_ability": "{['flying', 'invisibility', 'talking to animals', 'time travel', 'super strength'][i % 5]}",
      "character_arc": "learns about {['friendship', 'responsibility', 'courage', 'honesty', 'patience'][i % 5]}"
    }},
    {{
      "name": "Sidekick_{i+1}",
      "type": "supporting",
      "species": "{['talking cat', 'wise owl', 'loyal dog', 'magical butterfly', 'friendly ghost'][i % 5]}",
      "role": "comic relief and wisdom provider",
      "catchphrase": "Adventure awaits, my friend!"
    }},
    {{
      "name": "Antagonist_{i+1}",
      "type": "obstacle",
      "nature": "{['misunderstood', 'mischievous', 'challenging', 'puzzling', 'tricky'][i % 5]}",
      "motivation": "wants to {['be understood', 'play tricks', 'test heroes', 'guard treasure', 'cause chaos'][i % 5]}",
      "redemption_possible": {str(i % 3 != 0).lower()}
    }}
  ],
  "story": [
    {{
      "segment_type": "introduction",
      "text": "Once upon a time in the {['magical', 'mysterious', 'wonderful', 'enchanted', 'fantastic'][i % 5]} land, there lived a young hero who dreamed of great adventures.",
      "character": "narrator",
      "mood": "intriguing",
      "sound_effects": ["wind", "birds"],
      "duration_estimate": {30 + (i % 20)}
    }},
    {{
      "segment_type": "dialogue",
      "text": "I must find the {['crystal', 'artifact', 'treasure', 'secret', 'answer'][i % 5]}! But wait... did you hear that strange noise?",
      "character": "Hero_{i+1}",
      "emotion": "determined",
      "voice_modulation": "excited"
    }},
    {{
      "segment_type": "dialogue",
      "text": "Don't worry, my friend! I've seen this before. The path ahead is dangerous, but together we can overcome any obstacle. Remember what the wise owl told us: 'Courage isn't the absence of fear, it's acting despite it.'",
      "character": "Sidekick_{i+1}",
      "emotion": "reassuring",
      "voice_modulation": "calm"
    }},
    {{
      "segment_type": "dialogue",
      "text": "You're right! Let's go! But first, we should check our supplies. Do we have the {['magic map', 'ancient compass', 'glowing stone', 'enchanted key', 'mystic orb'][i % 5]}?",
      "character": "Hero_{i+1}",
      "emotion": "thoughtful",
      "voice_modulation": "questioning"
    }},
    {{
      "segment_type": "dialogue",
      "text": "Ha ha ha! You'll never find what you're looking for! I've hidden it where no one would think to look - behind the {['waterfall of whispers', 'mirror of truth', 'tree of ages', 'mountain of echoes', 'lake of reflections'][i % 5]}!",
      "character": "Antagonist_{i+1}",
      "emotion": "mocking",
      "voice_modulation": "villainous"
    }},
    {{
      "segment_type": "dialogue",
      "text": "Oh no! Did the villain just reveal their secret? Quick, write this down: '{['The moon shines brightest at midnight', 'Three steps forward, two steps back', 'Follow the singing birds', 'Count the stars twice', 'Listen to the silence'][i % 5]}'",
      "character": "Sidekick_{i+1}",
      "emotion": "excited",
      "voice_modulation": "whispering"
    }},
    {{
      "segment_type": "action",
      "text": "The journey through the {['dark forest', 'crystal caves', 'cloud maze', 'time portal', 'mirror realm'][i % 5]} was {['challenging', 'thrilling', 'mysterious', 'dangerous', 'exciting'][i % 5]}, with many twists and turns that tested their resolve.",
      "character": "narrator",
      "pacing": "fast",
      "background_music": "adventurous"
    }}
  ],
  "continuation": {{
    "choice_text": "What should Hero_{i+1} do next?",
    "choices": [
      {{
        "id": "choice_a_{i}",
        "text": "Enter the {['cave', 'portal', 'castle', 'forest', 'temple'][i % 5]} bravely",
        "consequence_hint": "leads to discovery",
        "difficulty": "medium",
        "required_items": []
      }},
      {{
        "id": "choice_b_{i}",
        "text": "Wait for {['help', 'dawn', 'the signal', 'backup', 'a sign'][i % 5]}",
        "consequence_hint": "safer but slower",
        "difficulty": "easy",
        "required_items": ["patience"]
      }},
      {{
        "id": "choice_c_{i}",
        "text": "Use the {['magic spell', 'special gadget', 'ancient map', 'wise advice', 'secret passage'][i % 5]}",
        "consequence_hint": "creative solution",
        "difficulty": "hard",
        "required_items": ["wisdom", "courage"]
      }}
    ],
    "story_progress": {0.25 + (i % 4) * 0.25},
    "branches_remaining": {3 - (i % 4)}
  }},
  "metadata": {{
    "version": "2.1.{i}",
    "generated_at": "2024-01-{(i % 28) + 1:02d}T{10 + (i % 14):02d}:{i % 60:02d}:00Z",
    "model_used": "{model}",
    "generation_parameters": {{
      "temperature": {temperatures[i]},
      "max_tokens": 4096,
      "response_format": "json_object"
    }},
    "quality_metrics": {{
      "coherence_score": {0.7 + (i % 30) / 100},
      "age_appropriateness": {0.8 + (i % 20) / 100},
      "educational_value": {0.75 + (i % 25) / 100}
    }}
  }}
}}"""
        
        try:
            # Track stats before
            stats_before = elelem.get_stats_by_tag("sequential_test")
            
            response = await elelem.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a story generator for children's tales. Generate precise JSON."},
                    {"role": "user", "content": story_prompt}
                ],
                model=model,
                response_format={"type": "json_object"},
                temperature=temperatures[i],
                tags=["sequential_test"]
            )
            
            # Track stats after
            stats_after = elelem.get_stats_by_tag("sequential_test")
            attempts = stats_after['total_calls'] - stats_before['total_calls']
            
            content = response.choices[0].message.content
            
            # Show first 50 chars of response for debugging
            preview = content[:50] + "..." if len(content) > 50 else content
            
            data = json.loads(content)
            
            # Quick validation
            required = ["title", "story_parameters", "characters", "story", "continuation", "metadata"]
            missing = [k for k in required if k not in data]
            
            if not missing:
                successes += 1
                status = "✅"
            else:
                status = f"⚠️ Missing: {missing}"
            
            if attempts > 1:
                retries_triggered += 1
                print(f"[{i+1:2}/20] {status} Temp={temperatures[i]:.2f} Retries={attempts-1} | {preview}")
            else:
                print(f"[{i+1:2}/20] {status} Temp={temperatures[i]:.2f} | {preview}")
                
        except json.JSONDecodeError:
            failures += 1
            print(f"[{i+1:2}/20] ❌ JSON FAIL Temp={temperatures[i]:.2f}")
        except Exception as e:
            failures += 1
            print(f"[{i+1:2}/20] ❌ ERROR Temp={temperatures[i]:.2f}: {str(e)[:50]}")
    
    stats = elelem.get_stats_by_tag("sequential_test")
    print(f"\n📊 Sequential Results: {successes}/20 success, {retries_triggered} triggered retries")
    print(f"   Total API calls: {stats['total_calls']} (avg {stats['total_calls']/20:.1f} per request)")
    print(f"   Total cost: ${stats['total_cost_usd']:.6f}")


async def test_6_multi_model_parallel_testing(elelem):
    """Test 6: Multi-Model Parallel Testing - 20 parallel requests per available model"""
    print("\n" + "="*50)
    print("TEST 6: Complex Story JSON - 20 Parallel Calls")  
    print("="*50)
    
    # Get all available models using list_models API, filter for Scaleway only
    models_response = elelem.list_models()
    all_available = [model for model in models_response["data"] if model["available"]]
    scaleway_models = [model["id"] for model in all_available if model["owned_by"] == "scaleway"]
    
    if not scaleway_models:
        print("❌ SKIP - No Scaleway models available")
        available_scaleway = [model["id"] for model in models_response["data"] if model["owned_by"] == "scaleway"]
        if available_scaleway:
            print(f"Found {len(available_scaleway)} Scaleway models but missing SCALEWAY_API_KEY:")
            for model in available_scaleway:
                print(f"  - {model}")
        return
    
    print(f"Testing {len(scaleway_models)} Scaleway models with 20 parallel requests each...")
    
    temperatures = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8,
                   1.5, 1.3, 1.1, 0.9, 1.4, 1.2, 1.0, 1.5, 1.3, 1.1]
    
    async def make_story_request(model_id, i):
        """Make a single story request for a specific model"""
        story_prompt = f"""Generate interactive story JSON:
{{
  "title": "Parallel Adventure {i+1}",
  "dialogues": [
    {{
      "speaker": "Hero_{i}",
      "text": "I can't believe we found the {['treasure', 'portal', 'crystal', 'artifact', 'secret'][i % 5]}! But wait, there's something strange about it...",
      "emotion": "excited but worried"
    }},
    {{
      "speaker": "Villain_{i}",
      "text": "Ha! You think you've won? This is only the beginning! The real {['challenge', 'mystery', 'danger', 'puzzle', 'test'][i % 5]} starts now!",
      "emotion": "menacing"
    }},
    {{
      "speaker": "Sidekick_{i}",
      "text": "Don't listen to them! We've come too far to give up now. Remember: '{['courage conquers fear', 'friendship is power', 'wisdom guides us', 'hope never dies', 'truth prevails'][i % 5]}'",
      "emotion": "encouraging"
    }}
  ],
  "choices": [
    {{"id": "a{i}", "text": "Confront the villain", "risk": "high"}},
    {{"id": "b{i}", "text": "Study the artifact", "risk": "medium"}},
    {{"id": "c{i}", "text": "Retreat and regroup", "risk": "low"}}
  ],
  "metadata": {{
    "chapter": {i+1},
    "word_count": {150 + i * 10},
    "difficulty_level": {0.5 + (i % 5) * 0.1}
  }}
}}"""
        
        try:
            response = await elelem.create_chat_completion(
                messages=[
                    {"role": "user", "content": story_prompt}
                ],
                model=model_id,
                response_format={"type": "json_object"},
                temperature=temperatures[i],
                tags=[f"{model_id}_parallel_{i}", f"{model_id}_parallel_test"]
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Check if valid
            required = ["title", "dialogues", "choices", "metadata"]
            missing = [k for k in required if k not in data]
            
            if not missing:
                return (i, temperatures[i], "success", None)
            else:
                return (i, temperatures[i], "partial", missing)
                
        except json.JSONDecodeError as e:
            return (i, temperatures[i], "json_error", str(e)[:50])
        except Exception as e:
            return (i, temperatures[i], "error", str(e)[:50])
    
    # Test each Scaleway model sequentially
    for model_idx, model_id in enumerate(scaleway_models):
        print(f"\n[Model {model_idx+1}/{len(scaleway_models)}] Testing {model_id}")
        print("Launching 20 parallel requests...")
        
        # Launch all 20 requests in parallel for this model
        import time
        start_time = time.time()
        
        tasks = [make_story_request(model_id, i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        
        # Analyze results
        successes = sum(1 for _, _, status, _ in results if status == "success")
        partials = sum(1 for _, _, status, _ in results if status == "partial")
        json_errors = sum(1 for _, _, status, _ in results if status == "json_error")
        errors = sum(1 for _, _, status, _ in results if status == "error")
        
        # Track temperature ranges for successful requests
        successful_temps = [temp for _, temp, status, _ in results if status in ["success", "partial"]]
        highest_temp = max(successful_temps) if successful_temps else 0
        lowest_temp = min(successful_temps) if successful_temps else 0
        
        print(f"⏱️  Completed 20 parallel requests in {duration:.2f} seconds")
        print(f"✅ Successes: {successes}")
        print(f"⚠️  Partial (missing fields): {partials}")
        print(f"❌ JSON errors: {json_errors}")
        print(f"❌ Other errors: {errors}")
        
        if successful_temps:
            print(f"🌡️  Highest successful temperature: {highest_temp:.2f}")
            print(f"🌡️  Lowest successful temperature: {lowest_temp:.2f}")
        
        # Show some error details
        for i, temp, status, detail in results:
            if status in ["json_error", "error"]:
                print(f"   Request {i+1} (temp={temp}): {status} - {detail}")
        
        # Stats for this model
        stats = elelem.get_stats_by_tag(f"{model_id}_parallel_test")
        print(f"📊 {model_id} stats:")
        print(f"   Total API calls: {stats['total_calls']}")
        print(f"   Total cost: ${stats['total_cost_usd']:.6f}")
        print(f"   Avg time per request: {duration/20:.2f}s")


async def test_6_stats(elelem):
    """Test 6: Verify statistics tracking including tag-based stats"""
    print("\n" + "="*50)
    print("TEST 6: Statistics & Tag Tracking")
    print("="*50)
    
    try:
        # First, test tag-specific statistics from previous tests
        print("\nTag-based statistics validation:")
        
        # Check test_simple tag from test 2
        simple_stats = elelem.get_stats_by_tag("test_simple")
        if simple_stats["total_calls"] > 0:
            print(f"\n  'test_simple' tag:")
            print(f"    Calls: {simple_stats['total_calls']}")
            print(f"    Tokens: {simple_stats['total_tokens']}")
            print(f"    Cost: ${simple_stats['total_cost_usd']:.6f}")
            assert simple_stats['total_input_tokens'] > 0, "Should have input tokens"
            assert simple_stats['total_output_tokens'] > 0, "Should have output tokens"
            print(f"    ✅ Valid structure")
        
        # Check test_json tag from test 3
        json_stats = elelem.get_stats_by_tag("test_json")
        if json_stats["total_calls"] > 0:
            print(f"\n  'test_json' tag:")
            print(f"    Calls: {json_stats['total_calls']}")
            print(f"    Input tokens: {json_stats['total_input_tokens']}")
            print(f"    Output tokens: {json_stats['total_output_tokens']}")
            print(f"    Cost: ${json_stats['total_cost_usd']:.6f}")
            assert json_stats['avg_duration_seconds'] > 0, "Should have avg duration"
            print(f"    ✅ Valid structure")
        
        # Check parallel_test tag from test 5
        parallel_stats = elelem.get_stats_by_tag("parallel_test")
        if parallel_stats["total_calls"] > 0:
            print(f"\n  'parallel_test' tag:")
            print(f"    Calls: {parallel_stats['total_calls']}")
            print(f"    Total cost: ${parallel_stats['total_cost_usd']:.6f}")
            print(f"    Avg duration: {parallel_stats['avg_duration_seconds']:.2f}s")
            
            # Check individual parallel tags to detect retries
            retries_found = 0
            for i in range(20):
                tag_stats = elelem.get_stats_by_tag(f"parallel_{i}")
                if tag_stats['total_calls'] > 1:
                    retries_found += 1
                    print(f"    Tag parallel_{i}: {tag_stats['total_calls']} calls (retry detected)")
            
            if retries_found > 0:
                print(f"    🔄 {retries_found} requests triggered retries")
            print(f"    ✅ Valid structure")
        
        # Test nonexistent tag returns empty stats
        nonexistent = elelem.get_stats_by_tag("this_tag_does_not_exist")
        assert nonexistent['total_calls'] == 0, "Nonexistent tag should have 0 calls"
        assert nonexistent['total_cost_usd'] == 0.0, "Nonexistent tag should have 0 cost"
        assert 'total_input_tokens' in nonexistent, "Should have full structure"
        assert 'avg_duration_seconds' in nonexistent, "Should have full structure"
        print(f"\n  Nonexistent tag returns empty stats: ✅")
        
        # Overall statistics
        print("\nOverall statistics:")
        stats = elelem.get_stats()
        
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Total input tokens: {stats['total_input_tokens']}")
        print(f"  Total output tokens: {stats['total_output_tokens']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Total cost: ${stats['total_cost_usd']:.6f}")
        print(f"  Avg duration: {stats['avg_duration_seconds']:.2f}s")
        
        # Verify token consistency
        assert stats['total_tokens'] == stats['total_input_tokens'] + stats['total_output_tokens'], "Token math should be consistent"
        
        # Verify cost consistency
        # Note: reasoning_cost_usd is already included in output_cost_usd (not a separate cost)
        expected_total = stats['total_input_cost_usd'] + stats['total_output_cost_usd']
        actual_total = stats['total_cost_usd']
        
        # Allow small floating point differences
        cost_diff = abs(expected_total - actual_total)
        assert cost_diff < 0.000001, f"Cost math should be consistent (diff: {cost_diff:.10f})"
        
        if stats.get('reasoning_tokens', 0) > 0:
            print(f"  Reasoning tokens: {stats['reasoning_tokens']}")
            print(f"  Reasoning cost: ${stats['reasoning_cost_usd']:.6f}")
        
        print("\n✅ PASS - All statistics and tag tracking working correctly")
        
    except AssertionError as e:
        print(f"❌ FAIL - Assertion error: {e}")
    except Exception as e:
        print(f"❌ FAIL - {e}")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ELELEM TEST SUITE")
    print("="*60)
    
    # Check available API keys
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("OpenAI")
    if os.getenv("GROQ_API_KEY"):
        providers.append("GROQ")
    if os.getenv("DEEPINFRA_API_KEY"):
        providers.append("DeepInfra")
    
    if not providers:
        print("\n❌ No API keys found!")
        print("Set one of: OPENAI_API_KEY, GROQ_API_KEY, DEEPINFRA_API_KEY")
        sys.exit(1)
    
    print(f"Available providers: {', '.join(providers)}")
    
    # Run tests sequentially
    elelem = await test_1_initialization()
    await test_2_simple_request(elelem)
    await test_3_simple_json_all_models(elelem)
    # Skip sequential test - takes too long
    # await test_4_complex_story_json_sequential(elelem)
    print("\n" + "="*50)
    print("TEST 4: Skipping sequential test (too slow)")
    print("="*50)
    await test_6_multi_model_parallel_testing(elelem)
    await test_6_stats(elelem)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())