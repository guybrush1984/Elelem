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
        "openrouter:openai/gpt-oss-20b",
        "openai:gpt-4.1-mini"
    ]
    
    model = None
    for m in models:
        provider = m.split(":")[0]
        api_key_name = f"{provider.upper()}_API_KEY"
        if os.getenv(api_key_name):
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
        "openrouter:openai/gpt-oss-20b",
        "openai:gpt-4.1-mini"
    ]
    
    model = None
    for m in models:
        provider = m.split(":")[0]
        api_key_name = f"{provider.upper()}_API_KEY"
        if os.getenv(api_key_name):
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
    
    # GROQ models prone to json_validate_failed errors at high temperature
    test_models = [
        "groq:openai/gpt-oss-20b"
    ]
    
    # Filter for available models
    available_models = []
    for model in test_models:
        if os.getenv("GROQ_API_KEY"):
            available_models.append(model)
    
    if not available_models:
        print("❌ SKIP - No GROQ models available (need GROQ_API_KEY)")
        return
    
    print(f"Testing {len(available_models)} GROQ models with 20 parallel requests each...")
    
    temperatures = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8,
                   1.5, 1.3, 1.1, 0.9, 1.4, 1.2, 1.0, 1.5, 1.3, 1.1]
    
    async def make_story_request(model_id, i):
        """Make a single story request for a specific model"""
        # Complex nested JSON structure prone to validation errors at high temperature
        story_prompt = f"""Generate this EXACT complex interactive story JSON with deeply nested dialogue and intricate character details:
{{
  "epic_story_title": "The Ultimate Quest #{i+1}: Chronicles of the Ancient Realm",
  "complex_character_system": {{
    "primary_hero": {{
      "name": "Hero_{i}_the_Brave_Explorer",
      "detailed_dialogue": "Listen carefully, brave companions! The ancient prophecy speaks of this moment: \\"When the crimson moon aligns with the Crystal of Infinite Power, a chosen hero shall declare these sacred words: 'I choose courage over comfort, wisdom over wealth, and friendship over fame!' But wait... do you hear those mysterious voices echoing from the forbidden caverns below? They sound like they're calling for help!\\"",
      "emotional_complexity": {{
        "primary_state": "determined yet frightened",
        "internal_thoughts": "Even though I'm terrified of failing everyone who believes in me, I understand that true courage isn't about not being scared - it's about doing what's right despite the fear",
        "voice_directions": {{
          "tone": "whispered determination building to confident proclamation",
          "delivery": "start softly, emphasize the prophecy quote, pause dramatically before the question",
          "background_ambiance": "wind whistling through ancient stone corridors with distant magical chimes"
        }}
      }}
    }},
    "primary_antagonist": {{
      "identity": "The Misunderstood Guardian of Forbidden Secrets", 
      "complex_speech": "Foolish mortals! You dare enter my sacred domain? For over 500 years I have protected these dangerous mysteries from those who would abuse them! Every so-called 'hero' who came before spoke the exact same words: \\"We're here to help!\\" But they ALL abandoned me when the trials became difficult. They ALL broke their sacred promises and left me alone! Tell me, young ones, why should THIS time be any different?!",
      "hidden_vulnerability": {{
        "true_emotion": "centuries of loneliness disguised as righteous anger",
        "secret_desire": "desperately wants someone to understand their noble sacrifice and stay as a true friend",
        "redemption_trigger": "when someone offers genuine companionship without expecting anything in return"
      }}
    }},
    "loyal_companion": {{
      "name": "Whiskers_the_Ancient_Wise_Cat",
      "inspirational_wisdom": "My dearest friend, I see the doubt clouding your brave heart, but remember the words your beloved grandmother always shared: \\"The most powerful magic in all the mystical realms isn't found in ancient spells or legendary artifacts - it lives in the unbreakable bonds between true friends!\\" You possess more of that rare magic than anyone I've encountered in my many lifetimes.",
      "special_abilities": ["can sense the deepest emotions of any living being", "sees through all illusions to reveal hidden truths", "remembers every act of kindness ever performed"]
    }}
  }},
  "intricate_plot_structure": {{
    "surface_quest": "Locate the Lost Crystal of Infinite Wisdom to restore harmony to the enchanted kingdom",
    "deeper_meaning": "Actually a journey about discovering that genuine power comes from helping others rather than controlling them",
    "hidden_lesson": "The most dangerous enemy is often simply someone who desperately needs a friend",
    "climactic_decision": {{
      "moral_dilemma": "Hero must choose between taking the easy path to victory or staying to help the lonely guardian find peace",
      "internal_struggle": "This is so much harder than I imagined. Part of me wants to just complete the quest and go home. But I know that real heroes don't abandon people who need them most.",
      "resolution_dialogue": "\\"I understand now - you're not evil, you're just heartbroken and alone. The real treasure isn't this magical crystal - it's the friendship we could share.\\" *extends hand in genuine friendship* \\"Will you give me the chance to prove that this time really can be different?\\""
    }}
  }},
  "interactive_choice_system": {{
    "critical_moment": "Standing before the ancient guardian with destiny hanging in the balance",
    "path_options": [
      {{
        "choice_name": "Courageous Compassion",
        "description": "Step forward with an open heart and offer understanding",
        "consequences": "Most challenging path but leads to the deepest transformation for everyone involved",
        "character_growth": "Hero becomes a champion of the misunderstood and lonely"
      }},
      {{
        "choice_name": "Strategic Wisdom", 
        "description": "Gather more information and seek a careful, balanced solution",
        "consequences": "Safer approach but potentially misses opportunities for profound change",
        "character_growth": "Hero develops into a thoughtful leader who considers all perspectives"
      }},
      {{
        "choice_name": "Creative Innovation",
        "description": "Invent a completely new approach that no one has ever tried before", 
        "consequences": "Unprecedented results that surprise everyone and reshape the entire realm",
        "character_growth": "Hero transforms into a visionary innovator who creates new possibilities"
      }}
    ]
  }},
  "rich_world_details": {{
    "mystical_location": "The Sanctuary of Whispering Crystal Formations",
    "atmospheric_description": "The very air shimmers with ancient magic and untold stories, while crystalline structures hum with otherworldly melodies that respond to the emotions of visitors",
    "sensory_experience": {{
      "visual_elements": "Everything glows with a soft rainbow aurora that shifts and changes with the characters' feelings",
      "audio_landscape": "Whispered conversations between ancient spirits blend with melodic chimes that echo heartbeats",
      "emotional_atmosphere": "Simultaneously ancient and timeless, mysterious yet familiar, scary but also deeply comforting"
    }},
    "magical_principles": {{
      "power_source": "Magic flows from emotional authenticity - the more genuine the feeling, the stronger the enchantment",
      "unique_properties": "Thoughts become visible as glowing whispers, memories can be shared through crystal touch",
      "ethical_limitations": "Selfish intentions cause the magical light to dim, while lies create painful discord in the harmony"
    }}
  }},
  "educational_themes": [
    "Empathy and understanding can transform the bitterest enemies into the truest friends",
    "Real courage means choosing to do what's right even when it's difficult or scary",
    "Every person has a meaningful story that explains their actions and deserves to be heard with compassion"
  ],
  "technical_metadata": {{
    "complexity_rating": 9.7,
    "estimated_duration": {45 + i * 3} minutes,
    "target_audience": ["ages 8-14", "families", "educators"],
    "narrative_themes": ["friendship", "courage", "understanding", "redemption", "personal growth"],
    "generation_parameters": {{
      "temperature_used": {temperatures[i]},
      "model_tested": "{model_id}",
      "request_number": {i+1},
      "parallel_batch": "test_fallback_system"
    }}
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
            required = ["epic_story_title", "complex_character_system", "intricate_plot_structure", "interactive_choice_system"]
            missing = [k for k in required if k not in data]
            
            if not missing:
                return (i, temperatures[i], "success", None)
            else:
                return (i, temperatures[i], "partial", missing)
                
        except json.JSONDecodeError as e:
            return (i, temperatures[i], "json_error", str(e)[:50])
        except Exception as e:
            return (i, temperatures[i], "error", str(e)[:50])
    
    # Test each GROQ model sequentially
    for model_idx, model_id in enumerate(available_models):
        print(f"\n[Model {model_idx+1}/{len(available_models)}] Testing {model_id}")
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


async def test_5_json_schema_validation(elelem):
    """Test 5: JSON Schema Validation with retry logic"""
    print("\n" + "="*50)
    print("TEST 5: JSON Schema Validation")
    print("="*50)
    
    # Test only with groq and deepinfra models
    test_models = [
        "groq:openai/gpt-oss-20b"
    ]
    
    available_models = []
    for model in test_models:
        provider = model.split(":")[0]
        if os.getenv(f"{provider.upper()}_API_KEY"):
            available_models.append(model)
    
    if not available_models:
        print("❌ SKIP - Need GROQ_API_KEY or DEEPINFRA_API_KEY")
        return
    
    print(f"Testing with models: {available_models}")
    
    # Define the interactive story schema from the user's example
    interactive_story_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["title", "story", "is_ending"],
        "properties": {
            "title": {"type": "string"},
            "story": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["character", "text", "voice_instructions"],
                    "properties": {
                        "character": {"type": "string"},
                        "text": {"type": "string"},
                        "voice_instructions": {
                            "type": "object",
                            "required": ["tone", "emotion", "delivery"],
                            "properties": {
                                "tone": {"type": "string"},
                                "emotion": {"type": "string"},
                                "delivery": {"type": "string"}
                            }
                        },
                        "ambiance_sound": {"type": "string"}
                    }
                }
            },
            "continuation": {
                "type": "object",
                "required": ["choice_text", "choices"],
                "properties": {
                    "choice_text": {"type": "string"},
                    "choices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "text", "consequence"],
                            "properties": {
                                "id": {"type": "string", "pattern": "^[A-D]$"},
                                "text": {"type": "string"},
                                "consequence": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "is_ending": {"type": "boolean"}
        }
    }
    
    for model in available_models:
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model}")
        print(f"{'='*60}")
        
        print("\n--- Warning Test: json_schema without JSON response format ---")
        try:
            response = await elelem.create_chat_completion(
                messages=[{"role": "user", "content": "Say hello"}],
                model=model,
                json_schema=interactive_story_schema,  # Schema without JSON format
                tags=["schema_warning_test"]
            )
            print("✅ PASS - Warning should appear in logs above")
            
        except Exception as e:
            print(f"❌ FAIL - Should not have failed: {e}")
        
        print(f"\n--- Schema Validation Loop Test (10 attempts with {model}) ---")
        successes = 0
        
        # Very high temperatures to increase chance of schema validation failures
        temperatures = [1.7, 1.8, 1.6, 1.9, 1.7, 1.8, 1.6, 1.9, 1.7, 1.8]
        
        for i in range(10):
            try:
                stats_before = elelem.get_stats_by_tag(f"schema_loop_{model}_{i}")
                
                # Complex chaotic prompt that still instructs to follow the schema structure
                response = await elelem.create_chat_completion(
                    messages=[
                        {"role": "user", "content": f"""Create an incredibly complex interactive story JSON (attempt #{i+1}) following this EXACT structure:

{{
  "title": "...",
  "story": [
    {{
      "character": "...",
      "text": "...",
      "voice_instructions": {{
        "tone": "...",
        "emotion": "...",
        "delivery": "..."
      }},
      "ambiance_sound": "..." (optional)
    }}
  ],
  "continuation": {{
    "choice_text": "...",
    "choices": [
      {{
        "id": "A",
        "text": "...",
        "consequence": "..."
      }}
    ]
  }},
  "is_ending": false
}}

Make this story absolutely CHAOTIC and mind-bendingly complex: featuring time-traveling wizards speaking in riddles, shape-shifting dragons with memory problems, sentient musical instruments creating reality through melodies, omniscient narrators who forget what they're narrating, characters existing in multiple dimensions simultaneously, plots flowing backward through time, magical systems based on emotional resonance of forgotten words, geography that changes based on collective mood, psychological profiles including childhood fears and philosophical opinions, dialogue with multiple layers of meaning and hidden subtext, conversations happening telepathically while pretending to discuss mundane topics, arguments resolved through interpretive dance, negotiations through meaningful silences, declarations of love as mathematical theorems, voice instructions specifying exact pitch fluctuations and micro-pauses, choice systems that unravel causality, options existing only when nobody looks at them, consequences manifesting in parallel timelines, moral frameworks questioning the nature of choice itself. Include at least 5 story segments with incredibly intricate character interactions, multiple complex choice options, and make every voice instruction absurdly detailed with breathing patterns, harmonic frequencies, linguistic quirks, and vocal textures. The more impossibly complex and chaotic while still following the JSON structure, the better!"""}
                    ],
                    model=model,
                    response_format={"type": "json_object"},
                    json_schema=interactive_story_schema,
                    temperature=temperatures[i],
                    tags=[f"schema_loop_{model}_{i}", f"schema_loop_{model}"]
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Verify schema compliance
                required_fields = ["title", "story", "is_ending", "continuation"]
                has_required = all(field in data for field in required_fields)
                
                if has_required:
                    successes += 1
                    status = "✅"
                else:
                    status = "⚠️"
                
                print(f"[{i+1:2}/10] {status} T={temperatures[i]:.1f} | {model}")
                    
            except json.JSONDecodeError as e:
                print(f"[{i+1:2}/10] ❌ JSON_FAIL T={temperatures[i]:.1f} | {model}")
            except Exception as e:
                print(f"[{i+1:2}/10] ❌ ERROR T={temperatures[i]:.1f} | {model}: {str(e)[:30]}...")
        
        # Summary for this model
        stats = elelem.get_stats_by_tag(f"schema_loop_{model}")
        retry_analytics = stats.get('retry_analytics', {})
        
        print(f"\n📊 {model} Results:")
        print(f"   Successes: {successes}/10")
        print(f"   Total API calls: {stats['total_calls']} (avg {stats['total_calls']/10:.1f} per request)")
        print(f"   Total cost: ${stats['total_cost_usd']:.6f}")
        
        # Display retry analytics
        print(f"\n🔄 Retry Analytics:")
        print(f"   JSON schema retries: {retry_analytics.get('json_schema_retries', 0)}")
        print(f"   JSON parse retries: {retry_analytics.get('json_parse_retries', 0)}")
        print(f"   API JSON validation retries: {retry_analytics.get('api_json_validation_retries', 0)}")
        print(f"   Temperature reductions: {retry_analytics.get('temperature_reductions', 0)}")
        print(f"   Response format removals: {retry_analytics.get('response_format_removals', 0)}")
        print(f"   Final failures: {retry_analytics.get('final_failures', 0)}")
        print(f"   Total retries: {retry_analytics.get('total_retries', 0)}")
        
        if retry_analytics.get('total_retries', 0) > 0:
            print(f"   🎯 Retry analytics working! Detected {retry_analytics.get('total_retries', 0)} total retry events")
        else:
            print(f"   ⚠️  No retries detected in analytics")
        
        print(f"\n\n\n--- Intentional Schema Mismatch Test (1 attempt with {model}) ---")
        try:
            stats_before = elelem.get_stats_by_tag(f"schema_mismatch_{model}")
            
            # Ask for something that doesn't match our story schema at all
            response = await elelem.create_chat_completion(
                messages=[
                    {"role": "user", "content": "Create a user profile JSON with fields: username, email, age, preferences array, and settings object with theme and notifications."}
                ],
                model=model,
                response_format={"type": "json_object"},
                json_schema=interactive_story_schema,  # Story schema but asking for different content
                temperature=1.4,
                tags=[f"schema_mismatch_{model}"]
            )
            
            stats_after = elelem.get_stats_by_tag(f"schema_mismatch_{model}")
            attempts = stats_after['total_calls'] - stats_before['total_calls']
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Check if it matches our story schema (it shouldn't initially)
            required_fields = ["title", "story", "is_ending", "continuation"]
            has_story_structure = all(field in data for field in required_fields)
            
            if attempts > 1:
                print(f"🎯 RETRIES={attempts-1} → {'Story schema (forced)' if has_story_structure else 'Still wrong structure'} | {model}")
            else:
                print(f"{'😮 FirstTry → Story schema (unexpected!)' if has_story_structure else '❌ FirstTry → Wrong structure (no retries)'} | {model}")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)[:60]}... | {model}")


async def test_openrouter_features(elelem):
    """Test OpenRouter-specific features: runtime cost, provider tracking, price-first routing"""
    print("\n" + "="*50)
    print("TEST OPENROUTER: Provider Tracking & Runtime Cost")
    print("="*50)
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ SKIP - OPENROUTER_API_KEY not available")
        return
    
    print("Testing OpenRouter integration features...")
    
    # Test 1: Basic OpenRouter request with runtime cost
    print("\n1. Runtime cost tracking:")
    try:
        response = await elelem.create_chat_completion(
            messages=[{"role": "user", "content": "Say hello in 3 words"}],
            model="openrouter:openai/gpt-oss-20b",
            max_tokens=10,
            tags=["openrouter_test"]
        )
        
        print(f"   Response: {response.choices[0].message.content}")
        
        # Check if provider info is captured
        if hasattr(response, 'provider'):
            print(f"   ✅ Provider captured: {response.provider}")
        else:
            print(f"   ❌ No provider info in response")
        
        # Get stats to verify runtime cost tracking
        stats = elelem.get_stats_by_tag("openrouter_test")
        if stats['total_cost_usd'] > 0:
            print(f"   ✅ Runtime cost: ${stats['total_cost_usd']:.8f}")
        else:
            print(f"   ❌ No cost tracked")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Provider diversity with multiple requests
    print("\n2. Provider diversity (5 requests):")
    providers_used = set()
    try:
        for i in range(5):
            response = await elelem.create_chat_completion(
                messages=[{"role": "user", "content": f"Count to {i+1}"}],
                model="openrouter:openai/gpt-oss-120b",
                max_tokens=10,
                tags=[f"openrouter_diversity_{i}"]
            )
            
            if hasattr(response, 'provider'):
                providers_used.add(response.provider)
                print(f"   Request {i+1}: {response.provider}")
        
        if len(providers_used) > 1:
            print(f"   ✅ Multiple providers used: {list(providers_used)}")
        elif len(providers_used) == 1:
            print(f"   ⚠️  Only one provider used: {list(providers_used)[0]}")
        else:
            print(f"   ❌ No providers tracked")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Verify price-first routing configuration
    print("\n3. Price-first routing configuration:")
    try:
        from elelem.config import Config
        config = Config()
        models = config._models_config.get('models', {})
        model_config = models.get('openrouter:openai/gpt-oss-120b')
        if model_config:
            model_id = model_config.get('model_id', '')
            if ':floor' in model_id:
                print(f"   ✅ Price-first routing enforced (:floor suffix)")
                print(f"      Model ID: {model_id}")
            else:
                print(f"   ❌ No :floor suffix in model ID: {model_id}")
        else:
            print(f"   ❌ Model config not found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Provider statistics aggregation
    print("\n4. Provider statistics aggregation:")
    try:
        overall_stats = elelem.get_stats()
        
        if 'providers' in overall_stats and overall_stats['providers']:
            print("   Provider breakdown:")
            for provider, provider_stats in overall_stats['providers'].items():
                print(f"   • {provider}:")
                print(f"       Requests: {provider_stats['count']}")
                print(f"       Cost: ${provider_stats['total_cost_usd']:.8f}")
                print(f"       Tokens: {provider_stats['total_tokens']}")
            print(f"   ✅ Provider statistics working")
        else:
            print(f"   ❌ No provider statistics available")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ OpenRouter tests complete")


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
        retry_analytics = stats.get('retry_analytics', {})
        
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Total input tokens: {stats['total_input_tokens']}")
        print(f"  Total output tokens: {stats['total_output_tokens']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Total cost: ${stats['total_cost_usd']:.6f}")
        print(f"  Avg duration: {stats['avg_duration_seconds']:.2f}s")
        
        # Display overall retry analytics
        print(f"\n  🔄 Overall Retry Analytics:")
        print(f"    JSON schema retries: {retry_analytics.get('json_schema_retries', 0)}")
        print(f"    JSON parse retries: {retry_analytics.get('json_parse_retries', 0)}")
        print(f"    API JSON validation retries: {retry_analytics.get('api_json_validation_retries', 0)}")
        print(f"    Rate limit retries: {retry_analytics.get('rate_limit_retries', 0)}")
        print(f"    Temperature reductions: {retry_analytics.get('temperature_reductions', 0)}")
        print(f"    Response format removals: {retry_analytics.get('response_format_removals', 0)}")
        print(f"    Fallback model usage: {retry_analytics.get('fallback_model_usage', 0)}")
        print(f"    Final failures: {retry_analytics.get('final_failures', 0)}")
        print(f"    Total retries: {retry_analytics.get('total_retries', 0)}")
        
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
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append("OpenRouter")
    
    if not providers:
        print("\n❌ No API keys found!")
        print("Set one of: OPENAI_API_KEY, GROQ_API_KEY, DEEPINFRA_API_KEY, OPENROUTER_API_KEY")
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
    #await test_6_multi_model_parallel_testing(elelem)
    #await test_5_json_schema_validation(elelem)
    await test_openrouter_features(elelem)  # Add OpenRouter test
    await test_6_stats(elelem)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())