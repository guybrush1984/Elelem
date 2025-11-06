"""
Example: Using YAML response format with Elelem

This demonstrates how to request YAML-formatted responses from models
with optional schema validation.
"""

import asyncio
import yaml
from elelem import Elelem


async def main():
    # Initialize Elelem
    elelem = Elelem()

    # Example 1: Basic YAML response without schema
    print("=" * 60)
    print("Example 1: Basic YAML Response (No Schema)")
    print("=" * 60)

    response1 = await elelem.create_chat_completion(
        model="groq:llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Provide a simple user profile in YAML format with name, age, and hobbies."}
        ],
        yaml_schema={}  # Request YAML mode without validation
    )

    yaml_content1 = response1.choices[0].message.content
    print("\nYAML Response:")
    print(yaml_content1)
    print("\nParsed YAML:")
    print(yaml.safe_load(yaml_content1))

    # Example 2: YAML response with schema validation
    print("\n" + "=" * 60)
    print("Example 2: YAML Response with Schema Validation")
    print("=" * 60)

    # Define a JSON Schema for validation
    user_schema = {
        "type": "object",
        "required": ["name", "age", "email", "active"],
        "properties": {
            "name": {"type": "string", "minLength": 2},
            "age": {"type": "number", "minimum": 0, "maximum": 150},
            "email": {"type": "string", "format": "email"},
            "active": {"type": "boolean"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    response2 = await elelem.create_chat_completion(
        model="groq:llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Create a user profile with name, age (25), email, active status (true), and a list of hobbies."}
        ],
        yaml_schema=user_schema
    )

    yaml_content2 = response2.choices[0].message.content
    print("\nYAML Response (validated against schema):")
    print(yaml_content2)
    print("\nParsed YAML:")
    user_data = yaml.safe_load(yaml_content2)
    print(user_data)
    print(f"\nValidation: Age is {type(user_data['age']).__name__}, active is {type(user_data['active']).__name__}")

    # Example 3: YAML response with schema in prompt
    print("\n" + "=" * 60)
    print("Example 3: Including Schema in Prompt")
    print("=" * 60)

    config_schema = {
        "type": "object",
        "required": ["database", "cache", "features"],
        "properties": {
            "database": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "number"},
                    "name": {"type": "string"}
                }
            },
            "cache": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "ttl": {"type": "number"}
                }
            },
            "features": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    response3 = await elelem.create_chat_completion(
        model="groq:llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Generate a sample application configuration."}
        ],
        yaml_schema=config_schema,
        enforce_schema_in_prompt=True  # Include schema in the prompt
    )

    yaml_content3 = response3.choices[0].message.content
    print("\nYAML Response (schema included in prompt):")
    print(yaml_content3)
    print("\nParsed YAML:")
    print(yaml.safe_load(yaml_content3))

    # Show metrics
    print("\n" + "=" * 60)
    print("Elelem Metrics")
    print("=" * 60)
    print(f"Total cost: ${response3.elelem_metrics['costs_usd']['total_cost_usd']:.6f}")
    print(f"Tokens: {response3.elelem_metrics['tokens']['input']} input, {response3.elelem_metrics['tokens']['output']} output")


if __name__ == "__main__":
    asyncio.run(main())
