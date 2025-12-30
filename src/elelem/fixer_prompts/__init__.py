"""Fixer prompt loader for Elelem.

This module loads fixer prompts from YAML files, making them easy to
edit and test independently.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def get_prompts_dir() -> Path:
    """Get the directory containing fixer prompt YAML files."""
    return Path(__file__).parent


def load_fixer_prompt(format_name: str) -> Dict[str, str]:
    """Load fixer prompt for a specific format.

    Args:
        format_name: Format identifier ('json', 'yaml', 'csv')

    Returns:
        Dict with 'system' and 'user' prompt templates

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_file = get_prompts_dir() / f"{format_name}.yaml"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Fixer prompt not found: {prompt_file}")

    with open(prompt_file) as f:
        return yaml.safe_load(f)


def build_fixer_messages(
    format_name: str,
    content: str,
    error: str,
    schema: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build fixer messages from template.

    Args:
        format_name: Format identifier ('json', 'yaml', 'csv')
        content: Invalid content to fix
        error: Validation error message
        schema: Schema definition

    Returns:
        List of message dicts ready for LLM API call
    """
    prompt = load_fixer_prompt(format_name)

    # Format schema based on format type
    if format_name == "json":
        schema_str = json.dumps(schema, indent=2)
    else:
        schema_str = yaml.dump(schema, default_flow_style=False)

    # Build messages with substituted variables
    system_content = prompt["system"]
    user_content = prompt["user"].format(
        error=error,
        schema=schema_str,
        content=content,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
