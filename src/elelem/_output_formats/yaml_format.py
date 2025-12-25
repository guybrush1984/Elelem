"""YAML output format handler."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jsonschema import ValidationError, validate

from .base import OutputFormat, ParseResult, ValidationResult

logger = logging.getLogger("elelem")


class YamlFormat(OutputFormat):
    """YAML output format with custom repair heuristics."""

    @property
    def name(self) -> str:
        return "yaml"

    @property
    def file_extensions(self) -> List[str]:
        return [".yaml", ".yml"]

    # =========================================================================
    # PARSING
    # =========================================================================

    def parse(self, content: str, repair: bool = True) -> ParseResult:
        """Parse YAML with optional repair heuristics."""
        try:
            data = yaml.safe_load(content)
            return ParseResult(success=True, data=data, content=content)
        except yaml.YAMLError as e:
            if not repair:
                return ParseResult(success=False, error=str(e))

            # Attempt repair with heuristics
            repaired = self._attempt_repair(content)
            if repaired and repaired != content:
                try:
                    data = yaml.safe_load(repaired)
                    logger.debug("YAML repair successful")
                    return ParseResult(
                        success=True,
                        data=data,
                        content=repaired,
                        was_repaired=True,
                    )
                except yaml.YAMLError:
                    pass

            return ParseResult(success=False, error=str(e))

    def _attempt_repair(self, content: str) -> Optional[str]:
        """Apply YAML repair heuristics."""
        repaired = content

        # Fix 1: Convert tabs to spaces (YAML spec requires spaces)
        repaired = repaired.replace("\t", "  ")

        # Fix 2: Normalize inconsistent indentation to multiples of 2
        lines = repaired.split("\n")
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip(" ")
            indent = len(line) - len(stripped)
            normalized_indent = (indent // 2) * 2
            fixed_lines.append(" " * normalized_indent + stripped)
        repaired = "\n".join(fixed_lines)

        # Fix 3: Remove common non-YAML prefixes from LLM
        prefixes = [
            r"^Here\'?s?\s+(the\s+)?YAML:?\s*\n",
            r"^```ya?ml\s*\n",
            r"\n```\s*$",
        ]
        for pattern in prefixes:
            repaired = re.sub(pattern, "", repaired, flags=re.IGNORECASE)

        return repaired

    def extract_from_markdown(self, content: str) -> str:
        """Extract YAML from markdown code blocks."""
        patterns = [
            r"```ya?ml\s*\n(.*?)\n```",
            r"```ya?ml\s*\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return content

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """Validate against JSON Schema (YAML is JSON superset)."""
        try:
            validate(instance=data, schema=schema)
            return ValidationResult(is_valid=True)
        except ValidationError as e:
            path = ".".join(str(p) for p in e.path) if e.path else None
            return ValidationResult(is_valid=False, error=e.message, error_path=path)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def serialize(self, data: Any, pretty: bool = True) -> str:
        """Serialize to YAML string."""
        return yaml.dump(
            data,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    # =========================================================================
    # PROMPT GENERATION
    # =========================================================================

    def get_response_instructions(self, schema: Dict[str, Any] = None) -> str:
        """Generate YAML response instructions."""
        return (
            "\n\nCRITICAL: You must respond with ONLY clean YAML - "
            "no markdown, no code blocks, no extra text. "
            "Do not wrap the YAML in ```yaml``` or ```yml``` blocks. "
            "Return raw, valid YAML that can be parsed directly. "
            "Use proper YAML syntax with correct indentation (2 spaces per level). "
            "Any non-YAML content will cause a parsing error."
        )

    def get_fixer_messages(
        self, invalid_content: str, error: str, schema: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate YAML fixer messages."""
        system = """You are a YAML fixer. Your task is to repair invalid YAML so it passes schema validation.

INSTRUCTIONS:
1. Read the validation error carefully
2. Fix ONLY what the error describes - make minimal changes
3. Ensure proper YAML indentation (2 spaces per level)
4. Use proper YAML syntax

OUTPUT FORMAT:
Return a YAML document with two top-level keys:
changes: "Brief description of what you fixed"
fixed:
  ... the complete corrected YAML structure ..."""

        user = f"""Fix this YAML that failed schema validation.

VALIDATION ERROR:
{error}

EXPECTED SCHEMA:
{yaml.dump(schema, default_flow_style=False)}

INVALID YAML:
{invalid_content}"""

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def extract_fixer_result(
        self, response_content: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract fixed YAML and changes from fixer response."""
        try:
            # First extract from markdown if present
            content = self.extract_from_markdown(response_content)

            wrapper = yaml.safe_load(content)
            if isinstance(wrapper, dict) and "fixed" in wrapper:
                changes = wrapper.get("changes", "")
                fixed_yaml = yaml.dump(
                    wrapper["fixed"],
                    allow_unicode=True,
                    default_flow_style=False,
                )
                return fixed_yaml, changes
        except Exception:
            pass

        return None, None
