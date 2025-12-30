"""JSON output format handler."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from json_repair import repair_json
from jsonschema import ValidationError, validate

from ..fixer_prompts import build_fixer_messages
from .base import FixerResult, OutputFormat, ParseResult, ValidationResult

logger = logging.getLogger("elelem")


class JsonFormat(OutputFormat):
    """JSON output format with json_repair integration."""

    @property
    def name(self) -> str:
        return "json"

    @property
    def file_extensions(self) -> List[str]:
        return [".json"]

    # =========================================================================
    # PARSING
    # =========================================================================

    def parse(self, content: str, repair: bool = True) -> ParseResult:
        """Parse JSON with optional repair using json_repair library."""
        try:
            data = json.loads(content)
            return ParseResult(success=True, data=data, content=content)
        except json.JSONDecodeError as e:
            if not repair:
                return ParseResult(success=False, error=str(e))

            # Attempt repair with json_repair library
            try:
                repaired = repair_json(content, return_objects=True)

                # Validate repair result is meaningful
                if repaired in ("", [], {}):
                    logger.debug("JSON repair returned empty result")
                    return ParseResult(success=False, error=str(e))

                repaired_str = json.dumps(repaired, ensure_ascii=False)
                logger.debug(f"JSON repair successful: {str(e)[:100]}")
                return ParseResult(
                    success=True,
                    data=repaired,
                    content=repaired_str,
                    was_repaired=True,
                )
            except Exception as repair_error:
                logger.debug(f"JSON repair failed: {repair_error}")
                return ParseResult(success=False, error=str(e))

    def extract_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks."""
        patterns = [
            r"```json\s*\n(.*?)\n```",
            r"```json\s*\n(.*?)```",
            r"```\s*\n(\{.*?\})\n```",
            r"```\s*\n(\[.*?\])\n```",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return content

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """Validate against JSON Schema using jsonschema library."""
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
        """Serialize to JSON string."""
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)

    # =========================================================================
    # PROMPT GENERATION
    # =========================================================================

    def get_response_instructions(self, schema: Dict[str, Any] = None) -> str:
        """Generate JSON response instructions."""
        instructions = (
            "\n\nCRITICAL: You must respond with ONLY a clean JSON object - "
            "no markdown, no code blocks, no extra text. "
            "Do not wrap the JSON in ```json``` blocks or any other formatting. "
            "Return raw, valid JSON that can be parsed directly. "
            "Start your response with { and end with }. "
            "Any non-JSON content will cause a parsing error."
        )

        # Include schema details if provided
        if schema:
            schema_str = json.dumps(schema, indent=2)
            instructions += (
                "\n\n=== REQUIRED OUTPUT FORMAT ===\n"
                "Your response MUST conform to this exact JSON schema:\n\n"
                f"{schema_str}\n\n"
                "Follow the schema precisely:\n"
                "- Include all required fields\n"
                "- Use correct data types (string, number, boolean, array, object)\n"
                "- Do not add extra fields unless allowed by the schema\n"
                "- Respect any constraints (enums, patterns, min/max values)\n"
                "=== END REQUIRED FORMAT ==="
            )

        return instructions

    def get_fixer_messages(
        self, invalid_content: str, error: str, schema: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate JSON fixer messages from YAML template."""
        return build_fixer_messages(
            format_name="json",
            content=invalid_content,
            error=error,
            schema=schema,
        )

    def extract_fixer_result(self, response_content: str) -> FixerResult:
        """Extract fixed JSON from fixer response."""
        content = response_content.strip()

        # Clean markdown if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:])

        # Find JSON boundaries
        start = content.find("{")
        end = content.rfind("}") + 1

        if start < 0 or end <= start:
            return FixerResult(content=None, is_fixable=True, changes=None)

        json_str = content[start:end]

        try:
            wrapper = json.loads(json_str)
            if isinstance(wrapper, dict) and "fixed" in wrapper:
                is_fixable = wrapper.get("fixable", True)
                changes = wrapper.get("changes", "")
                fixed = wrapper.get("fixed")

                if fixed is None:
                    return FixerResult(
                        content=None, is_fixable=is_fixable, changes=changes
                    )

                fixed_json = json.dumps(fixed, ensure_ascii=False)
                return FixerResult(
                    content=fixed_json, is_fixable=is_fixable, changes=changes
                )
        except json.JSONDecodeError:
            pass

        # Fallback: couldn't parse JSON wrapper
        return FixerResult(content=None, is_fixable=True, changes=None)
