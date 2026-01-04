"""Output format handling for Elelem.

Provides unified interface for JSON, YAML, and CSV output formats with:
- Parsing and repair
- Schema validation
- Prompt instructions
- Fixer integration
"""

from typing import Dict, List, Optional

from .base import (
    FormatParseError,
    FormatSchemaError,
    OutputFormat,
    ParseResult,
    ValidationResult,
)
from .csv_format import CsvFormat
from .json_format import JsonFormat
from .yaml_format import YamlFormat


class FormatRegistry:
    """Registry of available output formats.

    Usage:
        format = FormatRegistry.get("json")
        format = FormatRegistry.detect_from_schema(schema)
    """

    _formats: Dict[str, OutputFormat] = {}
    _default = "json"

    @classmethod
    def register(cls, format_instance: OutputFormat) -> None:
        """Register a format handler."""
        cls._formats[format_instance.name] = format_instance

    @classmethod
    def get(cls, name: str) -> OutputFormat:
        """Get format handler by name."""
        if name not in cls._formats:
            available = list(cls._formats.keys())
            raise ValueError(f"Unknown format: {name}. Available: {available}")
        return cls._formats[name]

    @classmethod
    def detect_from_schema(cls, schema: Dict) -> Optional[OutputFormat]:
        """Auto-detect format from schema structure.

        Detection rules:
        - "tables" key → CSV format
        - "type"/"properties"/"$schema" → JSON format (works for YAML too)
        - None → returns None
        """
        if schema is None:
            return None

        # CSV schema has "tables" key
        if "tables" in schema:
            return cls.get("csv")

        # JSON Schema indicators
        if any(key in schema for key in ["type", "properties", "$schema", "items"]):
            return cls.get("json")

        # Default to JSON
        return cls.get(cls._default)

    @classmethod
    def list_formats(cls) -> List[str]:
        """List available format names."""
        return list(cls._formats.keys())

    @classmethod
    def get_by_extension(cls, extension: str) -> Optional[OutputFormat]:
        """Get format handler by file extension."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for format_handler in cls._formats.values():
            if ext in format_handler.file_extensions:
                return format_handler
        return None


# Register default formats
FormatRegistry.register(JsonFormat())
FormatRegistry.register(YamlFormat())
FormatRegistry.register(CsvFormat())


__all__ = [
    # Base
    "OutputFormat",
    "ParseResult",
    "ValidationResult",
    "FormatParseError",
    "FormatSchemaError",
    # Registry
    "FormatRegistry",
    # Implementations
    "JsonFormat",
    "YamlFormat",
    "CsvFormat",
]
