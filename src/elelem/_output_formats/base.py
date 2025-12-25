"""
Base class and shared types for output format handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ParseResult:
    """Result of parsing raw LLM output."""

    success: bool
    data: Any = None  # Parsed data structure (dict, list, etc.)
    content: str = None  # Cleaned/repaired content string
    error: Optional[str] = None  # Error message if parsing failed
    was_repaired: bool = False  # True if content was auto-repaired


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    error: Optional[str] = None  # Human-readable error message
    error_path: Optional[str] = None  # Path to error, e.g., "nodes[0].chapter"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class FormatParseError(Exception):
    """Content cannot be parsed as this format.

    This is an infrastructure error - triggers failover to next candidate.
    """

    def __init__(self, message: str, format_type: str = None):
        super().__init__(message)
        self.format_type = format_type


class FormatSchemaError(Exception):
    """Content parsed successfully but failed schema validation.

    This is potentially fixable by the LLM fixer.
    Triggers: temperature reduction → fixer → retry → failover
    """

    def __init__(self, message: str, content: str = None, format_type: str = None):
        super().__init__(message)
        self.content = content  # The parseable but invalid content
        self.format_type = format_type


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================


class OutputFormat(ABC):
    """Abstract base class for output format handlers.

    Each format (JSON, YAML, CSV) implements this interface to provide:
    - Parsing raw LLM output (with optional repair)
    - Schema validation
    - Prompt instructions generation
    - Fixer prompt generation
    - Serialization back to string

    The base class provides high-level orchestration via process_response().
    """

    # =========================================================================
    # IDENTITY
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Format identifier: 'json', 'yaml', 'csv'"""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Associated file extensions: ['.json'], ['.yaml', '.yml'], ['.csv']"""
        pass

    @property
    def mime_type(self) -> str:
        """MIME type for this format."""
        mime_types = {
            "json": "application/json",
            "yaml": "application/x-yaml",
            "csv": "text/csv",
        }
        return mime_types.get(self.name, "text/plain")

    # =========================================================================
    # PARSING
    # =========================================================================

    @abstractmethod
    def parse(self, content: str, repair: bool = True) -> ParseResult:
        """Parse raw LLM output into structured data.

        Args:
            content: Raw string from LLM response
            repair: If True, attempt auto-repair on parse failure

        Returns:
            ParseResult with parsed data or error details

        Repair strategies by format:
            - JSON: Uses json_repair library for bracket/quote/comma fixes
            - YAML: Tab→space, indent normalization, quote special chars
            - CSV: Delimiter detection, column padding
        """
        pass

    @abstractmethod
    def extract_from_markdown(self, content: str) -> str:
        """Extract format content from markdown code blocks.

        Handles cases where LLM wraps output in ```json``` or ```yaml``` blocks
        despite being told not to.

        Args:
            content: Raw content possibly containing markdown blocks

        Returns:
            Extracted content without markdown wrapper
        """
        pass

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @abstractmethod
    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """Validate parsed data against schema.

        Args:
            data: Parsed data structure
            schema: Format-specific schema definition
                - JSON/YAML: JSON Schema (jsonschema library)
                - CSV: Custom table schema with columns definition

        Returns:
            ValidationResult with is_valid flag and error details
        """
        pass

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    @abstractmethod
    def serialize(self, data: Any, pretty: bool = True) -> str:
        """Convert structured data back to string format.

        Args:
            data: Parsed data structure
            pretty: Whether to use pretty-printing/indentation

        Returns:
            Serialized string in this format
        """
        pass

    # =========================================================================
    # PROMPT GENERATION
    # =========================================================================

    @abstractmethod
    def get_response_instructions(self, schema: Dict[str, Any] = None) -> str:
        """Generate instructions to append to system prompt.

        Tells the LLM how to format its response.

        Args:
            schema: Optional schema to reference in instructions

        Returns:
            Instruction string to append to system message
        """
        pass

    @abstractmethod
    def get_fixer_messages(
        self, invalid_content: str, error: str, schema: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate messages for the fixer LLM.

        Args:
            invalid_content: Content that failed validation
            error: Validation error message
            schema: Schema it should conform to

        Returns:
            List of message dicts with 'role' and 'content'
        """
        pass

    @abstractmethod
    def extract_fixer_result(
        self, response_content: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract fixed content and change description from fixer response.

        Args:
            response_content: Raw response from fixer LLM

        Returns:
            Tuple of (fixed_content, changes_description)
            Returns (None, None) if extraction fails
        """
        pass

    # =========================================================================
    # HIGH-LEVEL ORCHESTRATION
    # =========================================================================

    def process_response(
        self, content: str, schema: Dict[str, Any] = None, request_id: str = None
    ) -> str:
        """Full processing pipeline: extract → parse → validate.

        This is the main entry point called by core.py.

        Args:
            content: Raw LLM response
            schema: Optional schema for validation
            request_id: For logging

        Returns:
            Validated content string (possibly repaired)

        Raises:
            FormatParseError: Cannot parse - triggers failover
            FormatSchemaError: Schema mismatch - triggers fixer
        """
        # Step 1: Extract from markdown code blocks if needed
        extracted = self.extract_from_markdown(content)

        # Step 2: Parse (with repair attempt)
        parse_result = self.parse(extracted)
        if not parse_result.success:
            raise FormatParseError(
                f"{self.name.upper()} parse failed: {parse_result.error}",
                format_type=self.name,
            )

        # Step 3: Validate schema if provided
        if schema:
            validation = self.validate_schema(parse_result.data, schema)
            if not validation.is_valid:
                error_msg = validation.error
                if validation.error_path:
                    error_msg += f" at path: {validation.error_path}"
                raise FormatSchemaError(
                    error_msg, content=parse_result.content, format_type=self.name
                )

        return parse_result.content

    def add_instructions_to_messages(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any] = None,
        supports_system: bool = True,
    ) -> List[Dict[str, str]]:
        """Add format instructions to message list.

        Args:
            messages: Original message list
            schema: Optional schema to include
            supports_system: Whether model supports system messages

        Returns:
            Modified messages with format instructions appended
        """
        instructions = self.get_response_instructions(schema)
        modified = [dict(m) for m in messages]  # Shallow copy

        if supports_system:
            # Append to existing system message or create one
            system_found = False
            for msg in modified:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + instructions
                    system_found = True
                    break

            if not system_found:
                modified.insert(
                    0, {"role": "system", "content": "You are a helpful assistant." + instructions}
                )
        else:
            # Add as user message for models without system support
            modified.append({"role": "user", "content": instructions.strip()})

        return modified
