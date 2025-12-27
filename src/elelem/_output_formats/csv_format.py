"""Multi-table CSV output format handler."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml  # For schema display in fixer

from .base import OutputFormat, ParseResult, ValidationResult

logger = logging.getLogger("elelem")


class CsvFormat(OutputFormat):
    """Multi-table CSV output format with delimiter auto-detection."""

    def __init__(self, delimiter: str = ";"):
        """Initialize with preferred delimiter.

        Args:
            delimiter: Primary delimiter (default: semicolon to avoid comma issues)
        """
        self.delimiter = delimiter
        self.fallback_delimiters = [";", ",", "\t", "|"]

    @property
    def name(self) -> str:
        return "csv"

    @property
    def file_extensions(self) -> List[str]:
        return [".csv"]

    # =========================================================================
    # PARSING
    # =========================================================================

    def parse(self, content: str, repair: bool = True) -> ParseResult:
        """Parse multi-table CSV with delimiter auto-detection."""
        # Try primary delimiter first
        result = self._try_parse(content, self.delimiter)
        if result.success and self._looks_valid(result.data):
            return result

        if not repair:
            return ParseResult(success=False, error="Invalid CSV structure")

        # Try fallback delimiters
        for delim in self.fallback_delimiters:
            if delim == self.delimiter:
                continue
            result = self._try_parse(content, delim)
            if result.success and self._looks_valid(result.data):
                logger.debug(f"CSV auto-detected delimiter: '{delim}'")
                # Re-serialize with correct delimiter
                normalized = self.serialize(result.data)
                return ParseResult(
                    success=True,
                    data=result.data,
                    content=normalized,
                    was_repaired=True,
                )

        return ParseResult(success=False, error="Could not detect valid CSV structure")

    def _try_parse(self, content: str, delimiter: str) -> ParseResult:
        """Parse multi-table CSV with given delimiter."""
        tables: Dict[str, List[Dict[str, str]]] = {}
        current_table: Optional[str] = None
        headers: Optional[List[str]] = None

        try:
            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("###TABLE:"):
                    current_table = line.replace("###TABLE:", "").strip()
                    tables[current_table] = []
                    headers = None
                elif current_table is not None:
                    values = [v.strip() for v in line.split(delimiter)]

                    if headers is None:
                        headers = values
                    else:
                        # Pad row to match header length (repair missing columns)
                        while len(values) < len(headers):
                            values.append("")
                        row = dict(zip(headers, values[: len(headers)]))
                        tables[current_table].append(row)

            if tables:
                return ParseResult(success=True, data=tables, content=content)

            return ParseResult(success=False, error="No tables found")
        except Exception as e:
            return ParseResult(success=False, error=str(e))

    def _looks_valid(self, tables: Dict) -> bool:
        """Heuristic check if parsed tables are reasonable."""
        if not tables:
            return False

        for name, rows in tables.items():
            if not rows:
                continue
            first_row = rows[0]
            # Check that we have at least one column
            if len(first_row) < 1:
                return False
            # Check column names aren't suspiciously long
            for col in first_row.keys():
                if len(col) > 50 or "\n" in col:
                    return False

        return True

    def extract_from_markdown(self, content: str) -> str:
        """Extract CSV from markdown code blocks."""
        patterns = [
            r"```csv\s*\n(.*?)\n```",
            r"```\s*\n(###TABLE:.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return content

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_schema(
        self, data: Dict[str, List[Dict]], schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate tables against CSV schema definition."""
        errors = []

        for table_name, table_schema in schema.get("tables", {}).items():
            # Check required tables
            if table_schema.get("required", False) and table_name not in data:
                errors.append(f"Missing required table: {table_name}")
                continue

            if table_name not in data:
                continue

            # Validate columns in each row
            columns = table_schema.get("columns", {})
            for row_idx, row in enumerate(data[table_name]):
                for col_name, col_schema in columns.items():
                    # Required column check
                    if col_schema.get("required", False) and col_name not in row:
                        errors.append(
                            f"{table_name}[{row_idx}]: missing required column '{col_name}'"
                        )
                        continue

                    if col_name not in row or not row[col_name]:
                        continue

                    value = row[col_name]

                    # Type validation
                    col_type = col_schema.get("type", "string")
                    if col_type == "integer":
                        try:
                            int(value)
                        except ValueError:
                            errors.append(
                                f"{table_name}[{row_idx}].{col_name}: expected integer, got '{value}'"
                            )

                    # Enum validation
                    if "enum" in col_schema and value not in col_schema["enum"]:
                        errors.append(
                            f"{table_name}[{row_idx}].{col_name}: '{value}' not in {col_schema['enum']}"
                        )

                    # Pattern validation
                    if "pattern" in col_schema and not re.match(
                        col_schema["pattern"], value
                    ):
                        errors.append(
                            f"{table_name}[{row_idx}].{col_name}: doesn't match pattern"
                        )

        if errors:
            return ValidationResult(is_valid=False, error=" | ".join(errors[:5]))
        return ValidationResult(is_valid=True)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def serialize(self, data: Dict[str, List[Dict]], pretty: bool = True) -> str:
        """Serialize tables dict back to CSV string."""
        lines = []
        for table_name, rows in data.items():
            lines.append(f"###TABLE:{table_name}")
            if rows:
                headers = list(rows[0].keys())
                lines.append(self.delimiter.join(headers))
                for row in rows:
                    values = [str(row.get(h, "")) for h in headers]
                    lines.append(self.delimiter.join(values))
            lines.append("")  # Empty line between tables
        return "\n".join(lines).strip()

    # =========================================================================
    # PROMPT GENERATION
    # =========================================================================

    def get_response_instructions(self, schema: Dict[str, Any] = None) -> str:
        """Generate CSV response instructions."""
        tables_info = ""
        multi_value_info = ""
        multi_value_cols = []

        if schema and "tables" in schema:
            tables_info = "\n\nRequired tables:\n"
            for table_name, spec in schema["tables"].items():
                cols = list(spec.get("columns", {}).keys())
                req = " (required)" if spec.get("required") else ""
                tables_info += f"- ###TABLE:{table_name}{req}\n  Columns: {', '.join(cols)}\n"

                # Collect multi-value columns
                for col_name, col_spec in spec.get("columns", {}).items():
                    if col_spec.get("multi_value"):
                        fmt = col_spec.get("multi_value_format", "")
                        example = col_spec.get("multi_value_example", "")
                        multi_value_cols.append((table_name, col_name, fmt, example))

        if multi_value_cols:
            multi_value_info = "\n\nFor columns with multiple values, use pipe (|) as separator:\n"
            for table_name, col_name, fmt, example in multi_value_cols:
                if example:
                    multi_value_info += f"- {table_name}.{col_name}: {example}\n"
                elif fmt == "id:name":
                    multi_value_info += f"- {table_name}.{col_name}: pipe-separated id:name pairs (e.g., id1:Name1|id2:Name2)\n"
                else:
                    multi_value_info += f"- {table_name}.{col_name}: pipe-separated values (e.g., val1|val2|val3)\n"

        # Always add general multi-value guidance
        general_mv_note = ""
        if not multi_value_cols:
            general_mv_note = (
                "\n\nIf a cell needs multiple values, use pipe (|) as separator "
                "(e.g., val1|val2|val3). Never use semicolon for lists within a cell."
            )

        return (
            f"\n\nCRITICAL: Respond with CSV tables using semicolon (;) as delimiter. "
            f"Each table starts with ###TABLE:tablename on its own line, "
            f"followed by a header row, then data rows. "
            f"For empty/null values, leave the cell empty (do not use '-' or 'N/A'). "
            f"Do not use quotes around values unless they contain semicolons. "
            f"Do not wrap in markdown code blocks."
            f"{tables_info}"
            f"{multi_value_info}"
            f"{general_mv_note}"
        )

    def get_fixer_messages(
        self, invalid_content: str, error: str, schema: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate CSV fixer messages."""
        system = """You are a CSV fixer. Your task is to repair invalid CSV tables so they pass validation.

INSTRUCTIONS:
1. Read the validation error carefully
2. Fix ONLY what the error describes - make minimal changes
3. Use semicolon (;) as delimiter
4. Ensure all required columns are present

OUTPUT FORMAT:
Return corrected CSV tables starting with:
###TABLE:_changes
change
description of what you fixed

Then all the corrected tables:
###TABLE:table_name
column1;column2;...
value1;value2;..."""

        user = f"""Fix these CSV tables that failed validation.

VALIDATION ERROR:
{error}

EXPECTED SCHEMA:
{yaml.dump(schema, default_flow_style=False)}

INVALID CSV:
{invalid_content}"""

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def extract_fixer_result(
        self, response_content: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract fixed CSV and changes from fixer response."""
        changes = None
        content = response_content.strip()

        # Look for _changes table
        if "###TABLE:_changes" in content:
            parts = content.split("###TABLE:_changes")
            if len(parts) > 1:
                changes_section = parts[1].split("###TABLE:")[0]
                lines = [
                    line.strip()
                    for line in changes_section.strip().split("\n")
                    if line.strip()
                ]
                if len(lines) > 1:
                    changes = lines[1]  # Skip header row
                # Reconstruct without _changes table
                remaining = parts[1].split("###TABLE:")[1:]
                content = "###TABLE:" + "###TABLE:".join(remaining)

        if not content.strip():
            return None, None

        return content.strip(), changes
