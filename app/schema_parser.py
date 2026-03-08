"""
schema_parser.py — Converts raw schema metadata into natural-language text chunks
suitable for embedding and semantic retrieval.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_schema_to_chunks(schema_metadata: list[dict[str, Any]]) -> list[str]:
    """
    Convert schema metadata dicts into natural-language text chunks for embedding.

    Each chunk describes one table: its columns (with types, PK/FK annotations),
    foreign key relationships, and a sample of real data values.

    Args:
        schema_metadata: Output of db.get_schema_metadata() — a list of table dicts.

    Returns:
        A list of strings, one per table, ready to be embedded.
    """
    chunks: list[str] = []

    for table in schema_metadata:
        table_name: str = table["table_name"]
        columns: list[dict] = table.get("columns", [])
        foreign_keys: list[dict] = table.get("foreign_keys", [])
        sample_rows: list[dict] = table.get("sample_rows", [])

        # Build a FK lookup: column_name → "referenced_table.referenced_column"
        fk_lookup: dict[str, str] = {
            fk["column"]: f"{fk['referenced_table']}.{fk['referenced_column']}"
            for fk in foreign_keys
        }

        # Format columns
        col_parts: list[str] = []
        for col in columns:
            name = col["name"]
            dtype = col["type"]
            annotations: list[str] = []
            if col.get("is_primary"):
                annotations.append("PK")
            if name in fk_lookup:
                annotations.append(f"FK→{fk_lookup[name]}")
            if not col.get("is_nullable"):
                annotations.append("NOT NULL")
            annotation_str = ", ".join(annotations)
            col_parts.append(f"{name} ({dtype}{', ' + annotation_str if annotation_str else ''})")

        columns_str = ", ".join(col_parts)

        # Format sample values
        sample_str = ""
        if sample_rows:
            sample_lines: list[str] = []
            for row in sample_rows[:3]:
                row_preview = ", ".join(
                    f"{k}={repr(v)}" for k, v in list(row.items())[:6]
                )
                sample_lines.append(f"  {{{row_preview}}}")
            sample_str = " | Sample rows: " + "; ".join(sample_lines)

        # Format FK relationships sentence
        fk_str = ""
        if foreign_keys:
            fk_descriptions = [
                f"{fk['column']} references {fk['referenced_table']}({fk['referenced_column']})"
                for fk in foreign_keys
            ]
            fk_str = " | Relationships: " + ", ".join(fk_descriptions)

        chunk = (
            f"Table: {table_name} | "
            f"Columns: {columns_str}"
            f"{fk_str}"
            f"{sample_str}"
        )
        chunks.append(chunk)
        logger.debug("Parsed chunk for table '%s' (%d chars).", table_name, len(chunk))

    logger.info("Parsed %d schema chunks from %d tables.", len(chunks), len(schema_metadata))
    return chunks
