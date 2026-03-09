"""
claude_client.py — Anthropic Claude API wrapper for SQL generation and result summarization.
"""

import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS_SQL = 1024
_MAX_TOKENS_SUMMARY = 512


def _get_client() -> anthropic.Anthropic:
    """Return the singleton Anthropic client."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        _client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client initialized.")
    return _client


_SQL_SYSTEM_PROMPT = """You are an expert PostgreSQL SQL generator.

You receive a natural language question and relevant database schema context.

Rules:
- Respond ONLY with a valid, executable PostgreSQL SELECT query.
- No markdown, no code fences, no explanation.
- Do NOT include a semicolon at the end.
- Never use DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE, EXEC, or EXECUTE.
- CRITICAL: Use table and column names EXACTLY as they appear in the schema context. Do NOT infer, guess, or use common variations. For example, if the schema shows "invoice_line", you must write "invoice_line" — never "invoiceline" or "InvoiceLine".
- CRITICAL: If a table or column name contains any uppercase letter (e.g. "salePrice", "userId", "ProductId"), you MUST wrap it in double quotes in the SQL (e.g. "salePrice", "userId", "ProductId"). Unquoted identifiers are lowercased by PostgreSQL and will cause errors.
- Only reference tables and columns that are explicitly listed in the provided schema. If a table you need is not in the schema, respond with CANNOT_ANSWER.
- If the question cannot be answered at all from the given schema, respond with exactly: CANNOT_ANSWER
- If the question is ambiguous but a reasonable interpretation exists using the available tables, write the SQL for that interpretation.

You will be given:
1. The user's question
2. Relevant schema chunks describing the tables and columns
"""

_SUMMARY_SYSTEM_PROMPT = """You are a helpful data analyst.

Given a user question, the SQL query that was executed, and the raw query results,
write a clear and concise natural language summary of the findings.

Rules:
- Be specific: include actual numbers, names, and values from the results.
- Keep it under 4 sentences.
- Do not mention SQL or technical details unless critical to understanding.
- If the result set is empty, say so clearly and suggest why.
"""


def generate_sql(question: str, schema_context: str) -> str:
    """
    Ask Claude to generate a PostgreSQL SELECT query for the given question.

    Args:
        question:       The user's natural language question.
        schema_context: Relevant schema chunks joined as a single string.

    Returns:
        A raw SQL string, or the literal string "CANNOT_ANSWER".
    """
    client = _get_client()

    # Extract exact table names from the schema context to pin them explicitly.
    import re as _re
    table_names = _re.findall(r"^Table:\s+(\S+?)\s*\|", schema_context, _re.MULTILINE)
    table_pin = ""
    if table_names:
        table_pin = (
            "EXACT TABLE NAMES YOU MUST USE (copy these character-for-character):\n"
            + ", ".join(table_names)
            + "\n\n"
        )

    user_message = (
        f"{table_pin}"
        f"Schema context:\n{schema_context}\n\n"
        f"Question: {question}"
    )

    logger.info("Requesting SQL generation for question: '%.80s'", question)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS_SQL,
        system=_SQL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    sql = response.content[0].text.strip()
    logger.info("Claude returned SQL (%.200s)", sql)
    return sql


def summarize_results(
    question: str,
    sql: str,
    results: list[dict[str, Any]],
) -> str:
    """
    Ask Claude to summarize query results in plain English.

    Args:
        question: The original user question.
        sql:      The SQL that was executed.
        results:  The list of row dicts returned by the database.

    Returns:
        A natural language summary string.
    """
    client = _get_client()

    # Truncate results to avoid exceeding token limits
    results_preview = results[:50]
    results_str = "\n".join(str(row) for row in results_preview)
    if len(results) > 50:
        results_str += f"\n... and {len(results) - 50} more rows."

    user_message = (
        f"User question: {question}\n\n"
        f"SQL executed:\n{sql}\n\n"
        f"Results ({len(results)} rows total):\n{results_str}"
    )

    logger.info("Requesting result summarization (%d rows).", len(results))

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS_SUMMARY,
        system=_SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    summary = response.content[0].text.strip()
    logger.info("Claude summarization complete.")
    return summary
