"""
rag_pipeline.py — Core RAG orchestration: retrieve schema → generate SQL → execute → summarize.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.claude_client import generate_sql, summarize_results
from app.db import SQLExecutionError, execute_readonly_query, get_engine
from app.embeddings import retrieve_relevant_schema
from app.models import QueryHistory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL safety constants
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = frozenset([
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
    "TRUNCATE", "EXEC", "EXECUTE",
])
MAX_QUESTION_LENGTH = 500
MAX_SQL_LENGTH = 2000


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_question(question: str) -> None:
    """
    Raise ValueError if the question is empty or exceeds the character limit.

    Args:
        question: The raw user input string.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question exceeds {MAX_QUESTION_LENGTH} character limit "
            f"({len(question)} chars)."
        )


def _validate_sql(sql: str) -> None:
    """
    Raise ValueError if the SQL fails any safety check.

    Checks:
    - Must start with SELECT (case-insensitive)
    - Must not contain forbidden DDL/DML keywords
    - Must not contain comment markers or semicolons
    - Must not exceed MAX_SQL_LENGTH

    Args:
        sql: The SQL string returned by Claude.
    """
    stripped = sql.strip()

    if len(stripped) > MAX_SQL_LENGTH:
        raise ValueError(
            f"Generated SQL exceeds {MAX_SQL_LENGTH} character limit."
        )

    if not re.match(r"(?i)^\s*SELECT\b", stripped):
        raise ValueError("SQL must begin with SELECT.")

    # Check forbidden keywords as whole words
    upper_sql = stripped.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, upper_sql):
            raise ValueError(f"SQL contains forbidden keyword: {keyword}")

    # Block inline comments and statement terminators
    if "--" in stripped:
        raise ValueError("SQL must not contain comment sequences (--).")
    if ";" in stripped:
        raise ValueError("SQL must not contain semicolons.")


def _save_to_history(
    question: str,
    sql: str,
    result_count: int,
    summary: str,
) -> None:
    """
    Persist a completed query to the query_history table.

    Args:
        question:     The original user question.
        sql:          The executed SQL query.
        result_count: Number of rows returned.
        summary:      The Claude-generated plain-English summary.
    """
    try:
        engine = get_engine()
        with Session(engine) as session:
            record = QueryHistory(
                question=question,
                sql=sql,
                result_count=result_count,
                summary=summary,
            )
            session.add(record)
            session.commit()
            logger.info("Saved query to history (id=%s).", record.id)
    except Exception as exc:
        # Non-fatal — log and continue
        logger.error("Failed to save query history: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def handle_question(question: str) -> dict[str, Any]:
    """
    Full RAG pipeline: validate → retrieve schema → generate SQL → execute → summarize.

    Args:
        question: The user's natural language question.

    Returns:
        A dict containing:
            - question (str)
            - sql (str)
            - results (list[dict])
            - summary (str)
            - result_count (int)
            - timestamp (str, ISO 8601)

    Raises:
        ValueError:        For invalid input or SQL safety violations.
        SQLExecutionError: If the database query fails.
        RuntimeError:      For unexpected pipeline errors.
    """
    # Step 1: Validate input
    _validate_question(question)
    logger.info("Handling question: '%.100s'", question)

    # Step 2: Retrieve relevant schema chunks via semantic search
    try:
        schema_chunks = retrieve_relevant_schema(question, top_k=15)
    except Exception as exc:
        logger.error("Schema retrieval failed: %s", exc)
        raise RuntimeError(
            "Failed to retrieve schema context. Please ensure the schema has been embedded."
        ) from exc

    if not schema_chunks:
        raise RuntimeError(
            "No schema context found. Please run the schema embedding script first."
        )

    # Step 3: Format schema context string
    schema_context = "\n\n".join(schema_chunks)

    # Step 4: Generate SQL via Claude
    try:
        sql = generate_sql(question, schema_context)
    except Exception as exc:
        logger.error("SQL generation failed: %s", exc)
        raise RuntimeError(
            "Failed to generate SQL. Please check your Anthropic API key and try again."
        ) from exc

    # Step 5: Handle CANNOT_ANSWER
    if sql.strip().upper() == "CANNOT_ANSWER":
        logger.info("Claude could not answer the question from the schema.")
        return {
            "question": question,
            "sql": None,
            "results": [],
            "summary": (
                "I'm unable to answer that question based on the available database schema. "
                "Please rephrase your question or ask about data that exists in the database."
            ),
            "result_count": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "CANNOT_ANSWER",
        }

    # Step 6: Validate generated SQL
    try:
        _validate_sql(sql)
    except ValueError as exc:
        logger.warning("SQL safety validation failed: %s | SQL: %s", exc, sql)
        raise ValueError(f"Generated SQL failed safety validation: {exc}") from exc

    # Step 7: Execute SQL (read-only)
    try:
        results = execute_readonly_query(sql)
    except SQLExecutionError as exc:
        logger.error("SQL execution error: %s", exc)
        raise SQLExecutionError(
            f"The query could not be executed: {exc}"
        ) from exc

    # Step 8: Summarize results
    try:
        summary = summarize_results(question, sql, results)
    except Exception as exc:
        logger.warning("Summarization failed, using fallback: %s", exc)
        summary = f"Query returned {len(results)} row(s)."

    # Step 9: Save to history
    _save_to_history(question, sql, len(results), summary)

    # Step 10: Return response dict
    return {
        "question": question,
        "sql": sql,
        "results": results,
        "summary": summary,
        "result_count": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
