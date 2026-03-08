"""
db.py — PostgreSQL connection, read-only query execution, and schema metadata extraction.
"""

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class SQLExecutionError(Exception):
    """Raised when a SQL query fails during execution."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _build_database_url() -> str:
    """Build the PostgreSQL connection URL from environment variables."""
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "chinook")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


_engine = None


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it if necessary."""
    global _engine
    if _engine is None:
        url = _build_database_url()
        _engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("SQLAlchemy engine created.")
    return _engine


# ---------------------------------------------------------------------------
# Read-only query execution
# ---------------------------------------------------------------------------

def execute_readonly_query(sql: str) -> list[dict[str, Any]]:
    """
    Execute a SQL SELECT query inside a read-only transaction.

    Args:
        sql: A SQL string that must start with SELECT.

    Returns:
        A list of row dictionaries (column → value).

    Raises:
        SQLExecutionError: If the query fails for any reason.
    """
    engine = get_engine()
    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            result = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in result]
            elapsed = time.perf_counter() - start
            logger.info(
                "Query executed in %.3fs, returned %d rows. SQL: %.200s",
                elapsed,
                len(rows),
                sql,
            )
            return rows
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error(
            "Query failed after %.3fs: %s | SQL: %.200s",
            elapsed,
            exc,
            sql,
        )
        raise SQLExecutionError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Schema metadata extraction
# ---------------------------------------------------------------------------

def get_schema_metadata() -> list[dict[str, Any]]:
    """
    Extract full schema metadata from the connected PostgreSQL database.

    Returns a list of table metadata dicts, each containing:
        - table_name (str)
        - columns (list of dicts: name, type, is_primary, is_nullable)
        - foreign_keys (list of dicts: column, referenced_table, referenced_column)
        - sample_rows (list of dicts: up to 3 rows)
    """
    engine = get_engine()

    column_sql = """
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            CASE WHEN pk.column_name IS NOT NULL THEN TRUE ELSE FALSE END AS is_primary
        FROM information_schema.columns c
        LEFT JOIN (
            SELECT ku.table_name, ku.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage ku
                ON tc.constraint_name = ku.constraint_name
                AND tc.table_schema = ku.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = 'public'
        ) pk ON c.table_name = pk.table_name AND c.column_name = pk.column_name
        WHERE c.table_schema = 'public'
        ORDER BY c.table_name, c.ordinal_position
    """

    fk_sql = """
        SELECT
            kcu.table_name,
            kcu.column_name,
            ccu.table_name  AS referenced_table,
            ccu.column_name AS referenced_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
            AND tc.table_schema = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
    """

    tables_sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """

    with engine.connect() as conn:
        # Fetch all tables
        table_names = [
            row[0]
            for row in conn.execute(text(tables_sql))
        ]

        # Exclude internal tables
        excluded = {"schema_embeddings", "query_history"}
        table_names = [t for t in table_names if t not in excluded]

        # Fetch column metadata
        col_rows = conn.execute(text(column_sql)).fetchall()
        col_map: dict[str, list[dict]] = {}
        for row in col_rows:
            tname = row[0]
            if tname in excluded:
                continue
            col_map.setdefault(tname, []).append({
                "name": row[1],
                "type": row[2],
                "is_nullable": row[3] == "YES",
                "is_primary": bool(row[4]),
            })

        # Fetch foreign key metadata
        fk_rows = conn.execute(text(fk_sql)).fetchall()
        fk_map: dict[str, list[dict]] = {}
        for row in fk_rows:
            tname = row[0]
            if tname in excluded:
                continue
            fk_map.setdefault(tname, []).append({
                "column": row[1],
                "referenced_table": row[2],
                "referenced_column": row[3],
            })

        # Build result — fetch sample rows per table
        metadata = []
        for tname in table_names:
            if tname not in col_map:
                continue
            try:
                sample_result = conn.execute(
                    text(f'SELECT * FROM "{tname}" LIMIT 3')
                )
                sample_rows = [dict(r._mapping) for r in sample_result]
            except Exception as exc:
                logger.warning("Could not fetch sample rows for %s: %s", tname, exc)
                sample_rows = []

            metadata.append({
                "table_name": tname,
                "columns": col_map.get(tname, []),
                "foreign_keys": fk_map.get(tname, []),
                "sample_rows": sample_rows,
            })

    logger.info("Schema metadata extracted for %d tables.", len(metadata))
    return metadata


# ---------------------------------------------------------------------------
# CSV → table loader
# ---------------------------------------------------------------------------

def create_table_from_dataframe(df: Any, table_name: str) -> int:
    """
    Create a PostgreSQL table from a pandas DataFrame, replacing it if it exists.

    Infers column types from pandas dtypes (int→BIGINT, float→DOUBLE PRECISION,
    bool→BOOLEAN, datetime→TIMESTAMP, everything else→TEXT), then uses plain
    SQLAlchemy Core execute so it works with SQLAlchemy 2.x without touching
    any DBAPI cursor directly.

    Args:
        df:         pandas DataFrame with already-sanitised column names.
        table_name: Sanitised table name (letters, digits, underscores only).

    Returns:
        Number of rows inserted.
    """
    import pandas as pd

    def _pg_type(dtype: Any) -> str:
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        if pd.api.types.is_float_dtype(dtype):
            return "DOUBLE PRECISION"
        if pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        return "TEXT"

    col_defs = ", ".join(
        f'"{col}" {_pg_type(dtype)}' for col, dtype in df.dtypes.items()
    )

    # Replace NaN/NaT with None so psycopg2 sends NULL
    df = df.where(df.notna(), other=None)
    records: list[dict] = df.to_dict(orient="records")

    col_names   = ", ".join(f'"{c}"' for c in df.columns)
    # SQLAlchemy :name bindparams — column names are already sanitised identifiers
    placeholders = ", ".join(f":{c}" for c in df.columns)
    insert_sql  = text(f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})')

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
        conn.execute(text(f'CREATE TABLE "{table_name}" ({col_defs})'))

        chunk = 500
        for i in range(0, len(records), chunk):
            conn.execute(insert_sql, records[i : i + chunk])

        conn.commit()

    logger.info("Created/replaced table '%s' with %d rows.", table_name, len(df))
    return len(df)
