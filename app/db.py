"""
db.py — PostgreSQL connection, read-only query execution, and schema metadata extraction.
"""

import logging
import os
import threading
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
_external_engine = None
_active_connection_info: dict | None = None
_engine_lock = threading.Lock()


def get_engine():
    """Return the singleton local SQLAlchemy engine, creating it if necessary."""
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


def get_query_engine():
    """Return the active query engine (external if connected, otherwise local)."""
    with _engine_lock:
        if _external_engine is not None:
            return _external_engine
    return get_engine()


def set_active_connection(host: str, port: int, user: str, password: str, database: str) -> None:
    """
    Connect to an external PostgreSQL database and make it the active query engine.

    Tests the connection before committing. Thread-safe.

    Raises:
        Exception: If the connection test fails (e.g. wrong credentials, timeout).
    """
    global _external_engine, _active_connection_info

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    new_engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=3,
        max_overflow=5,
        pool_pre_ping=True,
        echo=False,
        connect_args={"connect_timeout": 5},
    )

    # Validate the connection before storing it
    with new_engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    with _engine_lock:
        if _external_engine is not None:
            _external_engine.dispose()
        _external_engine = new_engine
        _active_connection_info = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
        }

    logger.info("External DB connection set: %s@%s:%d/%s", user, host, port, database)


def reset_connection() -> None:
    """Dispose the external engine and return to the default local database."""
    global _external_engine, _active_connection_info
    with _engine_lock:
        if _external_engine is not None:
            _external_engine.dispose()
            _external_engine = None
        _active_connection_info = None
    logger.info("Connection reset to local database.")


def get_active_connection_info() -> dict | None:
    """Return the current external connection info (no password), or None if using local DB."""
    with _engine_lock:
        return dict(_active_connection_info) if _active_connection_info else None


# ---------------------------------------------------------------------------
# Read-only query execution
# ---------------------------------------------------------------------------

def execute_readonly_query(sql: str) -> list[dict[str, Any]]:
    """
    Execute a SQL SELECT query inside a read-only transaction.

    Uses the active query engine (external if connected, otherwise local).

    Args:
        sql: A SQL string that must start with SELECT.

    Returns:
        A list of row dictionaries (column → value).

    Raises:
        SQLExecutionError: If the query fails for any reason.
    """
    engine = get_query_engine()
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
    Extract full schema metadata from the active query database.

    Uses the active query engine (external if connected, otherwise local).

    Returns a list of table metadata dicts, each containing:
        - table_name (str)
        - columns (list of dicts: name, type, is_primary, is_nullable)
        - foreign_keys (list of dicts: column, referenced_table, referenced_column)
        - sample_rows (list of dicts: up to 3 rows)
    """
    engine = get_query_engine()

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


# ---------------------------------------------------------------------------
# SQL dump executor
# ---------------------------------------------------------------------------

# Tables that must never be touched by an external SQL dump.
_DUMP_PROTECTED: frozenset[str] = frozenset({
    "schema_embeddings", "query_history",
    "album", "artist", "customer", "employee", "genre",
    "invoice", "invoice_line", "media_type", "playlist",
    "playlist_track", "track",
})

# Lines to discard from SQL dumps before executing.
_DUMP_STRIP: list = None  # lazy-init below to avoid re import at module level


def _get_dump_strip():
    import re
    global _DUMP_STRIP
    if _DUMP_STRIP is None:
        _DUMP_STRIP = [
            re.compile(r"^\s*DROP\s+DATABASE\b", re.IGNORECASE),
            re.compile(r"^\s*CREATE\s+DATABASE\b", re.IGNORECASE),
            re.compile(r"^\s*\\connect\b", re.IGNORECASE),
            re.compile(r"^\s*\\c\s", re.IGNORECASE),
        ]
    return _DUMP_STRIP


def execute_sql_dump(sql_content: str, existing_tables: set[str]) -> dict:
    """
    Safely execute a PostgreSQL SQL dump string against the local database.

    Strips DROP/CREATE DATABASE and \\connect statements. Blocks any
    statement that targets a protected table. Handles both INSERT-style
    and COPY-FROM-stdin dumps generated by pg_dump.

    Args:
        sql_content:     Raw SQL string from the dump file.
        existing_tables: Set of table names that existed before the dump,
                         used to determine which tables were newly created.

    Returns:
        {tables_created: list[str], rows_inserted: {table: int}}

    Raises:
        ValueError:     If the dump targets a protected table.
        RuntimeError:   If SQL execution fails.
    """
    import io
    import re

    # ---- Strip forbidden top-level statements --------------------------------
    strip_patterns = _get_dump_strip()
    cleaned = "\n".join(
        line for line in sql_content.splitlines()
        if not any(p.match(line) for p in strip_patterns)
    )

    # ---- Guard against destructive ops on protected tables ------------------
    _destructive_re = re.compile(
        r"\b(DROP\s+TABLE(?:\s+IF\s+EXISTS)?|TRUNCATE(?:\s+TABLE)?"
        r"|DELETE\s+FROM|ALTER\s+TABLE)\b"
        r"\s+(?:(?:public|pg_catalog)\s*\.\s*)?[\"'`]?(\w+)[\"'`]?",
        re.IGNORECASE,
    )
    for m in _destructive_re.finditer(cleaned):
        tname = m.group(2).lower()
        if tname in _DUMP_PROTECTED:
            raise ValueError(
                f"SQL dump targets protected table '{tname}' "
                f"with '{m.group(1).split()[0].upper()}'."
            )

    _copy_target_re = re.compile(
        r"\bCOPY\s+(?:(?:public|pg_catalog)\s*\.\s*)?[\"'`]?(\w+)[\"'`]?",
        re.IGNORECASE,
    )
    for m in _copy_target_re.finditer(cleaned):
        tname = m.group(1).lower()
        if tname in _DUMP_PROTECTED:
            raise ValueError(
                f"SQL dump tries to COPY into protected table: '{tname}'."
            )

    # ---- Execute via raw psycopg2 with autocommit ---------------------------
    # COPY ... FROM stdin blocks need special handling via copy_expert();
    # everything else is passed as regular SQL.
    _copy_block_re = re.compile(
        r"(COPY\b[^;]+FROM\s+stdin[^;]*;)\n(.*?)\n\\\.(?=\n|$)",
        re.IGNORECASE | re.DOTALL,
    )

    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        raw_conn.autocommit = True
        cur = raw_conn.cursor()

        pos = 0
        for match in _copy_block_re.finditer(cleaned):
            # Execute SQL before this COPY block
            before = cleaned[pos : match.start()].strip()
            if before:
                cur.execute(before)

            # Execute COPY block via copy_expert
            copy_cmd = match.group(1).rstrip(";").strip()
            copy_cmd = re.sub(r"\bstdin\b", "STDIN", copy_cmd, flags=re.IGNORECASE)
            copy_data = match.group(2)
            cur.copy_expert(copy_cmd, io.StringIO(copy_data + "\n"))
            pos = match.end()

        tail = cleaned[pos:].strip()
        if tail:
            cur.execute(tail)

        cur.close()
    except Exception as exc:
        raise RuntimeError(f"SQL dump execution failed: {exc}") from exc
    finally:
        try:
            raw_conn.autocommit = False
        except Exception:
            pass
        raw_conn.close()

    # ---- Detect newly created tables ----------------------------------------
    with engine.connect() as conn:
        all_now = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            ).fetchall()
        }
    new_tables = sorted(all_now - existing_tables - {"schema_embeddings", "query_history"})

    rows_inserted: dict[str, int] = {}
    with engine.connect() as conn:
        for t in new_tables:
            try:
                rows_inserted[t] = conn.execute(
                    text(f'SELECT COUNT(*) FROM "{t}"')
                ).scalar() or 0
            except Exception:
                rows_inserted[t] = 0

    logger.info(
        "SQL dump executed: %d new table(s): %s", len(new_tables), new_tables
    )
    return {"tables_created": new_tables, "rows_inserted": rows_inserted}
