"""
load_chinook.py — Downloads and seeds the Chinook sample database into PostgreSQL.

Usage:
    python scripts/load_chinook.py

Requires a .env file (or environment variables) with POSTGRES_* credentials.
"""

import logging
import os
import sys
import time

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Chinook PostgreSQL SQL script (official GitHub source)
CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_PostgreSql.sql"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_connection():
    """Return a psycopg2 connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dbname=os.getenv("POSTGRES_DB", "chinook"),
    )


def _download_chinook_sql() -> str:
    """
    Download the Chinook PostgreSQL SQL script from GitHub.

    Returns:
        The raw SQL content as a string.

    Raises:
        SystemExit: If the download fails.
    """
    logger.info("Downloading Chinook SQL from GitHub...")
    try:
        response = requests.get(CHINOOK_URL, timeout=30)
        response.raise_for_status()
        logger.info("Download complete (%d bytes).", len(response.content))
        return response.text
    except requests.RequestException as exc:
        logger.error("Failed to download Chinook SQL: %s", exc)
        sys.exit(1)


def _wait_for_db(retries: int = 10, delay: float = 3.0) -> None:
    """
    Retry connecting to PostgreSQL until it's ready or retries are exhausted.

    Args:
        retries: Maximum number of connection attempts.
        delay:   Seconds to wait between attempts.
    """
    for attempt in range(1, retries + 1):
        try:
            conn = _get_connection()
            conn.close()
            logger.info("Database is ready.")
            return
        except psycopg2.OperationalError as exc:
            logger.warning(
                "DB not ready (attempt %d/%d): %s. Retrying in %.0fs...",
                attempt, retries, exc, delay,
            )
            time.sleep(delay)
    logger.error("Could not connect to the database after %d attempts.", retries)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download and execute the Chinook SQL script to seed the database."""
    logger.info("=== Chinook Database Loader ===")

    _wait_for_db()
    sql_content = _download_chinook_sql()

    try:
        conn = _get_connection()
        conn.autocommit = True
        cursor = conn.cursor()

        logger.info("Executing Chinook SQL script...")
        # Strip psql meta-commands and DB-level DDL that can't run via psycopg2:
        # - \c, \set, etc. (psql-only commands)
        # - DROP/CREATE DATABASE (we're already in the target DB)
        skip_prefixes = ("\\", "drop database", "create database")
        cleaned_lines = [
            line for line in sql_content.splitlines()
            if not line.strip().lower().startswith(skip_prefixes)
        ]
        cleaned_sql = "\n".join(cleaned_lines)
        cursor.execute(cleaned_sql)

        # Verify tables were created
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        logger.info("Tables created/updated:")
        for t in tables:
            cursor.execute(f'SELECT COUNT(*) FROM "{t}"')
            count = cursor.fetchone()[0]
            logger.info("  ✓ %-30s %d rows", t, count)

        cursor.close()
        conn.close()
        logger.info("Chinook database loaded successfully.")

    except psycopg2.Error as exc:
        logger.error("Database error during seed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
