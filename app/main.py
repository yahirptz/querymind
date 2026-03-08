"""
main.py — Flask application factory, routes, and request middleware.
"""

import io
import logging
import re
import time
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from app.db import SQLExecutionError, get_schema_metadata
from app.models import QueryHistory, init_db
from app.rag_pipeline import handle_question
from app.schema_parser import parse_schema_to_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV upload constants & helpers
# ---------------------------------------------------------------------------

# Chinook sample-data tables that must never be overwritten or deleted.
_CHINOOK_TABLES: frozenset[str] = frozenset({
    "album", "artist", "customer", "employee", "genre",
    "invoice", "invoiceline", "mediatype", "playlist",
    "playlisttrack", "track",
})
_INTERNAL_TABLES: frozenset[str] = frozenset({"schema_embeddings", "query_history"})
_PROTECTED_TABLES: frozenset[str] = _CHINOOK_TABLES | _INTERNAL_TABLES

_MAX_CSV_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_CSV_ROWS  = 50_000


def _sanitize_table_name(name: str) -> str:
    """Return a lowercase, SQL-safe identifier or raise ValueError."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError("Table name cannot be empty.")
    if not name[0].isalpha():
        raise ValueError("Table name must start with a letter.")
    if len(name) > 63:
        raise ValueError("Table name must be 63 characters or fewer.")
    return name


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """
    Flask application factory.

    Returns:
        A configured Flask app instance.
    """
    app = Flask(__name__, static_folder="../static", static_url_path="/static")
    CORS(app)

    # -----------------------------------------------------------------------
    # Startup: ensure DB tables exist and schema is embedded
    # -----------------------------------------------------------------------
    with app.app_context():
        try:
            init_db()
            logger.info("Database tables initialized on startup.")
        except Exception as exc:
            logger.error("DB init failed: %s", exc)

        try:
            _bootstrap_embeddings()
        except Exception as exc:
            logger.warning("Schema embedding bootstrap failed (non-fatal): %s", exc)

    # -----------------------------------------------------------------------
    # Request logging middleware
    # -----------------------------------------------------------------------
    @app.before_request
    def _log_request() -> None:
        request._start_time = time.perf_counter()  # type: ignore[attr-defined]
        logger.info("→ %s %s", request.method, request.path)

    @app.after_request
    def _log_response(response):
        elapsed = time.perf_counter() - getattr(request, "_start_time", time.perf_counter())
        logger.info(
            "← %s %s %d (%.3fs)",
            request.method,
            request.path,
            response.status_code,
            elapsed,
        )
        return response

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.route("/")
    def index():
        """Serve the chat UI."""
        return send_from_directory("../static", "index.html")

    @app.route("/health")
    def health():
        """
        Health check endpoint.

        Returns:
            JSON with application and database status.
        """
        from sqlalchemy import text
        from app.db import get_engine
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception as exc:
            logger.error("Health check DB ping failed: %s", exc)
            db_status = "unavailable"

        status = "ok" if db_status == "connected" else "degraded"
        return jsonify({"status": status, "db": db_status}), 200

    @app.route("/api/query", methods=["POST"])
    def query():
        """
        Accept a natural language question and return SQL + results + summary.

        Request body (JSON):
            {"question": "What are the top 5 selling artists?"}

        Returns:
            200: {question, sql, results, summary, result_count, timestamp}
            400: {error} — bad input
            422: {error} — SQL safety or CANNOT_ANSWER
            500: {error} — internal error
        """
        body = request.get_json(silent=True)
        if not body or "question" not in body:
            return jsonify({"error": "Request must include a 'question' field."}), 400

        question: str = str(body["question"]).strip()
        tables_filter = body.get("tables") or None  # list[str] | None

        try:
            result = handle_question(question, tables_filter=tables_filter)
            return jsonify(result), 200

        except ValueError as exc:
            logger.warning("Validation error: %s", exc)
            return jsonify({"error": str(exc)}), 422

        except SQLExecutionError as exc:
            logger.error("SQL execution error: %s", exc)
            return jsonify({"error": "The query could not be executed against the database."}), 422

        except RuntimeError as exc:
            logger.error("Pipeline runtime error: %s", exc)
            return jsonify({"error": str(exc)}), 500

        except Exception as exc:
            logger.exception("Unexpected error handling question.")
            return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

    @app.route("/api/history")
    def history():
        """
        Return the last 20 queries from query_history.

        Returns:
            200: list of query history records as JSON.
        """
        from sqlalchemy.orm import Session
        from app.db import get_engine

        try:
            engine = get_engine()
            with Session(engine) as session:
                records = (
                    session.query(QueryHistory)
                    .order_by(QueryHistory.created_at.desc())
                    .limit(20)
                    .all()
                )
                return jsonify([r.to_dict() for r in records]), 200
        except Exception as exc:
            logger.error("Failed to fetch history: %s", exc)
            return jsonify({"error": "Could not retrieve query history."}), 500

    @app.route("/api/schema")
    def schema():
        """
        Return the parsed schema metadata for the connected database.

        Returns:
            200: list of table metadata dicts.
        """
        try:
            raw = get_schema_metadata()
            return jsonify(raw), 200
        except Exception as exc:
            logger.error("Failed to fetch schema: %s", exc)
            return jsonify({"error": "Could not retrieve schema metadata."}), 500

    # -----------------------------------------------------------------------
    # CSV upload
    # -----------------------------------------------------------------------

    @app.route("/api/upload-csv", methods=["POST"])
    def upload_csv():
        """
        Accept a CSV file upload and load it into a new PostgreSQL table.

        Form fields:
            file       — multipart CSV file (.csv only, ≤ 10 MB)
            table_name — desired table name (letters/numbers/underscores)

        Returns:
            200: {message, table_name, row_count}
            400: {error} — bad input / validation failure
            500: {error} — DB or embedding error
        """
        if "file" not in request.files:
            return jsonify({"error": "No file provided."}), 400

        f = request.files["file"]
        if not f.filename or not f.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only .csv files are accepted."}), 400

        raw = f.read()
        if len(raw) > _MAX_CSV_BYTES:
            return jsonify({"error": "File exceeds the 10 MB size limit."}), 400

        table_name_raw = request.form.get("table_name", "").strip()
        try:
            table_name = _sanitize_table_name(table_name_raw)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if table_name in _PROTECTED_TABLES:
            return jsonify({"error": f"'{table_name}' is a reserved table name."}), 400

        import pandas as pd
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

        if df.empty or len(df.columns) == 0:
            return jsonify({"error": "CSV file is empty or has no columns."}), 400

        if len(df) > _MAX_CSV_ROWS:
            return jsonify({
                "error": f"CSV has {len(df):,} rows; the maximum is {_MAX_CSV_ROWS:,}."
            }), 400

        # Sanitize column names, dedup with numeric suffix
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        for c in df.columns:
            s = re.sub(r"[^a-z0-9_]", "_", str(c).strip().lower()) or "col"
            if not s[0].isalpha():
                s = "col_" + s
            count = seen.get(s, 0)
            seen[s] = count + 1
            new_cols.append(f"{s}_{count}" if count else s)
        df.columns = new_cols  # type: ignore[assignment]

        try:
            from app.db import create_table_from_dataframe
            row_count = create_table_from_dataframe(df, table_name)
        except Exception as exc:
            logger.error("CSV table creation failed: %s", exc)
            return jsonify({"error": f"Failed to load data into database: {exc}"}), 500

        # Embed the full schema into pgvector so semantic search stays current.
        # Errors are surfaced — never swallowed silently.
        embedding_warning: str | None = None
        try:
            from app.embeddings import embed_and_store_schema, verify_embeddings_exist
            all_meta = get_schema_metadata()
            if all_meta:
                chunks = parse_schema_to_chunks(all_meta)
                embed_and_store_schema(chunks)
                logger.info("Schema re-embedded after CSV upload (%d tables).", len(all_meta))

            # Verify the new table's embedding was actually stored
            missing = verify_embeddings_exist([table_name])
            if missing:
                embedding_warning = (
                    f"Table '{table_name}' was created but its schema could not be "
                    "indexed in pgvector. Queries will still work via direct DB lookup, "
                    "but semantic search accuracy may be reduced. Try re-uploading if issues persist."
                )
                logger.warning("Embedding verification failed for: %s", missing)
            else:
                logger.info("Embedding verified for table '%s'.", table_name)
        except Exception as exc:
            embedding_warning = (
                f"Table '{table_name}' was created successfully, but schema indexing "
                f"failed: {exc}. Queries will still work via direct DB lookup."
            )
            logger.error("Schema embedding failed after CSV upload: %s", exc)

        response: dict = {
            "message": f"Table '{table_name}' created with {row_count:,} rows.",
            "table_name": table_name,
            "row_count": row_count,
        }
        if embedding_warning:
            response["embedding_warning"] = embedding_warning

        return jsonify(response), 200

    # -----------------------------------------------------------------------
    # List user-uploaded tables
    # -----------------------------------------------------------------------

    @app.route("/api/tables")
    def user_tables():
        """
        Return all user-uploaded tables (non-Chinook, non-internal) with row counts.

        Returns:
            200: list of {table_name, row_count}
        """
        from sqlalchemy import text
        from app.db import get_engine

        try:
            engine = get_engine()
            with engine.connect() as conn:
                all_tables = [
                    row[0]
                    for row in conn.execute(text("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)).fetchall()
                ]

                result = []
                for tname in all_tables:
                    if tname in _PROTECTED_TABLES:
                        continue
                    try:
                        count = conn.execute(
                            text(f'SELECT COUNT(*) FROM "{tname}"')
                        ).scalar()
                    except Exception:
                        count = None
                    result.append({"table_name": tname, "row_count": count})

            return jsonify(result), 200
        except Exception as exc:
            logger.error("Failed to list user tables: %s", exc)
            return jsonify({"error": "Could not retrieve tables."}), 500

    # -----------------------------------------------------------------------
    # Delete a user-uploaded table
    # -----------------------------------------------------------------------

    @app.route("/api/tables/<table_name>", methods=["DELETE"])
    def delete_table(table_name: str):
        """
        Drop a user-uploaded table and remove its schema embeddings.

        Args (URL):
            table_name — name of the table to drop

        Returns:
            200: {message}
            400: invalid name
            403: protected table
            404: table not found
            500: DB error
        """
        from sqlalchemy import text
        from app.db import get_engine

        try:
            safe_name = _sanitize_table_name(table_name)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if safe_name in _PROTECTED_TABLES:
            return jsonify({
                "error": f"'{safe_name}' is a protected table and cannot be deleted."
            }), 403

        try:
            engine = get_engine()
            with engine.connect() as conn:
                exists = conn.execute(text("""
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = :name
                """), {"name": safe_name}).scalar()

                if not exists:
                    return jsonify({"error": f"Table '{safe_name}' not found."}), 404

                conn.execute(text(f'DROP TABLE IF EXISTS "{safe_name}"'))
                conn.execute(
                    text("DELETE FROM schema_embeddings WHERE chunk_text LIKE :prefix"),
                    {"prefix": f"Table: {safe_name} |%"},
                )
                conn.commit()

            return jsonify({"message": f"Table '{safe_name}' deleted."}), 200
        except Exception as exc:
            logger.error("Failed to delete table '%s': %s", safe_name, exc)
            return jsonify({"error": f"Failed to delete table: {exc}"}), 500

    return app


# ---------------------------------------------------------------------------
# Schema bootstrap helper
# ---------------------------------------------------------------------------

def _bootstrap_embeddings() -> None:
    """
    On first startup, embed the database schema into pgvector if not already done.

    Checks whether schema_embeddings has any rows; skips if already populated.
    """
    from sqlalchemy import text
    from app.db import get_engine
    from app.embeddings import embed_and_store_schema

    engine = get_engine()
    with engine.connect() as conn:
        try:
            row = conn.execute(
                text("SELECT COUNT(*) FROM schema_embeddings")
            ).scalar()
            if row and row > 0:
                logger.info(
                    "Schema embeddings already exist (%d rows). Skipping bootstrap.", row
                )
                return
        except Exception:
            # Table may not exist yet — proceed with embedding
            pass

    logger.info("No schema embeddings found. Running initial embedding...")
    metadata = get_schema_metadata()
    chunks = parse_schema_to_chunks(metadata)
    embed_and_store_schema(chunks)
    logger.info("Schema embedding bootstrap complete.")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
