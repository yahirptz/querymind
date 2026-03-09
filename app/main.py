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

from app.db import SQLExecutionError, get_engine, get_schema_metadata
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
    "invoice", "invoice_line", "media_type", "playlist",
    "playlist_track", "track",
})
_INTERNAL_TABLES: frozenset[str] = frozenset({"schema_embeddings", "query_history"})
_PROTECTED_TABLES: frozenset[str] = _CHINOOK_TABLES | _INTERNAL_TABLES

_MAX_CSV_BYTES         = 10  * 1024 * 1024   # 10 MB  — single CSV
_MAX_ZIP_BYTES         = 100 * 1024 * 1024   # 100 MB — ZIP archive
_MAX_ZIP_UNCOMPRESSED  = 500 * 1024 * 1024   # 500 MB — bomb guard
_MAX_ZIP_FILES         = 20
_MAX_CSV_ROWS          = 50_000


def _sanitize_col_names(df) -> None:
    """Sanitize a DataFrame's column names in-place (SQL-safe, deduped)."""
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
    # CSV / ZIP upload
    # -----------------------------------------------------------------------

    @app.route("/api/upload-csv", methods=["POST"])
    def upload_csv():
        """
        Accept a CSV, XLSX, or ZIP file upload and load it into PostgreSQL.

        Form fields:
            file       — .csv (≤10 MB), .xlsx (≤10 MB), or .zip (≤100 MB)
            table_name — required for single CSV/XLSX; ignored for ZIP

        Returns (single file):
            200: {message, table_name, row_count}
        Returns (ZIP):
            200: {zip: true, tables: [{table_name, row_count}], skipped: [...]}
        """
        import zipfile
        import pandas as pd
        from app.db import create_table_from_dataframe
        from app.embeddings import embed_and_store_schema, verify_embeddings_exist

        if "file" not in request.files:
            return jsonify({"error": "No file provided."}), 400

        f = request.files["file"]
        fname = (f.filename or "").lower()

        _ALLOWED = (".csv", ".xlsx", ".zip")
        if not fname or not any(fname.endswith(ext) for ext in _ALLOWED):
            return jsonify({"error": "Only .csv, .xlsx, and .zip files are accepted."}), 400

        raw = f.read()
        is_zip = fname.endswith(".zip")
        max_bytes = _MAX_ZIP_BYTES if is_zip else _MAX_CSV_BYTES
        limit_label = "100 MB" if is_zip else "10 MB"
        if len(raw) > max_bytes:
            return jsonify({"error": f"File exceeds the {limit_label} size limit."}), 400

        # ------------------------------------------------------------------ #
        #  ZIP branch                                                          #
        # ------------------------------------------------------------------ #
        if is_zip:
            return _handle_zip_upload(raw, embed_and_store_schema)

        # ------------------------------------------------------------------ #
        #  Single CSV / XLSX branch                                            #
        # ------------------------------------------------------------------ #
        table_name_raw = request.form.get("table_name", "").strip()
        try:
            table_name = _sanitize_table_name(table_name_raw)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if table_name in _PROTECTED_TABLES:
            return jsonify({"error": f"'{table_name}' is a reserved table name."}), 400

        try:
            if fname.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(raw))
            else:
                df = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            return jsonify({"error": f"Could not parse file: {exc}"}), 400

        if df.empty or len(df.columns) == 0:
            return jsonify({"error": "File is empty or has no columns."}), 400

        if len(df) > _MAX_CSV_ROWS:
            return jsonify({
                "error": f"File has {len(df):,} rows; the maximum is {_MAX_CSV_ROWS:,}."
            }), 400

        _sanitize_col_names(df)

        try:
            row_count = create_table_from_dataframe(df, table_name)
        except Exception as exc:
            logger.error("Table creation failed: %s", exc)
            return jsonify({"error": f"Failed to load data into database: {exc}"}), 500

        embedding_warning: str | None = None
        try:
            all_meta = get_schema_metadata()
            if all_meta:
                chunks = parse_schema_to_chunks(all_meta)
                embed_and_store_schema(chunks)
                logger.info("Schema re-embedded after upload (%d tables).", len(all_meta))

            missing = verify_embeddings_exist([table_name])
            if missing:
                embedding_warning = (
                    f"Table '{table_name}' was created but its schema could not be "
                    "indexed in pgvector. Queries will still work via direct DB lookup."
                )
                logger.warning("Embedding verification failed for: %s", missing)
        except Exception as exc:
            embedding_warning = (
                f"Table '{table_name}' was created but schema indexing failed: {exc}."
            )
            logger.error("Schema embedding failed after upload: %s", exc)

        response: dict = {
            "message": f"Table '{table_name}' created with {row_count:,} rows.",
            "table_name": table_name,
            "row_count": row_count,
        }
        if embedding_warning:
            response["embedding_warning"] = embedding_warning
        return jsonify(response), 200

    # -----------------------------------------------------------------------
    # External database connection
    # -----------------------------------------------------------------------

    @app.route("/api/connect-db", methods=["POST"])
    def connect_db():
        """
        Connect to an external PostgreSQL database.

        Request body (JSON):
            {host, port, user, password, database}

        Returns:
            200: {success: true, table_count: N}
            400: {error} — invalid input or connection failure
        """
        from app.db import set_active_connection, get_engine
        from app.embeddings import embed_and_store_schema
        from sqlalchemy import text

        body = request.get_json(silent=True) or {}
        host     = str(body.get("host", "")).strip()
        port_raw = body.get("port", 5432)
        user     = str(body.get("user", "")).strip()
        password = str(body.get("password", ""))
        database = str(body.get("database", "")).strip()

        if not host:
            return jsonify({"error": "Host is required."}), 400
        if not user:
            return jsonify({"error": "Username is required."}), 400
        if not database:
            return jsonify({"error": "Database name is required."}), 400
        try:
            port = int(port_raw)
            if not (1 <= port <= 65535):
                raise ValueError()
        except (ValueError, TypeError):
            return jsonify({"error": "Port must be a number between 1 and 65535."}), 400

        try:
            set_active_connection(host, port, user, password, database)
        except Exception as exc:
            logger.warning("External DB connection failed: %s", exc)
            return jsonify({"error": f"Connection failed: {exc}"}), 400

        # Clear old embeddings and re-embed the new database's schema
        embedding_warning: str | None = None
        try:
            with get_engine().connect() as conn:
                try:
                    conn.execute(text("DELETE FROM schema_embeddings"))
                    conn.commit()
                except Exception:
                    conn.rollback()

            meta = get_schema_metadata()
            if meta:
                chunks = parse_schema_to_chunks(meta)
                embed_and_store_schema(chunks)
                logger.info("Schema re-embedded for external DB (%d tables).", len(meta))
        except Exception as exc:
            logger.error("Schema re-embedding after connect failed: %s", exc)
            embedding_warning = str(exc)
            try:
                meta = get_schema_metadata()
            except Exception:
                meta = []

        response: dict = {"success": True, "table_count": len(meta)}
        if embedding_warning:
            response["embedding_warning"] = embedding_warning
        return jsonify(response), 200

    @app.route("/api/connections")
    def get_connections():
        """
        Return info about the currently active external connection, or indicate local DB.

        Returns:
            200: {connected: true, host, port, database, user} or {connected: false}
        """
        from app.db import get_active_connection_info
        info = get_active_connection_info()
        if info:
            return jsonify({"connected": True, **info}), 200
        return jsonify({"connected": False}), 200

    @app.route("/api/connections/reset", methods=["POST"])
    def reset_connections():
        """
        Disconnect from the external database and return to the default local Chinook DB.

        Returns:
            200: {success: true, message}
        """
        from app.db import reset_connection, get_engine
        from app.embeddings import embed_and_store_schema
        from sqlalchemy import text

        reset_connection()

        # Re-embed local schema
        try:
            with get_engine().connect() as conn:
                try:
                    conn.execute(text("DELETE FROM schema_embeddings"))
                    conn.commit()
                except Exception:
                    conn.rollback()

            meta = get_schema_metadata()
            if meta:
                chunks = parse_schema_to_chunks(meta)
                embed_and_store_schema(chunks)
                logger.info("Schema re-embedded for local DB after reset (%d tables).", len(meta))
        except Exception as exc:
            logger.warning("Schema re-embedding after reset failed (non-fatal): %s", exc)

        return jsonify({
            "success": True,
            "message": "Disconnected. Back to local Chinook database.",
        }), 200

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
# ZIP upload handler
# ---------------------------------------------------------------------------

def _handle_zip_upload(raw: bytes, embed_fn) -> "flask.Response":
    """
    Process a ZIP file upload: extract CSVs, XLSXs, and/or a SQL dump.

    Safety:
      - Max 100 MB compressed (already checked by caller)
      - Max 500 MB uncompressed (zip-bomb guard)
      - Max 20 files
      - Path-traversal guard (.., leading /)
      - Only .csv / .xlsx / .sql processed; everything else skipped

    Returns a Flask JSON response directly.
    """
    import os
    import zipfile
    import pandas as pd
    from flask import jsonify
    from app.db import create_table_from_dataframe, execute_sql_dump
    from app.embeddings import embed_and_store_schema
    from sqlalchemy import text as sqla_text

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return jsonify({"error": "Uploaded file is not a valid ZIP archive."}), 400

    # Zip-bomb guard
    total_uncompressed = sum(info.file_size for info in zf.infolist())
    if total_uncompressed > _MAX_ZIP_UNCOMPRESSED:
        return jsonify({
            "error": f"ZIP contents exceed the {_MAX_ZIP_UNCOMPRESSED // (1024**2):,} MB uncompressed limit."
        }), 400

    members = [m for m in zf.infolist() if not m.is_dir()]
    if len(members) > _MAX_ZIP_FILES:
        return jsonify({
            "error": f"ZIP contains {len(members)} files; maximum is {_MAX_ZIP_FILES}."
        }), 400

    # Classify files
    csv_xlsx: list = []
    sql_files: list = []
    skipped: list[str] = []

    for info in members:
        name = info.filename
        # Path-traversal guard
        if ".." in name or name.startswith("/"):
            skipped.append(f"{name} (path traversal blocked)")
            continue
        basename = os.path.basename(name)
        if not basename:
            continue
        ext = os.path.splitext(basename)[1].lower()
        if ext in (".csv", ".xlsx"):
            csv_xlsx.append(info)
        elif ext == ".sql":
            sql_files.append(info)
        else:
            skipped.append(f"{name} (unsupported type)")

    if not csv_xlsx and not sql_files:
        return jsonify({
            "error": "ZIP contains no usable files (.csv, .xlsx, or .sql).",
            "skipped": skipped,
        }), 400

    if sql_files and csv_xlsx:
        return jsonify({
            "error": "ZIP contains both .sql and .csv/.xlsx files. Include only one type per ZIP."
        }), 400

    if len(sql_files) > 1:
        return jsonify({
            "error": f"ZIP contains {len(sql_files)} .sql files; only one SQL dump per ZIP is supported."
        }), 400

    results: list[dict] = []

    # ------------------------------------------------------------------ #
    # SQL dump path                                                        #
    # ------------------------------------------------------------------ #
    if sql_files:
        try:
            sql_content = zf.read(sql_files[0].filename).decode("utf-8", errors="replace")
        except Exception as exc:
            return jsonify({"error": f"Could not read SQL file: {exc}"}), 400

        # Capture table state before execution
        try:
            engine = get_engine()
            with engine.connect() as conn:
                existing = {
                    r[0]
                    for r in conn.execute(sqla_text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                    )).fetchall()
                }
        except Exception as exc:
            return jsonify({"error": f"Could not query database: {exc}"}), 500

        try:
            dump_result = execute_sql_dump(sql_content, existing)
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

        for tname in dump_result["tables_created"]:
            results.append({
                "table_name": tname,
                "row_count": dump_result["rows_inserted"].get(tname, 0),
            })

    # ------------------------------------------------------------------ #
    # CSV / XLSX path                                                      #
    # ------------------------------------------------------------------ #
    else:
        for info in csv_xlsx:
            basename = os.path.basename(info.filename)
            ext = os.path.splitext(basename)[1].lower()
            stem = os.path.splitext(basename)[0]

            try:
                table_name = _sanitize_table_name(stem)
            except ValueError as exc:
                skipped.append(f"{info.filename} (bad table name: {exc})")
                continue

            if table_name in _PROTECTED_TABLES:
                skipped.append(f"{info.filename} (reserved name: {table_name})")
                continue

            file_bytes = zf.read(info.filename)
            try:
                if ext == ".xlsx":
                    df = pd.read_excel(io.BytesIO(file_bytes))
                else:
                    df = pd.read_csv(io.BytesIO(file_bytes))
            except Exception as exc:
                skipped.append(f"{info.filename} (parse error: {exc})")
                continue

            if df.empty or len(df.columns) == 0:
                skipped.append(f"{info.filename} (empty)")
                continue

            if len(df) > _MAX_CSV_ROWS:
                skipped.append(f"{info.filename} (exceeds {_MAX_CSV_ROWS:,} row limit)")
                continue

            _sanitize_col_names(df)

            try:
                row_count = create_table_from_dataframe(df, table_name)
                results.append({"table_name": table_name, "row_count": row_count})
                logger.info("ZIP: created table '%s' (%d rows).", table_name, row_count)
            except Exception as exc:
                skipped.append(f"{info.filename} (DB error: {exc})")

    if not results:
        return jsonify({
            "error": "No tables were created. All files failed or were skipped.",
            "skipped": skipped,
        }), 400

    # Re-embed schema once for all new tables
    embedding_warning: str | None = None
    try:
        all_meta = get_schema_metadata()
        if all_meta:
            chunks = parse_schema_to_chunks(all_meta)
            embed_fn(chunks)
            logger.info(
                "Schema re-embedded after ZIP upload (%d tables total, %d new).",
                len(all_meta), len(results),
            )
    except Exception as exc:
        embedding_warning = f"Tables created but schema indexing failed: {exc}"
        logger.error("Schema embedding failed after ZIP upload: %s", exc)

    response: dict = {
        "zip": True,
        "tables": results,
        "table_count": len(results),
    }
    if skipped:
        response["skipped"] = skipped
    if embedding_warning:
        response["embedding_warning"] = embedding_warning
    return jsonify(response), 200


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
