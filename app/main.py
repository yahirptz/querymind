"""
main.py — Flask application factory, routes, and request middleware.
"""

import logging
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

        try:
            result = handle_question(question)
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
