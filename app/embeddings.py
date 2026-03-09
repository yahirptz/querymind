"""
embeddings.py — Schema embedding generation and cosine-similarity retrieval via pgvector.
"""

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from app.db import get_engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model (loaded once at import time)
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the singleton sentence-transformer model."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model '%s'...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Model loaded.")
    return _model


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ---------------------------------------------------------------------------
# Table bootstrap
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS schema_embeddings (
    id          SERIAL PRIMARY KEY,
    chunk_text  TEXT NOT NULL UNIQUE,
    embedding   vector({EMBEDDING_DIM}) NOT NULL
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS schema_embeddings_embedding_idx
ON schema_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10)
"""


def _ensure_table_exists() -> None:
    """Create the schema_embeddings table and index if they don't exist."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(_CREATE_TABLE_SQL))
        # IVFFlat index requires at least 1 row; create it lazily
        try:
            conn.execute(text(_CREATE_INDEX_SQL))
        except Exception:
            pass  # Index creation may fail if table is empty — that's OK
        conn.commit()
    logger.debug("schema_embeddings table ensured.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clear_all_embeddings() -> None:
    """
    Delete all rows from schema_embeddings, removing stale schema context.

    Called before re-embedding when switching databases so old chunks
    don't pollute retrieval results.

    Raises:
        RuntimeError: If the DELETE fails for any reason.
    """
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM schema_embeddings"))
            conn.commit()
        logger.info("Cleared all schema embeddings.")
    except Exception as exc:
        raise RuntimeError(f"Failed to clear schema embeddings: {exc}") from exc


def embed_and_store_schema(chunks: list[str]) -> None:
    """
    Generate embeddings for schema chunks and upsert them into schema_embeddings.

    Raises:
        RuntimeError: If embedding or storage fails — never swallowed silently.
    """
    if not chunks:
        logger.warning("embed_and_store_schema called with empty chunk list.")
        return

    _ensure_table_exists()
    model = _get_model()

    logger.info("Embedding %d schema chunks...", len(chunks))
    try:
        vectors: np.ndarray = model.encode(
            chunks, show_progress_bar=False, normalize_embeddings=True
        )
    except Exception as exc:
        raise RuntimeError(f"Sentence-transformer encoding failed: {exc}") from exc

    upsert_sql = text("""
        INSERT INTO schema_embeddings (chunk_text, embedding)
        VALUES (:chunk_text, :embedding)
        ON CONFLICT (chunk_text) DO UPDATE
            SET embedding = EXCLUDED.embedding
    """)

    engine = get_engine()
    try:
        with engine.connect() as conn:
            for chunk, vec in zip(chunks, vectors):
                conn.execute(upsert_sql, {
                    "chunk_text": chunk,
                    "embedding": vec.tolist(),
                })
            conn.commit()
    except Exception as exc:
        raise RuntimeError(f"Failed to store embeddings in pgvector: {exc}") from exc

    logger.info("Stored %d schema embeddings.", len(chunks))


def verify_embeddings_exist(table_names: list[str]) -> list[str]:
    """
    Check which tables from table_names have NO embedding stored in pgvector.

    Returns:
        List of table names that are missing from schema_embeddings.
    """
    if not table_names:
        return []

    engine = get_engine()
    try:
        with engine.connect() as conn:
            # Each schema chunk starts with "Table: <name> |"
            rows = conn.execute(
                text("SELECT chunk_text FROM schema_embeddings")
            ).fetchall()
    except Exception as exc:
        logger.warning("Could not query schema_embeddings for verification: %s", exc)
        # If the table doesn't exist yet treat everything as missing
        return list(table_names)

    embedded_tables: set[str] = set()
    for (chunk_text,) in rows:
        if chunk_text.startswith("Table: "):
            tname = chunk_text.split("|")[0].replace("Table: ", "").strip()
            embedded_tables.add(tname)

    missing = [t for t in table_names if t not in embedded_tables]
    if missing:
        logger.warning("Tables missing from pgvector: %s", missing)
    return missing


def embed_table(table_name: str) -> None:
    """
    Embed a single table's schema chunk on demand.

    Fetches the table's metadata from the database, builds the chunk text,
    and upserts it into schema_embeddings.

    Raises:
        RuntimeError: If the table is not found or embedding fails.
    """
    from app.db import get_schema_metadata
    from app.schema_parser import parse_schema_to_chunks

    all_meta = get_schema_metadata()
    meta = [t for t in all_meta if t["table_name"] == table_name]
    if not meta:
        raise RuntimeError(
            f"embed_table: table '{table_name}' not found in database schema."
        )

    chunks = parse_schema_to_chunks(meta)
    embed_and_store_schema(chunks)
    logger.info("On-demand embedding complete for table '%s'.", table_name)


def retrieve_relevant_schema(question: str, top_k: int = 5) -> list[str]:
    """
    Embed the question and return the top-k most semantically similar schema chunks.

    Returns:
        List of chunk strings ordered by descending cosine similarity.
        Returns an empty list if schema_embeddings is empty or unreachable.
    """
    model = _get_model()
    question_vec: np.ndarray = model.encode(
        [question], show_progress_bar=False, normalize_embeddings=True
    )[0]

    vec_list = question_vec.tolist()

    similarity_sql = text("""
        SELECT chunk_text
        FROM schema_embeddings
        ORDER BY embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
    """)

    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(similarity_sql, {
                "query_vec": str(vec_list),
                "top_k": top_k,
            }).fetchall()
    except Exception as exc:
        logger.warning("pgvector retrieval failed: %s", exc)
        return []

    chunks = [row[0] for row in rows]
    logger.info(
        "Retrieved %d schema chunks for question: '%.80s'", len(chunks), question
    )
    return chunks
