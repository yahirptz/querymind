"""
embeddings.py — Schema embedding generation and cosine-similarity retrieval via pgvector.
"""

import logging
from typing import Any

import numpy as np
from pgvector.sqlalchemy import Vector
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

def embed_and_store_schema(chunks: list[str]) -> None:
    """
    Generate embeddings for schema chunks and upsert them into schema_embeddings.

    Existing rows with the same chunk_text are updated in place (upsert by text).

    Args:
        chunks: List of natural-language schema description strings.
    """
    if not chunks:
        logger.warning("embed_and_store_schema called with empty chunk list.")
        return

    _ensure_table_exists()
    model = _get_model()

    logger.info("Embedding %d schema chunks...", len(chunks))
    vectors: np.ndarray = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

    upsert_sql = text("""
        INSERT INTO schema_embeddings (chunk_text, embedding)
        VALUES (:chunk_text, :embedding)
        ON CONFLICT (chunk_text) DO UPDATE
            SET embedding = EXCLUDED.embedding
    """)

    engine = get_engine()
    with engine.connect() as conn:
        for chunk, vec in zip(chunks, vectors):
            conn.execute(upsert_sql, {
                "chunk_text": chunk,
                "embedding": vec.tolist(),
            })
        conn.commit()

    logger.info("Stored %d schema embeddings.", len(chunks))


def retrieve_relevant_schema(question: str, top_k: int = 5) -> list[str]:
    """
    Embed the question and return the top-k most semantically similar schema chunks.

    Args:
        question: The user's natural language question.
        top_k:    Number of chunks to return (default 5).

    Returns:
        List of schema chunk strings ordered by descending cosine similarity.
    """
    model = _get_model()
    question_vec: np.ndarray = model.encode(
        [question], show_progress_bar=False, normalize_embeddings=True
    )[0]

    # Cast to list for pgvector compatibility
    vec_list = question_vec.tolist()

    similarity_sql = text("""
        SELECT chunk_text
        FROM schema_embeddings
        ORDER BY embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
    """)

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(similarity_sql, {
            "query_vec": str(vec_list),
            "top_k": top_k,
        }).fetchall()

    chunks = [row[0] for row in rows]
    logger.info(
        "Retrieved %d schema chunks for question: '%.80s'", len(chunks), question
    )
    return chunks
