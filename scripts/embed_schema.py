"""
embed_schema.py — One-time script to embed the database schema into pgvector.

Usage:
    python scripts/embed_schema.py

This script:
1. Connects to PostgreSQL and extracts schema metadata
2. Converts metadata to natural-language chunks
3. Generates embeddings via sentence-transformers
4. Stores embeddings in the schema_embeddings table

Run this after loading the Chinook data and whenever the schema changes.
"""

import logging
import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Extract schema, parse to chunks, embed, and store in pgvector."""
    logger.info("=== Schema Embedding Script ===")

    try:
        from app.db import get_schema_metadata
        from app.schema_parser import parse_schema_to_chunks
        from app.embeddings import embed_and_store_schema
    except ImportError as exc:
        logger.error("Import error — ensure you're running from the project root: %s", exc)
        sys.exit(1)

    # Step 1: Extract schema metadata
    logger.info("Extracting schema metadata from database...")
    try:
        metadata = get_schema_metadata()
    except Exception as exc:
        logger.error("Failed to get schema metadata: %s", exc)
        logger.error("Is the database running? Check your .env POSTGRES_* settings.")
        sys.exit(1)

    if not metadata:
        logger.warning("No tables found in the database. Did you run load_chinook.py first?")
        sys.exit(1)

    logger.info("Found %d tables.", len(metadata))

    # Step 2: Parse schema into text chunks
    logger.info("Parsing schema into natural-language chunks...")
    chunks = parse_schema_to_chunks(metadata)
    logger.info("Generated %d chunks.", len(chunks))

    for i, chunk in enumerate(chunks[:3]):
        logger.debug("Chunk %d preview: %s", i, chunk[:120])

    # Step 3: Embed and store
    logger.info("Generating embeddings and storing in pgvector...")
    try:
        embed_and_store_schema(chunks)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        sys.exit(1)

    logger.info("Schema embedding complete. %d chunks stored.", len(chunks))
    logger.info("The app is ready to handle queries.")


if __name__ == "__main__":
    main()
