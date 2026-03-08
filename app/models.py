"""
models.py — SQLAlchemy ORM model for query history persistence.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.db import get_engine

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class QueryHistory(Base):
    """Stores every question asked, the generated SQL, and metadata."""

    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    sql: Mapped[str] = mapped_column(Text, nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def to_dict(self) -> dict:
        """Serialize the record to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "sql": self.sql,
            "result_count": self.result_count,
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db() -> None:
    """Create all ORM-managed tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables initialized.")
