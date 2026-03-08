"""
test_db.py — Unit and integration tests for the database layer.

Run with:
    pytest tests/test_db.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, call
from app.db import SQLExecutionError, execute_readonly_query, get_schema_metadata


# ---------------------------------------------------------------------------
# execute_readonly_query tests
# ---------------------------------------------------------------------------

class TestExecuteReadonlyQuery:
    """Tests for execute_readonly_query()."""

    def test_returns_list_of_dicts(self):
        """Valid SELECT returns rows as list of dicts."""
        mock_row = MagicMock()
        mock_row._mapping = {"id": 1, "name": "Alice"}

        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([mock_row]))

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("app.db.get_engine", return_value=mock_engine):
            result = execute_readonly_query("SELECT id, name FROM customers")

        assert isinstance(result, list)
        assert result[0] == {"id": 1, "name": "Alice"}

    def test_raises_sql_execution_error_on_db_failure(self):
        """Database errors are wrapped in SQLExecutionError."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = Exception("connection refused")

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("app.db.get_engine", return_value=mock_engine):
            with pytest.raises(SQLExecutionError, match="connection refused"):
                execute_readonly_query("SELECT * FROM nonexistent")

    def test_sets_read_only_transaction(self):
        """The function always sets transaction to READ ONLY."""
        from sqlalchemy import text

        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("app.db.get_engine", return_value=mock_engine):
            execute_readonly_query("SELECT 1")

        # First call should be the READ ONLY set
        first_call_arg = str(mock_conn.execute.call_args_list[0][0][0])
        assert "READ ONLY" in first_call_arg


# ---------------------------------------------------------------------------
# SQL validation tests (via rag_pipeline)
# ---------------------------------------------------------------------------

class TestSqlValidation:
    """Tests for the SQL safety validator in rag_pipeline."""

    def _validate(self, sql: str) -> None:
        from app.rag_pipeline import _validate_sql
        _validate_sql(sql)

    def test_valid_select(self):
        """Plain SELECT passes validation."""
        self._validate("SELECT id, name FROM customers LIMIT 10")

    def test_select_case_insensitive(self):
        """SELECT keyword matching is case-insensitive."""
        self._validate("select * from artists")

    def test_rejects_drop(self):
        from app.rag_pipeline import _validate_sql
        with pytest.raises(ValueError, match="DROP"):
            _validate_sql("DROP TABLE customers")

    def test_rejects_delete(self):
        from app.rag_pipeline import _validate_sql
        with pytest.raises(ValueError, match="DELETE"):
            _validate_sql("DELETE FROM customers WHERE id = 1")

    def test_rejects_insert(self):
        from app.rag_pipeline import _validate_sql
        with pytest.raises(ValueError, match="INSERT"):
            _validate_sql("INSERT INTO customers VALUES (1, 'x')")

    def test_rejects_update(self):
        from app.rag_pipeline import _validate_sql
        with pytest.raises(ValueError, match="UPDATE"):
            _validate_sql("UPDATE customers SET name='x'")

    def test_rejects_semicolons(self):
        from app.rag_pipeline import _validate_sql
        with pytest.raises(ValueError, match="semicolon"):
            _validate_sql("SELECT 1; DROP TABLE users")

    def test_rejects_comments(self):
        from app.rag_pipeline import _validate_sql
        with pytest.raises(ValueError, match="comment"):
            _validate_sql("SELECT 1 -- malicious comment")

    def test_rejects_non_select(self):
        from app.rag_pipeline import _validate_sql
        with pytest.raises(ValueError, match="SELECT"):
            _validate_sql("TRUNCATE customers")

    def test_rejects_too_long_query(self):
        from app.rag_pipeline import _validate_sql
        long_sql = "SELECT " + "a, " * 1000 + "b FROM t"
        with pytest.raises(ValueError, match="character limit"):
            _validate_sql(long_sql)


# ---------------------------------------------------------------------------
# Question validation tests
# ---------------------------------------------------------------------------

class TestQuestionValidation:
    """Tests for _validate_question() in rag_pipeline."""

    def test_empty_question_raises(self):
        from app.rag_pipeline import _validate_question
        with pytest.raises(ValueError, match="empty"):
            _validate_question("")

    def test_whitespace_only_raises(self):
        from app.rag_pipeline import _validate_question
        with pytest.raises(ValueError, match="empty"):
            _validate_question("   ")

    def test_too_long_raises(self):
        from app.rag_pipeline import _validate_question
        with pytest.raises(ValueError, match="character limit"):
            _validate_question("a" * 501)

    def test_valid_question_passes(self):
        from app.rag_pipeline import _validate_question
        _validate_question("What are the top selling artists?")
