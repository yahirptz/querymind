"""
test_rag_pipeline.py — Unit tests for the RAG pipeline orchestration.

Run with:
    pytest tests/test_rag_pipeline.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# handle_question tests
# ---------------------------------------------------------------------------

class TestHandleQuestion:
    """End-to-end pipeline tests with all external dependencies mocked."""

    def _run(self, question: str, **overrides):
        """Helper: patch all external calls and run handle_question."""
        defaults = {
            "retrieve_relevant_schema": ["Table: artists | Columns: ArtistId (int, PK), Name (varchar)"],
            "generate_sql": "SELECT ArtistId, Name FROM artists LIMIT 5",
            "execute_readonly_query": [{"ArtistId": 1, "Name": "AC/DC"}],
            "summarize_results": "The top artist is AC/DC.",
            "_save_to_history": None,
        }
        defaults.update(overrides)

        patches = {
            "app.rag_pipeline.retrieve_relevant_schema": MagicMock(
                return_value=defaults["retrieve_relevant_schema"]
            ),
            "app.rag_pipeline.generate_sql": MagicMock(
                return_value=defaults["generate_sql"]
            ),
            "app.rag_pipeline.execute_readonly_query": MagicMock(
                return_value=defaults["execute_readonly_query"]
            ),
            "app.rag_pipeline.summarize_results": MagicMock(
                return_value=defaults["summarize_results"]
            ),
            "app.rag_pipeline._save_to_history": MagicMock(),
        }

        with patch.multiple("app.rag_pipeline", **{k.split(".")[-1]: v for k, v in patches.items()}):
            from app.rag_pipeline import handle_question
            return handle_question(question)

    def test_successful_pipeline_returns_expected_keys(self):
        """A successful run returns all required response fields."""
        result = self._run("Who are the top artists?")
        assert "question" in result
        assert "sql" in result
        assert "results" in result
        assert "summary" in result
        assert "result_count" in result
        assert "timestamp" in result

    def test_question_is_echoed_in_response(self):
        """The original question is returned unchanged."""
        q = "What is the total revenue?"
        result = self._run(q)
        assert result["question"] == q

    def test_result_count_matches_rows(self):
        """result_count equals len(results)."""
        rows = [{"id": i} for i in range(7)]
        result = self._run("anything", execute_readonly_query=rows)
        assert result["result_count"] == 7

    def test_cannot_answer_returns_error_response(self):
        """CANNOT_ANSWER from Claude returns a friendly error dict, not an exception."""
        with patch("app.rag_pipeline.retrieve_relevant_schema", return_value=["chunk"]), \
             patch("app.rag_pipeline.generate_sql", return_value="CANNOT_ANSWER"):
            from app.rag_pipeline import handle_question
            result = handle_question("What is the meaning of life?")

        assert result["sql"] is None
        assert result["results"] == []
        assert result["error"] == "CANNOT_ANSWER"
        assert "unable" in result["summary"].lower()

    def test_empty_question_raises_value_error(self):
        """An empty question raises ValueError before any API calls."""
        with pytest.raises(ValueError, match="empty"):
            with patch("app.rag_pipeline.retrieve_relevant_schema") as mock_schema:
                from app.rag_pipeline import handle_question
                handle_question("")
                mock_schema.assert_not_called()

    def test_oversized_question_raises_value_error(self):
        """A question over 500 chars raises ValueError."""
        from app.rag_pipeline import handle_question
        with pytest.raises(ValueError, match="character limit"):
            handle_question("x" * 501)

    def test_forbidden_sql_raises_value_error(self):
        """SQL containing forbidden keywords raises ValueError."""
        with patch("app.rag_pipeline.retrieve_relevant_schema", return_value=["chunk"]), \
             patch("app.rag_pipeline.generate_sql", return_value="DROP TABLE customers"):
            from app.rag_pipeline import handle_question
            with pytest.raises(ValueError, match="safety validation"):
                handle_question("Delete all customers")

    def test_schema_retrieval_failure_raises_runtime_error(self):
        """Failure in schema retrieval raises RuntimeError with a user-friendly message."""
        with patch("app.rag_pipeline.retrieve_relevant_schema", side_effect=Exception("pgvector down")):
            from app.rag_pipeline import handle_question
            with pytest.raises(RuntimeError, match="schema context"):
                handle_question("How many artists are there?")

    def test_empty_schema_chunks_raises_runtime_error(self):
        """Empty schema chunks raises RuntimeError."""
        with patch("app.rag_pipeline.retrieve_relevant_schema", return_value=[]):
            from app.rag_pipeline import handle_question
            with pytest.raises(RuntimeError, match="No schema context"):
                handle_question("How many artists are there?")


# ---------------------------------------------------------------------------
# Schema parser tests
# ---------------------------------------------------------------------------

class TestSchemaParser:
    """Tests for parse_schema_to_chunks()."""

    def _make_table(self, name="orders", cols=None, fks=None, samples=None):
        return {
            "table_name": name,
            "columns": cols or [
                {"name": "id", "type": "integer", "is_primary": True, "is_nullable": False},
                {"name": "total", "type": "numeric", "is_primary": False, "is_nullable": True},
            ],
            "foreign_keys": fks or [],
            "sample_rows": samples or [],
        }

    def test_returns_one_chunk_per_table(self):
        from app.schema_parser import parse_schema_to_chunks
        tables = [self._make_table("a"), self._make_table("b")]
        chunks = parse_schema_to_chunks(tables)
        assert len(chunks) == 2

    def test_chunk_contains_table_name(self):
        from app.schema_parser import parse_schema_to_chunks
        chunks = parse_schema_to_chunks([self._make_table("invoices")])
        assert "invoices" in chunks[0]

    def test_chunk_contains_pk_annotation(self):
        from app.schema_parser import parse_schema_to_chunks
        chunks = parse_schema_to_chunks([self._make_table()])
        assert "PK" in chunks[0]

    def test_chunk_contains_foreign_key(self):
        from app.schema_parser import parse_schema_to_chunks
        fks = [{"column": "customer_id", "referenced_table": "customers", "referenced_column": "id"}]
        cols = [
            {"name": "id", "type": "integer", "is_primary": True, "is_nullable": False},
            {"name": "customer_id", "type": "integer", "is_primary": False, "is_nullable": False},
        ]
        chunks = parse_schema_to_chunks([self._make_table(cols=cols, fks=fks)])
        assert "customers" in chunks[0]

    def test_chunk_contains_sample_values(self):
        from app.schema_parser import parse_schema_to_chunks
        samples = [{"id": 1, "total": 9.99}]
        chunks = parse_schema_to_chunks([self._make_table(samples=samples)])
        assert "Sample rows" in chunks[0]

    def test_empty_input_returns_empty_list(self):
        from app.schema_parser import parse_schema_to_chunks
        assert parse_schema_to_chunks([]) == []
