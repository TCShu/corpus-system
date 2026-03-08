import pytest
from sqlalchemy import text, inspect


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def test_database_is_reachable(db_session):
    result = db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

EXPECTED_TABLES = [
    "users",
    "corpora",
    "documents",
    "queries",
    "analysis_results",
    "agent_execution_logs",
    "frequency_analysis",
    "kwic_entries",
    "ngrams",
    "document_chunks",
    "vector_embeddings",
]


@pytest.fixture(scope="module")
def existing_tables(db_session):
    inspector = inspect(db_session.bind)
    return inspector.get_table_names()


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_table_exists(table, existing_tables):
    assert table in existing_tables, f"Table '{table}' was not found in the database"
