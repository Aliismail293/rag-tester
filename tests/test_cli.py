from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragaudit.cli import app
from ragaudit.models.adapter import RAGResponse

runner = CliRunner()


class StubClient:
    """Replaces LLMClient in cli.py so tests never touch the network."""

    def __init__(self, *args, **kwargs):
        self.model = "stub-model"

    def complete(self, system: str, user: str) -> str:
        if "generating a test set" in system:
            return '{"questions": [{"question": "What is in this document?", "answer": "some fact"}]}'
        if "judging whether a candidate answer" in system:
            return '{"is_correct": true, "score": 1.0, "rationale": "matches"}'
        if "judging whether an answer is faithfully grounded" in system:
            return '{"is_grounded": true, "score": 1.0, "rationale": "grounded"}'
        return "a plain no-context guess"


@pytest.fixture(autouse=True)
def stub_llm_client(monkeypatch):
    monkeypatch.setattr("ragaudit.cli.LLMClient", StubClient)


@pytest.fixture
def corpus_dir(tmp_path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc1.txt").write_text("This document contains some fact about widgets.")
    return docs_dir


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "ragaudit.db"


@pytest.fixture
def adapter_path(tmp_path) -> Path:
    adapter_file = tmp_path / "my_adapter.py"
    adapter_file.write_text(
        "from ragaudit.models.adapter import RAGResponse\n\n"
        "def query(question: str) -> RAGResponse:\n"
        '    return RAGResponse(answer="some fact", retrieved_chunk_ids=[], contexts=[])\n'
    )
    return adapter_file


def assert_clean_failure(result):
    assert result.exit_code != 0
    assert "Traceback" not in result.output


# --- error paths: commands run out of order ---


def test_generate_before_ingest_gives_clear_error(db_path):
    result = runner.invoke(app, ["generate", "--db", str(db_path)])
    assert_clean_failure(result)
    assert "ingest" in result.output.lower()


def test_run_before_generate_gives_clear_error(corpus_dir, db_path, adapter_path):
    ingest_result = runner.invoke(app, ["ingest", str(corpus_dir), "--db", str(db_path)])
    assert ingest_result.exit_code == 0

    result = runner.invoke(app, ["run", "--adapter", str(adapter_path), "--db", str(db_path)])
    assert_clean_failure(result)
    assert "no test set found" in result.output.lower()
    assert "generate" in result.output.lower()


def test_report_before_run_gives_clear_error(corpus_dir, db_path):
    runner.invoke(app, ["ingest", str(corpus_dir), "--db", str(db_path)])
    runner.invoke(app, ["generate", "--db", str(db_path)])

    result = runner.invoke(app, ["report", "--db", str(db_path), "--out", "unused.html"])
    assert_clean_failure(result)
    assert "no run found" in result.output.lower()


# --- error paths: missing db ---


def test_generate_with_missing_db_gives_clear_error(tmp_path):
    missing_db = tmp_path / "does_not_exist.db"
    result = runner.invoke(app, ["generate", "--db", str(missing_db)])
    assert_clean_failure(result)
    assert "database not found" in result.output.lower()


def test_run_with_missing_db_gives_clear_error(tmp_path, adapter_path):
    missing_db = tmp_path / "does_not_exist.db"
    result = runner.invoke(app, ["run", "--adapter", str(adapter_path), "--db", str(missing_db)])
    assert_clean_failure(result)
    assert "database not found" in result.output.lower()


def test_report_with_missing_db_gives_clear_error(tmp_path):
    missing_db = tmp_path / "does_not_exist.db"
    result = runner.invoke(app, ["report", "--db", str(missing_db), "--out", "unused.html"])
    assert_clean_failure(result)
    assert "database not found" in result.output.lower()


# --- error paths: missing adapter file ---


def test_run_with_missing_adapter_file_gives_clear_error(corpus_dir, db_path):
    runner.invoke(app, ["ingest", str(corpus_dir), "--db", str(db_path)])
    runner.invoke(app, ["generate", "--db", str(db_path)])

    result = runner.invoke(
        app, ["run", "--adapter", str(db_path.parent / "nonexistent_adapter.py"), "--db", str(db_path)]
    )
    assert_clean_failure(result)
    assert "not found" in result.output.lower()


# --- error paths: ingest itself ---


def test_ingest_missing_path_gives_clear_error(db_path, tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope"), "--db", str(db_path)])
    assert_clean_failure(result)
    assert "not found" in result.output.lower()


def test_ingest_empty_directory_gives_clear_error(tmp_path, db_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = runner.invoke(app, ["ingest", str(empty_dir), "--db", str(db_path)])
    assert_clean_failure(result)
    assert "no .txt/.md documents" in result.output.lower()


# --- happy path: full pipeline wiring, stubbed LLM, no network ---


def test_full_pipeline_ingest_generate_run_report(corpus_dir, db_path, adapter_path, tmp_path):
    ingest_result = runner.invoke(app, ["ingest", str(corpus_dir), "--db", str(db_path)])
    assert ingest_result.exit_code == 0
    assert "Ingested 1 document(s) into" in ingest_result.output

    generate_result = runner.invoke(app, ["generate", "--n", "1", "--db", str(db_path)])
    assert generate_result.exit_code == 0
    assert "Generated" in generate_result.output

    run_result = runner.invoke(app, ["run", "--adapter", str(adapter_path), "--db", str(db_path)])
    assert run_result.exit_code == 0
    assert "complete" in run_result.output.lower()

    out_path = tmp_path / "report.html"
    report_result = runner.invoke(app, ["report", "--db", str(db_path), "--out", str(out_path)])
    assert report_result.exit_code == 0
    assert out_path.exists()

    html = out_path.read_text(encoding="utf-8")
    assert "ragaudit report" in html
    assert "What is in this document?" in html
