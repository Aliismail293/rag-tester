"""`ragaudit init --framework demo`: a pipeline that works with zero edits, on the mock LLM."""

import os

import pytest
from typer.testing import CliRunner

from ragaudit.adapters.base import load_adapter
from ragaudit.cli import app
from ragaudit.models.adapter import RAGResponse

runner = CliRunner()

TRAMS = (
    "The Riverside Line opened in 1991 and was resignalled in 2023.\n\n"
    "It runs from Millbrook Exchange to Ostley Quarry Gate and carries about forty thousand riders on a weekday.\n\n"
    "Night trams run every thirty minutes on Fridays and Saturdays."
)
DEPOTS = (
    "Fenwick Depot, on the eastern edge of Millbrook, is the largest depot and houses ninety trams.\n\n"
    "Weirside Depot is smaller and handles heavy maintenance for the whole fleet."
)
FARES = "A single adult fare costs 2.40 pounds and a day ticket costs 6.50 pounds."


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGAUDIT_LLM", raising=False)
    monkeypatch.delenv("RAGAUDIT_DEMO_DOCS", raising=False)
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "trams.md").write_text(TRAMS)
    (docs / "sub" / "depots.txt").write_text(DEPOTS)
    (docs / "fares.txt").write_text(FARES)
    (docs / "ignored.pdf").write_text("Fenwick Fenwick Fenwick depot depot")
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    return tmp_path


def test_init_demo_writes_a_ready_adapter(project):
    result = runner.invoke(app, ["init", "--framework", "demo"])
    assert result.exit_code == 0, result.output
    assert "ragaudit check --docs docs --adapter ragaudit_adapter.py" in result.output
    assert "--llm mock" in result.output
    assert (project / ".env.example").is_file()
    assert "does not exist yet" not in result.output


def test_init_demo_warns_when_the_docs_folder_is_missing(project):
    result = runner.invoke(app, ["init", "--framework", "demo", "--docs", "my-docs"])
    assert result.exit_code == 0 and "my-docs does not exist yet" in result.output


def load_demo(monkeypatch, path="ragaudit_adapter.py"):
    monkeypatch.setenv("RAGAUDIT_LLM", "mock")
    return load_adapter(path)


def test_demo_retrieves_the_top_three_paragraphs_by_word_overlap(project, monkeypatch):
    runner.invoke(app, ["init", "--framework", "demo"])
    adapter = load_demo(monkeypatch)
    response = adapter.query("Which depot houses ninety trams?")
    assert isinstance(response, RAGResponse)
    assert len(response.contexts) == 3
    assert response.contexts[0].startswith("Fenwick Depot")  # paragraphs, from nested folders too
    assert all("\n" not in c for c in response.contexts) and not any("Fenwick Fenwick" in c for c in response.contexts)
    assert "ninety trams" in response.answer  # the mock LLM answers from the retrieved context
    assert response.latency_ms is not None


def test_demo_always_returns_the_top_three_like_a_search_engine(project, monkeypatch):
    runner.invoke(app, ["init", "--framework", "demo"])
    response = load_demo(monkeypatch).query("zzz qqq?")
    assert len(response.contexts) == 3 and response.answer


def test_demo_finds_docs_relative_to_the_adapter_file(project, monkeypatch, tmp_path):
    out = project / "rag" / "adapter.py"
    assert runner.invoke(app, ["init", "--framework", "demo", "--out", str(out)]).exit_code == 0
    monkeypatch.chdir(project / "rag")  # run from elsewhere: the path is baked relative to the adapter
    assert load_demo(monkeypatch, str(out)).query("Weirside Depot maintenance?").contexts


def test_demo_explains_a_missing_docs_folder(project, monkeypatch):
    runner.invoke(app, ["init", "--framework", "demo", "--docs", "nowhere"])
    with pytest.raises(FileNotFoundError, match="docs folder not found"):
        load_demo(monkeypatch).query("anything?")


def test_doctor_passes_on_the_demo_with_mock(project):
    runner.invoke(app, ["init", "--framework", "demo"])
    result = runner.invoke(app, ["doctor", "--llm", "mock"])
    assert result.exit_code == 0, result.output
    assert "FAIL" not in result.output
    assert "RAGAUDIT_LLM" not in os.environ  # --llm is passed to the adapter, then restored


def test_check_runs_the_demo_with_zero_edits(project):
    assert runner.invoke(app, ["init", "--framework", "demo"]).exit_code == 0
    result = runner.invoke(app, ["check", "--docs", "docs", "--llm", "mock", "--questions", "4", "--yes", "--no-open"])
    assert result.exit_code == 0, result.output
    assert "The demo adapter answers through the same LLM: 4 more call(s)" in result.output
    assert "ragaudit check: summary" in result.output and "matched by text" in result.output
    assert "Retrieval hit rate: 0%" not in result.output
    assert (project / "ragaudit-report.html").is_file()
    assert "RAGAUDIT_LLM" not in os.environ and "RAGAUDIT_DEMO_DOCS" not in os.environ


def test_check_points_the_demo_at_its_own_docs_folder(project):
    other = project / "other-docs"
    other.mkdir()
    (other / "zoo.md").write_text(
        "The city zoo opened in 1962 and keeps four snow leopards in the northern enclosure.\n\n"
        "Feeding time for the penguins is at half past two every afternoon."
    )
    runner.invoke(app, ["init", "--framework", "demo"])  # baked folder: docs/
    result = runner.invoke(app, ["check", "--docs", "other-docs", "--llm", "mock", "--questions", "2", "--yes", "--no-open"])
    assert result.exit_code == 0, result.output
    assert "Retrieval hit rate: 100%" in result.output


def test_cancelling_the_demo_says_one_test_call_was_made(project):
    runner.invoke(app, ["init", "--framework", "demo"])
    result = runner.invoke(app, ["check", "--docs", "docs", "--llm", "mock", "--questions", "2"], input="n\n")
    assert result.exit_code == 0 and "Only the adapter's one test call" in result.output
    assert not (project / "ragaudit-report.html").exists()
