"""`ragaudit improve` on the mock LLM: end to end, KNOBS discovery, no knobs, budget stop, question reuse."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragaudit import history
from ragaudit.cli import app
from ragaudit.cli_improve import _knobs_space
from ragaudit.models.question import Split
from ragaudit.storage.store import Store

runner = CliRunner()
SAMPLE = Path(__file__).resolve().parent.parent / "demo" / "sample_corpus"

PLAIN_NO_KNOBS = '''
import re
from pathlib import Path
from ragaudit.models.adapter import RAGResponse

DOCS = Path(__file__).parent / "docs"

def query(question):
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    paras = [p for f in sorted(DOCS.iterdir()) for p in f.read_text().split("\\n\\n") if p.strip()]
    ranked = sorted(paras, key=lambda p: -len(words & set(re.findall(r"[a-z0-9]+", p.lower()))))[:3]
    return RAGResponse(answer=ranked[0], contexts=ranked)
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(SAMPLE, tmp_path / "docs")
    assert runner.invoke(app, ["init", "--framework", "demo"]).exit_code == 0
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    return tmp_path, opened


def improve(*args):
    return runner.invoke(app, ["improve", "--docs", "docs", "--llm", "mock", "--yes", "--max-trials", "3", *args])


def store_of(tmp_path):
    return Store(tmp_path / ".ragaudit" / "store.db")


def test_improve_runs_the_loop_end_to_end(project):
    tmp_path, opened = project
    result = improve("--note", "first tune")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Estimated LLM calls: about" in out and "Budget: 400" in out
    assert "  generate 1/" in out and "  trial 1/3 practice" in out and "  final exam: baseline" in out
    summary = out.split("ragaudit improve: summary\n", 1)[1].strip().splitlines()
    assert len(summary) == 4 and summary[0].strip().startswith("Final exam")
    assert (tmp_path / "best_config.yaml").is_file() and "top_k" in (tmp_path / "best_config.yaml").read_text()
    report = (tmp_path / "ragaudit-report.html").read_text()
    assert "Self-improvement loop" in report and "first tune" in report and "No improvement run" not in report
    assert opened and opened[0].endswith("ragaudit-report.html")
    with store_of(tmp_path) as store:
        rows = history.load_history(store)
        assert [r.kind for r in rows] == ["improve"]
        assert rows[0].extra["trials"] and rows[0].n_questions == len(store.question_ids(Split.TEST))


def test_a_second_run_reuses_every_question(project):
    tmp_path, _ = project
    assert improve("--no-open").exit_code == 0
    with store_of(tmp_path) as store:
        before = (store.question_ids(Split.DEV), store.question_ids(Split.TEST))
    second = improve("--no-open")
    assert second.exit_code == 0, second.output
    assert "writing more from 0 passage(s)" in second.output and "Questions: 0 new" in second.output
    assert "  generate" not in second.output
    with store_of(tmp_path) as store:
        assert (store.question_ids(Split.DEV), store.question_ids(Split.TEST)) == before
        rows = history.load_history(store)
    assert rows[1].prev_run_id == rows[0].run_id and rows[1].shared_questions == rows[0].n_questions


def test_knobs_are_discovered_from_the_adapter(project):
    tmp_path, _ = project
    space, has_factory = _knobs_space(tmp_path / "ragaudit_adapter.py", None)
    assert has_factory and set(space.knobs) == {"chunk_size", "top_k", "prompt"}
    assert space.baseline == {"chunk_size": "paragraph", "top_k": 3, "prompt": "strict"}
    yaml_space = tmp_path / "space.yaml"
    yaml_space.write_text("knobs:\n  top_k: [5, 8]\n")
    override, _ = _knobs_space(tmp_path / "ragaudit_adapter.py", yaml_space)
    assert override.knobs == {"top_k": [5, 8]}


def test_init_templates_declare_knobs_and_make_adapter(tmp_path):
    for framework in ("plain", "langchain", "llamaindex"):
        out = tmp_path / f"{framework}.py"
        assert runner.invoke(app, ["init", "--framework", framework, "--out", str(out), "--force"]).exit_code == 0
        text = out.read_text()
        assert "KNOBS: dict[str, list] = {}" in text and "def make_adapter(config: dict)" in text
        assert text.count("# TODO") == 1
        space, has_factory = _knobs_space(out, None)
        assert space is None and has_factory  # empty KNOBS: improve only evaluates


def test_without_knobs_it_explains_and_still_reports(project):
    tmp_path, _ = project
    (tmp_path / "ragaudit_adapter.py").write_text(PLAIN_NO_KNOBS)
    result = improve("--no-open")
    assert result.exit_code == 0, result.output
    assert "declares no settings to tune" in result.output and "KNOBS" in result.output
    assert "trial" not in result.output and not (tmp_path / "best_config.yaml").exists()
    assert "Nothing was tuned" in result.output
    report = (tmp_path / "ragaudit-report.html").read_text()
    assert "nothing was tuned" in report and "No improvement run for this project yet" in report
    with store_of(tmp_path) as store:
        assert [r.kind for r in history.load_history(store)] == ["evaluate"]


def test_budget_stop_is_clean_and_keeps_progress(project):
    tmp_path, _ = project
    result = improve("--no-open", "--budget-calls", "30")
    assert result.exit_code == 0, result.output
    assert "Stopped" in result.output and "--budget-calls" in result.output and "Traceback" not in result.output
    assert "add KNOBS" not in result.output  # the adapter has KNOBS; the budget is what stopped it
    assert not (tmp_path / "ragaudit-report.html").exists()
    with store_of(tmp_path) as store:
        saved = len(store.question_ids(Split.DEV)) + len(store.question_ids(Split.TEST))
        assert store.attempted_source_keys()  # the passages it processed are remembered
        assert history.load_history(store) == []
    resumed = improve("--no-open")
    assert resumed.exit_code == 0, resumed.output
    assert f"Questions: {saved} reused" in resumed.output


def test_cancel_sends_nothing(project):
    tmp_path, _ = project
    result = runner.invoke(app, ["improve", "--docs", "docs", "--llm", "mock"], input="n\n")
    assert result.exit_code == 0 and "Cancelled. Nothing was sent to the LLM." in result.output
    assert not (tmp_path / "ragaudit-report.html").exists()


def test_knobs_without_make_adapter_is_explained(project):
    tmp_path, _ = project
    (tmp_path / "ragaudit_adapter.py").write_text(PLAIN_NO_KNOBS + '\nKNOBS = {"top_k": [3, 5]}\n')
    result = improve("--no-open")
    assert result.exit_code == 1 and "no make_adapter(config)" in result.output
