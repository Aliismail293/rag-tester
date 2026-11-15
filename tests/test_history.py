"""v3 run history: question-set reuse in `check`, shared-question deltas, per-project stores, migration."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from ragaudit import history
from ragaudit.cli import app
from ragaudit.storage.store import SCHEMA_VERSION, Store

runner = CliRunner()

ADAPTER = '''
import re
from pathlib import Path
from ragaudit.models.adapter import RAGResponse

DOCS = Path(__file__).parent / "docs"

def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def query(question):
    paragraphs = [p.read_text() for p in sorted(DOCS.iterdir())]
    ranked = sorted(paragraphs, key=lambda s: -len(words(question) & words(s)))[:2]
    return RAGResponse(answer=ranked[0], contexts=ranked)
'''

DOCS = {
    "a.md": "The Riverside Line opened in 1991 and carries about forty thousand riders on a weekday.",
    "b.md": "Fenwick Depot houses ninety trams and sits on the eastern edge of Millbrook.",
    "c.md": "A single adult fare costs 2.40 pounds and a day ticket costs 6.50 pounds.",
    "d.md": "Weirside Depot opened in 2004 and handles heavy maintenance for 120 trams.",
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    (tmp_path / "ragaudit_adapter.py").write_text(ADAPTER)
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    return tmp_path


def check(*args):
    return runner.invoke(app, ["check", "--docs", "docs", "--llm", "mock", "--yes", "--no-open", *args])


def rows(project_dir):
    with Store(project_dir / ".ragaudit" / "store.db") as store:
        return history.load_history(store)


def test_check_reuses_its_question_set_when_docs_are_unchanged(project):
    first = check("--questions", "4")
    assert first.exit_code == 0, first.output
    assert "writing a new set of 4 (first check)" in first.output
    second = check("--questions", "4", "--note", "same again")
    assert second.exit_code == 0, second.output
    assert "reusing 4 from earlier checks" in second.output
    assert "0 to write questions" in second.output
    assert "on the 4 question(s) both runs share" in second.output

    a, b = rows(project)
    assert a.question_set_id == b.question_set_id and a.n_questions == 4
    assert b.prev_run_id == a.run_id and b.shared_questions == 4 and b.delta_true_pass is not None
    assert b.note == "same again" and not b.corpus_changed and a.corpus_fingerprint == b.corpus_fingerprint


def test_editing_one_doc_regenerates_only_its_questions(project):
    assert check("--questions", "4").exit_code == 0
    (project / "docs" / "c.md").write_text("A single adult fare costs 2.80 pounds and a week ticket costs 19 pounds.")
    second = check("--questions", "4")
    assert second.exit_code == 0, second.output
    assert "reusing 3 from earlier checks" in second.output
    assert "writing 1 for new or edited passages (0 new, 1 edited, 0 deleted)" in second.output
    assert "1 to write questions" in second.output

    a, b = rows(project)
    assert b.corpus_changed and b.shared_questions == 3 and b.n_questions == 4
    assert a.question_set_id != b.question_set_id


def test_added_and_deleted_docs(project):
    assert check("--questions", "4").exit_code == 0
    (project / "docs" / "d.md").unlink()
    (project / "docs" / "e.md").write_text("The night tram leaves Millbrook Exchange at 23:40 on Fridays.")
    result = check("--questions", "4")
    assert result.exit_code == 0, result.output
    assert "1 new, 0 edited, 1 deleted" in result.output and "retiring 1" in result.output
    b = rows(project)[-1]
    assert b.corpus_changed and b.shared_questions == 3 and b.n_questions == 4


def test_fresh_questions_and_a_new_count_start_a_new_set(project):
    assert check("--questions", "3").exit_code == 0
    fresh = check("--questions", "3", "--fresh-questions")
    assert "writing a new set of 3 (--fresh-questions)" in fresh.output
    recount = check("--questions", "2")
    assert "--questions changed from 3 to 2" in recount.output
    a, b, c = rows(project)
    assert b.n_questions == 3 and c.n_questions == 2
    # The mock writes the same question from the same passage, so a fresh set can still share questions.
    assert all(r.kind == "check" for r in (a, b, c))


def test_history_is_kept_per_project(tmp_path, monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    for name in ("one", "two"):
        root = tmp_path / name
        (root / "docs").mkdir(parents=True)
        for doc, text in DOCS.items():
            (root / "docs" / doc).write_text(text)
        (root / "ragaudit_adapter.py").write_text(ADAPTER)
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        result = runner.invoke(app, [
            "check", "--docs", "one/docs", "--adapter", "one/ragaudit_adapter.py", "--llm", "mock", "--yes",
            "--no-open", "--questions", "2",
        ])
        assert result.exit_code == 0, result.output
    result = runner.invoke(app, [
        "check", "--docs", "two/docs", "--adapter", "two/ragaudit_adapter.py", "--llm", "mock", "--yes",
        "--no-open", "--questions", "2",
    ])
    assert result.exit_code == 0, result.output
    assert len(rows(tmp_path / "one")) == 2
    two = rows(tmp_path / "two")
    assert len(two) == 1 and two[0].prev_run_id is None


def _record(store, run_id, outcomes, minutes, kind="check"):
    return history.record_run(
        store, kind=kind, run_id=run_id, outcomes=outcomes,
        created_at=datetime(2026, 11, 15, tzinfo=timezone.utc) + timedelta(minutes=minutes),
    )


def test_deltas_use_only_the_questions_both_runs_share(tmp_path):
    with Store(tmp_path / "s.db") as store:
        _record(store, "r1", {"q1": "true_pass", "q2": "retrieval_failure", "q3": "true_pass"}, 0)
        # q3 retired, q4 new and failing: the headline drops, but on shared q1/q2 the run improved.
        second = _record(store, "r2", {"q1": "true_pass", "q2": "true_pass", "q4": "retrieval_failure"}, 1)
        assert second.shared_questions == 2
        assert second.delta_true_pass == 50.0 and second.delta_retrieval_hit == 50.0
        assert second.true_pass_pct == pytest.approx(66.7)
        assert second.small_sample

        unrelated = _record(store, "r3", {"x": "true_pass"}, 2)
        assert unrelated.shared_questions == 0 and unrelated.delta_true_pass is None

        other_kind = _record(store, "e1", {"q1": "lucky_pass"}, 3, kind="evaluate")
        assert other_kind.prev_run_id is None  # deltas compare like with like

        loaded = history.load_history(store)
        assert [r.run_id for r in loaded] == ["r1", "r2", "r3", "e1"]
        report_rows = history.history_for_report(loaded)
        assert report_rows[1]["number"] == 2 and "outcomes" not in report_rows[1]


def test_history_counts_and_corpus_change(tmp_path):
    with Store(tmp_path / "s.db") as store:
        a = history.record_run(store, kind="check", run_id="a", corpus_fingerprint="f1",
                               outcomes={"q1": "lucky_pass", "q2": "generation_failure"})
        assert a.lucky_pass_count == 1 and a.retrieval_hit_pct == 50.0 and a.correctness_pct == 50.0
        assert not a.corpus_changed
        b = history.record_run(store, kind="check", run_id="b", corpus_fingerprint="f2", outcomes={"q1": "true_pass"})
        assert b.corpus_changed
        with pytest.raises(ValueError):
            history.record_run(store, kind="nope", run_id="c", outcomes={})


def test_a_schema_2_store_migrates_and_keeps_its_data(tmp_path):
    path = tmp_path / "store.db"
    with Store(path) as store:
        store.record_corpus_version("idx", {"c1": "h1"})
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE history")
    conn.execute("DROP TABLE check_questions")
    conn.execute("DROP TABLE store_meta")
    conn.execute("UPDATE schema_version SET version = 2")
    conn.commit()
    conn.close()

    with Store(path) as store:
        assert store.schema_version == SCHEMA_VERSION == 3
        assert store.latest_corpus_version() == ("idx", {"c1": "h1"})
        assert history.load_history(store) == []
        history.record_run(store, kind="evaluate", run_id="e", outcomes={"q": "true_pass"})
    with Store(path) as store:
        assert len(history.load_history(store)) == 1


def test_evaluate_and_tune_write_history_rows(tmp_path, monkeypatch):
    from pathlib import Path

    demo = Path(__file__).resolve().parent.parent / "demo" / "v2"
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "store.db"
    common = ["--store", str(store), "--llm", "mock"]
    corpus = str(demo / "corpus.jsonl")
    assert runner.invoke(app, ["generate", "--corpus", corpus, *common]).exit_code == 0
    result = runner.invoke(app, ["evaluate", "--adapter", str(demo / "toy_adapter.py"), "--corpus", corpus,
                                 "--note", "baseline", *common])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["tune", "--adapter", str(demo / "toy_adapter.py"), "--search-space",
                                 str(demo / "search_space.yaml"), "--max-trials", "3", "--out",
                                 str(tmp_path / "best.yaml"), *common])
    assert result.exit_code == 0, result.output
    with Store(store) as s:
        loaded = history.load_history(s)
    kinds = [r.kind for r in loaded]
    assert kinds == ["evaluate", "tune"]
    assert loaded[0].note == "baseline" and loaded[0].n_questions > 0
    tune_row = loaded[1]
    assert tune_row.extra["trials"] and tune_row.extra["n_exam"] == tune_row.n_questions
    assert len(tune_row.outcomes) == tune_row.n_questions
