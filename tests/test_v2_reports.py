import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragaudit.cli import app
from ragaudit.generate.pipeline import GenerateSummary
from ragaudit.models.question import Question, QuestionStatus, RejectReason, Split
from ragaudit.report.generator import generate_report
from ragaudit.report.v2 import generate_generate_report, generate_tune_report, overlap_histogram
from ragaudit.storage import db as db_module
from ragaudit.storage import repository
from ragaudit.storage.lock import _unlock_test_split
from ragaudit.storage.store import Store

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo" / "v2"


@pytest.fixture(scope="module")
def demo_store(tmp_path_factory):
    """A store with one mock generate run and one short llm-strategy tune run on the v2 demo."""
    tmp = tmp_path_factory.mktemp("reports")
    store_path = str(tmp / "store.db")
    corpus = str(DEMO_DIR / "corpus.jsonl")
    runner = CliRunner()
    assert runner.invoke(app, ["generate", "--corpus", corpus, "--store", store_path, "--llm", "mock"]).exit_code == 0
    result = runner.invoke(
        app,
        [
            "tune", "--adapter", str(DEMO_DIR / "toy_adapter.py"), "--search-space", str(DEMO_DIR / "search_space.yaml"),
            "--corpus", corpus, "--store", store_path, "--llm", "mock", "--strategy", "llm", "--max-trials", "5",
            "--out", str(tmp / "best.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    return Path(store_path)


def render(kind: str, store_path: Path, out: Path, *extra: str):
    return CliRunner().invoke(app, ["report", "--kind", kind, "--store", str(store_path), "--out", str(out), *extra])


def visible_text(page: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))


def test_generate_report_shows_counts_reasons_split_and_overlap(demo_store, tmp_path):
    result = render("generate", demo_store, tmp_path / "gen.html")
    assert result.exit_code == 0, result.output
    page = (tmp_path / "gen.html").read_text()
    with Store(demo_store) as store:
        counts = store.question_counts()
    text = visible_text(page)

    assert "built with the mock LLM" in page
    assert "Why questions were rejected" in page and "too_lexical" in page
    assert "Lexical overlap" in page and "threshold 0.5" in page
    assert f"{counts['by_split_status']['dev']['active']}" in text
    assert "test (locked)" in text
    assert page.count("<svg") >= 2
    assert "<script src" not in page and "<link" not in page  # self-contained


def test_generate_report_never_shows_test_split_question_text(demo_store, tmp_path):
    render("generate", demo_store, tmp_path / "gen.html")
    page = (tmp_path / "gen.html").read_text()
    with Store(demo_store) as store, _unlock_test_split():
        test_questions = store.load_questions(split=Split.TEST)
    assert test_questions
    for q in test_questions:
        assert q.question not in page


def test_generate_report_refuses_test_split_examples(tmp_path):
    q = Question.create(["c1"], ["h1"], "fact", "question?", "answer", "gen")
    q.split = Split.TEST
    summary = GenerateSummary(run_id="r", created_at=q.created_at, generator_model="g", verifier_model="v", corpus_chunks=1)
    totals = {"by_split_status": {s: {"active": 0, "stale": 0, "rejected": 0} for s in ("dev", "test")}}
    with pytest.raises(ValueError, match="dev split only"):
        generate_generate_report(summary, totals, [q], [], tmp_path / "x.html")


def test_generate_report_escapes_question_text(tmp_path):
    q = Question.create(["c1"], ["h1"], "fact <b>", "<script>alert(1)</script>?", "a&b", "gen")
    q.split = Split.DEV
    rejected = Question.create(["c2"], ["h2"], "f", "<img src=x>?", "a", "gen")
    rejected.status, rejected.reject_reason = QuestionStatus.REJECTED, RejectReason.TOO_LEXICAL.value
    summary = GenerateSummary(
        run_id="r", created_at=q.created_at, generator_model="g", verifier_model="v", corpus_chunks=2,
        generated=1, rejected=1, reject_reasons={"too_lexical": 1}, lexical_overlaps=[0.2, 0.9],
    )
    totals = {"by_split_status": {s: {"active": 0, "stale": 0, "rejected": 0} for s in ("dev", "test")}}
    page = generate_generate_report(summary, totals, [q], [rejected], tmp_path / "x.html").read_text()
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<img src=x>" not in page
    assert "mock LLM" not in page  # real model names, no mock banner


def test_overlap_histogram_bins_against_the_threshold():
    _, rows = overlap_histogram([0.0, 0.1, 0.45, 0.5, 0.51, 1.0], threshold=0.5)
    assert sum(r["count"] for r in rows) == 6
    assert rows[0]["count"] == 2  # [0.0, 0.1]
    assert rows[4]["count"] == 2 and rows[4]["kept"]  # (0.4, 0.5]
    assert rows[5]["count"] == 1 and not rows[5]["kept"]  # (0.5, 0.6]
    assert rows[9]["count"] == 1


def test_tune_report_shows_trials_dev_vs_test_failures_overfit_and_best_config(demo_store, tmp_path):
    result = render("tune", demo_store, tmp_path / "tune.html")
    assert result.exit_code == 0, result.output
    page = (tmp_path / "tune.html").read_text()
    with Store(demo_store) as store:
        tune = store.load_tune_result()
    text = visible_text(page)

    assert "Dev score per trial" in page and "Failure classes before and after" in page
    assert f"{tune.baseline_test.score:.1f}" in text and f"{tune.best_test.score:.1f}" in text
    assert "Overfitting check" in page
    assert ("Likely overfit" in page) == tune.overfit_warning
    for knob in tune.search_space:
        assert knob in page
    assert len(re.findall(r'class="v2-hit"', page)) == len(tune.trials)


def test_tune_report_flags_overfitting(demo_store, tmp_path):
    with Store(demo_store) as store:
        tune = store.load_tune_result()
    tune.overfit_warning, tune.overfit_gap = True, 25.0
    page = generate_tune_report(tune, tmp_path / "t.html").read_text()
    assert "Likely overfit to the dev set" in page


def test_report_kind_by_id_and_errors(demo_store, tmp_path):
    with Store(demo_store) as store:
        tune_id = store.load_tune_result().tune_id
    assert render("tune", demo_store, tmp_path / "t.html", "--run-id", tune_id).exit_code == 0
    missing = render("tune", demo_store, tmp_path / "t.html", "--run-id", "nope")
    assert missing.exit_code != 0 and "not found" in missing.output
    bad_kind = render("bogus", demo_store, tmp_path / "t.html")
    assert bad_kind.exit_code != 0 and "--kind" in bad_kind.output
    no_store = render("generate", tmp_path / "absent.db", tmp_path / "g.html")
    assert no_store.exit_code != 0 and "not found" in no_store.output
    assert not (tmp_path / "absent.db").exists()


def test_v1_report_command_output_is_unchanged(tmp_path):
    """The default `report` path renders exactly what the v1 generator renders."""
    db = ROOT / "demo" / "output" / "ragaudit.db"
    conn = db_module.get_connection(db)
    try:
        run = repository.load_run(conn, repository.get_latest_run_id(conn))
        test_set = repository.load_test_set(conn, run.config.testset_id)
    finally:
        conn.close()
    expected = generate_report(run, test_set, tmp_path / "direct.html").read_bytes()

    result = CliRunner().invoke(app, ["report", "--db", str(db), "--out", str(tmp_path / "cli.html")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli.html").read_bytes() == expected
    assert b"v2-" not in expected  # no v2 styles leak into v1 reports


def test_v2_demo_script_runs_end_to_end_offline(tmp_path):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.update(
        RAGAUDIT_LLM="mock",
        RAGAUDIT_STORE=str(tmp_path / "store.db"),
        RAGAUDIT_DEMO_OUT=str(tmp_path / "out"),
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",
    )
    result = subprocess.run(["sh", str(DEMO_DIR / "run_demo.sh")], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mode regression" in result.stdout and "mode cached" in result.stdout
    for name in ("generate-report.html", "tune-report.html", "best_config.yaml"):
        assert (tmp_path / "out" / name).is_file()
