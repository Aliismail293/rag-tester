"""`ragaudit check`: ingest -> generate -> run -> report in one command, on the mock LLM."""

import re

import pytest
from typer.testing import CliRunner

from ragaudit.cli import app
from ragaudit.cli_easy import plan_check
from ragaudit.models.adapter import RAGResponse
from ragaudit.models.run import RunSummary
from ragaudit.models.scoring import CorrectnessResult, GroundednessResult, RetrievalResult, ScoredCase, Verdict
from ragaudit.report.plain_summary import build_plain_summary

runner = CliRunner()

DOCS = {
    "trams.md": (
        "The Riverside Line opened in 1991 and was resignalled in 2023. It runs from Millbrook Exchange to "
        "Ostley Quarry Gate and carries about forty thousand riders on a weekday."
    ),
    "depots.md": (
        "Fenwick Depot, on the eastern edge of Millbrook, is the largest depot and houses ninety trams. "
        "Weirside Depot is smaller and handles heavy maintenance for the whole fleet."
    ),
    "fares.txt": (
        "A single adult fare costs 2.40 pounds and a day ticket costs 6.50 pounds. Children under five "
        "travel free, and riders over sixty pay half fare outside the morning peak."
    ),
}

# A pipeline with its own chunking (sentences) and ids: nothing ragaudit assigned.
ADAPTER = '''
import re
from pathlib import Path
from ragaudit.models.adapter import RAGResponse

DOCS = Path(__file__).parent / "docs"
SENTENCES = [s for p in sorted(DOCS.iterdir()) for s in re.split(r"(?<=[.]) ", p.read_text()) if s.strip()]

def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def query(question):
    ranked = sorted(SENTENCES, key=lambda s: -len(words(question) & words(s)))[:2]
    return RAGResponse(answer=ranked[0], contexts=ranked, retrieved_chunk_ids=["my-own-id"])
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    (tmp_path / "ragaudit_adapter.py").write_text(ADAPTER)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    return tmp_path, opened


def check(*args, input=None):
    return runner.invoke(app, ["check", "--docs", "docs", "--llm", "mock", *args], input=input)


def summary_lines(output: str) -> list[str]:
    tail = output.split("ragaudit check: summary\n", 1)[1]
    return [line for line in tail.splitlines() if line.strip()]


def test_check_runs_end_to_end_and_prints_a_short_summary(project):
    tmp_path, opened = project
    result = check("--yes", "--questions", "5")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ragaudit-report.html").is_file()
    assert (tmp_path / ".ragaudit" / "check.db").is_file()
    assert opened and opened[0].startswith("file://") and opened[0].endswith("ragaudit-report.html")

    lines = summary_lines(result.output)
    assert 1 + len(lines) <= 6  # title + body
    body = "\n".join(lines)
    assert re.search(r"Verdict: (HEALTHY|NEEDS WORK)", body)
    assert "Retrieval hit rate:" in body and "matched by text" in body
    assert "Answer correctness:" in body and "Lucky passes:" in body and "Most useful fix:" in body


def test_check_defaults_to_text_matching_so_own_chunking_still_scores(project):
    by_text = check("--yes", "--no-open", "--questions", "3")
    by_id = check("--yes", "--no-open", "--questions", "3", "--match-by", "id")
    assert by_text.exit_code == 0 and by_id.exit_code == 0
    assert "Retrieval hit rate: 0%" not in by_text.output
    assert "Retrieval hit rate: 0% (0 of 3" in by_id.output
    assert "--match-by text" in by_id.output  # the fix points id-mode users at text matching


def test_check_prints_the_estimate_and_asks_first(project):
    tmp_path, opened = project
    result = check("--questions", "3", input="n\n")
    assert result.exit_code == 0
    assert "Estimated LLM calls: about 9 (3 to write questions, 6 to judge answers), up to 15" in result.output
    assert "Continue?" in result.output and "Cancelled. Nothing was sent to the LLM." in result.output
    assert not (tmp_path / "ragaudit-report.html").exists() and not opened

    accepted = check("--questions", "3", "--no-open", input="y\n")
    assert accepted.exit_code == 0 and "ragaudit check: summary" in accepted.output


def test_check_enforces_the_budget_cap(project):
    tmp_path, _ = project
    result = check("--yes", "--no-open", "--questions", "3", "--max-llm-calls", "4")
    assert result.exit_code == 1
    assert "budget of 4 LLM call(s) was used up" in result.output and "--max-llm-calls" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "ragaudit-report.html").exists()


def test_check_never_exceeds_its_default_cap(project, monkeypatch):
    from ragaudit.llm import factory

    used = []
    original = factory.CallBudget.spend

    def spend(self):
        original(self)
        used.append(self.used)

    monkeypatch.setattr(factory.CallBudget, "spend", spend)
    result = check("--yes", "--no-open", "--questions", "4")
    assert result.exit_code == 0
    cap = int(re.search(r"Budget cap: (\d+)", result.output).group(1))
    assert used and max(used) <= cap
    assert f"Used {max(used)} LLM call(s)" in result.output


def test_no_open_skips_the_browser(project):
    _, opened = project
    assert check("--yes", "--no-open", "--questions", "2").exit_code == 0
    assert opened == []


def test_check_recreates_its_database(project):
    assert check("--yes", "--no-open", "--questions", "2").exit_code == 0
    assert check("--yes", "--no-open", "--questions", "3").exit_code == 0
    from ragaudit.storage import db as db_module
    from ragaudit.storage import repository

    conn = db_module.get_connection(project[0] / ".ragaudit" / "check.db")
    try:
        run = repository.load_run(conn, repository.get_latest_run_id(conn))
    finally:
        conn.close()
    assert run.summary.total_cases == 3


@pytest.mark.parametrize(
    "args, message",
    [
        (["--docs", "missing"], "docs folder not found"),
        (["--adapter", "nope.py"], "Adapter file not found"),
        (["--questions", "0"], "--questions must be between 1 and 500"),
        (["--match-by", "fuzzy"], "id or text"),
    ],
)
def test_check_fails_fast_before_any_llm_call(project, args, message):
    result = runner.invoke(app, ["check", "--docs", "docs", "--llm", "mock", "--yes", *args])
    assert result.exit_code == 1 and message in result.output
    assert "Estimated LLM calls" not in result.output or "--questions" not in message


def test_unfilled_template_stops_before_any_llm_call(project):
    tmp_path, _ = project
    (tmp_path / "ragaudit_adapter.py").unlink()
    assert runner.invoke(app, ["init"]).exit_code == 0
    result = check("--yes", "--no-open", "--questions", "2")
    assert result.exit_code == 1
    assert "not filled in" in result.output and "TODO in ragaudit_adapter.py" in result.output
    assert "Estimated LLM calls" not in result.output


def test_adapter_without_contexts_is_stopped_in_text_mode_only(project):
    tmp_path, _ = project
    (tmp_path / "ragaudit_adapter.py").write_text(
        "from ragaudit.models.adapter import RAGResponse\n"
        "def query(q):\n    return RAGResponse(answer='a', retrieved_chunk_ids=['x'])\n"
    )
    text = check("--yes", "--no-open", "--questions", "2")
    assert text.exit_code == 1 and "`contexts` is empty" in text.output
    assert check("--yes", "--no-open", "--questions", "2", "--match-by", "id").exit_code == 0


def test_plan_spreads_questions_over_the_corpus():
    chunks = list(range(10))
    plan = plan_check(chunks, 5)
    assert plan.chunks == [0, 2, 4, 6, 8] and plan.per_chunk == [1] * 5
    plan = plan_check(chunks[:3], 8)
    assert plan.chunks == [0, 1, 2] and plan.per_chunk == [3, 3, 2] and plan.questions == 8
    assert plan.generate_calls == 3 and plan.expected_calls == 3 + 16 and plan.max_calls == 3 + 32


# --- the plain-English summary ---


def case(verdict: Verdict, contexts=("ctx",), match_by="text") -> ScoredCase:
    hit = verdict in (Verdict.TRUE_PASS, Verdict.GENERATION_FAILURE)
    correct = verdict in (Verdict.TRUE_PASS, Verdict.LUCKY_PASS)
    return ScoredCase(
        test_id=f"t-{verdict}",
        response=RAGResponse(answer="a", contexts=list(contexts)),
        correctness=CorrectnessResult(is_correct=correct, judge_model="j"),
        retrieval=RetrievalResult(hit=hit, expected_chunk_id="c", retrieved_chunk_ids=[], match_by=match_by),
        groundedness=GroundednessResult(is_grounded=hit, judge_model="j"),
        verdict=verdict,
    )


def summarize(*verdicts, **kwargs):
    from ragaudit.runner.summary import compute_run_summary

    cases = [case(v, **kwargs) for v in verdicts]
    return build_plain_summary(compute_run_summary(cases), cases)


def test_summary_healthy():
    s = summarize(*[Verdict.TRUE_PASS] * 9, Verdict.GENERATION_FAILURE)
    assert s.status == "healthy" and s.verdict.startswith("Verdict: HEALTHY. 9 of 10")
    assert "Retrieval hit rate: 100%" in s.retrieval and "Answer correctness: 90%" in s.correctness
    assert len(s.lines()) == 5


@pytest.mark.parametrize(
    "verdicts, dominant, fix",
    [
        ([Verdict.RETRIEVAL_FAILURE] * 3 + [Verdict.TRUE_PASS], Verdict.RETRIEVAL_FAILURE, "improve retrieval"),
        ([Verdict.LUCKY_PASS] * 3 + [Verdict.TRUE_PASS], Verdict.LUCKY_PASS, "answered from memory"),
        ([Verdict.GENERATION_FAILURE] * 3 + [Verdict.TRUE_PASS], Verdict.GENERATION_FAILURE, "improve answer generation"),
        ([Verdict.RETRIEVAL_FAILURE, Verdict.LUCKY_PASS], Verdict.RETRIEVAL_FAILURE, "improve retrieval"),  # tie
    ],
)
def test_summary_fix_follows_the_dominant_failure(verdicts, dominant, fix):
    s = summarize(*verdicts)
    assert s.status == "needs_work" and s.dominant == dominant and fix in s.fix
    assert "NEEDS WORK" in s.verdict


def test_summary_when_adapter_returns_no_contexts():
    s = summarize(Verdict.RETRIEVAL_FAILURE, Verdict.LUCKY_PASS, contexts=())
    assert s.status == "needs_work" and "RAGResponse.contexts" in s.fix


def test_summary_with_no_id_hits_suggests_text_matching():
    s = summarize(Verdict.LUCKY_PASS, Verdict.RETRIEVAL_FAILURE, match_by="id")
    assert "--match-by text" in s.fix


def test_summary_of_an_empty_run():
    empty = RunSummary(
        total_cases=0, verdict_counts={}, retrieval_hit_rate=0.0, answer_correctness_rate=0.0,
        lucky_pass_rate=0.0, mean_groundedness=None, ablation_confirmed_count=None, judge_failure_count=0,
    )
    assert build_plain_summary(empty, []).status == "empty"


def test_summary_mentions_ablation_confirmations():
    from ragaudit.runner.summary import compute_run_summary

    cases = [case(Verdict.LUCKY_PASS), case(Verdict.TRUE_PASS)]
    summary = compute_run_summary(cases).model_copy(update={"ablation_confirmed_count": 1})
    assert "1 still correct with no context at all" in build_plain_summary(summary, cases).lucky
