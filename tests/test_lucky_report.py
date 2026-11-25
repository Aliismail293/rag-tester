import csv
from datetime import datetime

from ragaudit.models.adapter import RAGResponse
from ragaudit.models.run import Run, RunConfig
from ragaudit.models.scoring import (
    AblationResult,
    CorrectnessResult,
    GroundednessResult,
    RetrievalResult,
    ScoredCase,
    Verdict,
)
from ragaudit.models.testset import TestCase, TestSet
from ragaudit.report.lucky import (
    GUESS_LUCKY,
    MEMORY_LUCKY,
    UNCHECKED,
    bootstrap_ci,
    lucky_rows,
    render_markdown,
    summarize,
    write_lucky_report,
)

NOW = datetime(2026, 1, 1)


def make_scored_case(test_id: str, verdict: Verdict, ablation: AblationResult | None = None) -> ScoredCase:
    return ScoredCase(
        test_id=test_id,
        response=RAGResponse(answer="ans", retrieved_chunk_ids=[], contexts=[]),
        correctness=CorrectnessResult(is_correct=verdict in (Verdict.TRUE_PASS, Verdict.LUCKY_PASS), judge_model="j"),
        retrieval=RetrievalResult(
            hit=verdict in (Verdict.TRUE_PASS, Verdict.GENERATION_FAILURE), expected_chunk_id="c1", retrieved_chunk_ids=[]
        ),
        groundedness=GroundednessResult(is_grounded=True, judge_model="j"),
        verdict=verdict,
        ablation=ablation,
    )


def make_run(cases) -> tuple[Run, TestSet]:
    test_cases = [
        TestCase(
            test_id=c.test_id, question=f"Q {c.test_id}?", expected_answer=f"A {c.test_id}", source_chunk_id="c1",
            source_doc_id="d1", generation_model="g", created_at=NOW,
        )
        for c in cases
    ]
    run = Run(
        config=RunConfig(
            run_id="r1", testset_id="t1", adapter_path="a.py", correctness_judge_model="j",
            groundedness_judge_model="j", started_at=NOW,
        ),
        scored_cases=cases,
    )
    return run, TestSet(testset_id="t1", corpus_hash="h", created_at=NOW, generation_model="g", test_cases=test_cases)


def ablation(correct: bool) -> AblationResult:
    return AblationResult(answer="closed", is_correct=correct, confirms_lucky_pass=correct)


def sample_cases():
    return [
        make_scored_case("1", Verdict.TRUE_PASS),
        make_scored_case("2", Verdict.LUCKY_PASS, ablation=ablation(True)),
        make_scored_case("3", Verdict.LUCKY_PASS, ablation=ablation(False)),
        make_scored_case("4", Verdict.LUCKY_PASS),
        make_scored_case("5", Verdict.RETRIEVAL_FAILURE),
    ]


def test_lucky_kinds_split_by_closed_book_check():
    run, test_set = make_run(sample_cases())
    kinds = [r.lucky_kind for r in lucky_rows(run, test_set)]
    assert kinds == [None, MEMORY_LUCKY, GUESS_LUCKY, UNCHECKED, None]


def test_overstatement_is_accuracy_minus_true_pass_rate():
    run, test_set = make_run(sample_cases())
    s = summarize(lucky_rows(run, test_set))
    assert s.n == 5
    assert s.accuracy == 0.8
    assert s.true_pass_rate == 0.2
    assert abs(s.overstatement - (s.accuracy - s.true_pass_rate)) < 1e-9
    assert (s.lucky_passes, s.memory_lucky, s.guess_lucky, s.unchecked) == (3, 1, 1, 1)
    lo, hi = s.overstatement_ci
    assert 0.0 <= lo <= s.overstatement <= hi <= 1.0


def test_bootstrap_ci_is_seeded_and_degenerate_cases():
    flags = [True, False] * 20
    assert bootstrap_ci(flags) == bootstrap_ci(flags)
    assert bootstrap_ci([True] * 10) == (1.0, 1.0)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_markdown_lists_lucky_passes_and_notes_unchecked():
    run, test_set = make_run(sample_cases())
    rows = lucky_rows(run, test_set)
    md = render_markdown(run, rows, summarize(rows))
    assert "Overstatement" in md and "n = 5" in md
    assert "Q 2?" in md and "Q 1?" not in md
    assert "--no-ablation" in md


def test_markdown_with_no_lucky_passes():
    run, test_set = make_run([make_scored_case("1", Verdict.TRUE_PASS)])
    rows = lucky_rows(run, test_set)
    assert "None: every correct answer" in render_markdown(run, rows, summarize(rows))


def test_write_lucky_report_writes_markdown_and_csv(tmp_path):
    run, test_set = make_run(sample_cases())
    out = write_lucky_report(run, test_set, tmp_path / "lucky-report.md")
    assert out.exists()
    with (tmp_path / "lucky.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert rows[1]["lucky_kind"] == MEMORY_LUCKY

