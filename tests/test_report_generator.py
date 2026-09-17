import re
from datetime import datetime, timezone
from unittest import mock

from ragaudit.models.adapter import RAGResponse
from ragaudit.models.run import Run, RunConfig, RunSummary
from ragaudit.models.scoring import (
    AblationResult,
    CorrectnessResult,
    GroundednessResult,
    RetrievalResult,
    ScoredCase,
    Verdict,
)
from ragaudit.models.testset import TestCase, TestSet
from ragaudit.report.generator import generate_report


def make_test_case(test_id: str, question: str) -> TestCase:
    return TestCase(
        test_id=test_id,
        question=question,
        expected_answer=f"expected answer for {test_id}",
        source_chunk_id="chunk1",
        source_doc_id="doc1",
        generation_model="gen-model",
        created_at=datetime.now(timezone.utc),
    )


def make_scored_case(
    test_id: str,
    verdict: Verdict,
    rank: int | None,
    answer: str = "some answer",
    judge_parse_failed: bool = False,
    ablation: AblationResult | None = None,
) -> ScoredCase:
    return ScoredCase(
        test_id=test_id,
        response=RAGResponse(answer=answer, retrieved_chunk_ids=["chunk1"] if rank is not None else [], contexts=["a context"]),
        correctness=CorrectnessResult(
            is_correct=verdict in (Verdict.TRUE_PASS, Verdict.LUCKY_PASS),
            judge_model="j1",
            judge_parse_failed=judge_parse_failed,
        ),
        retrieval=RetrievalResult(
            hit=rank is not None, expected_chunk_id="chunk1", retrieved_chunk_ids=["chunk1"] if rank is not None else [], rank=rank
        ),
        groundedness=GroundednessResult(is_grounded=True, judge_model="j2"),
        verdict=verdict,
        ablation=ablation,
    )


def make_fixture(with_ablation: bool = True):
    test_set = TestSet(
        testset_id="ts1",
        corpus_hash="hash1",
        created_at=datetime.now(timezone.utc),
        generation_model="gen-model",
        test_cases=[
            make_test_case("t1", "What is the capital of Freedonia?"),
            make_test_case("t2", "Who invented the widget?"),
            make_test_case("t3", "When was the treaty signed?"),
            make_test_case("t4", "How many moons does Zorg have?"),
        ],
    )

    scored_cases = [
        make_scored_case("t1", Verdict.TRUE_PASS, rank=0),
        make_scored_case(
            "t2",
            Verdict.LUCKY_PASS,
            rank=None,
            ablation=(
                AblationResult(answer="no-context answer", is_correct=True, confirms_lucky_pass=True)
                if with_ablation
                else None
            ),
        ),
        make_scored_case("t3", Verdict.GENERATION_FAILURE, rank=0),
        make_scored_case("t4", Verdict.RETRIEVAL_FAILURE, rank=None, judge_parse_failed=True),
    ]

    summary = RunSummary(
        total_cases=4,
        verdict_counts={
            Verdict.TRUE_PASS: 1,
            Verdict.LUCKY_PASS: 1,
            Verdict.GENERATION_FAILURE: 1,
            Verdict.RETRIEVAL_FAILURE: 1,
        },
        retrieval_hit_rate=0.5,
        answer_correctness_rate=0.5,
        lucky_pass_rate=0.25,
        mean_groundedness=None,
        ablation_confirmed_count=1 if with_ablation else None,
        judge_failure_count=1,
    )

    run = Run(
        config=RunConfig(
            run_id="run1",
            testset_id="ts1",
            adapter_path="my_adapter.py",
            correctness_judge_model="j1",
            groundedness_judge_model="j2",
            run_ablation=with_ablation,
            started_at=datetime.now(timezone.utc),
        ),
        finished_at=datetime.now(timezone.utc),
        scored_cases=scored_cases,
        summary=summary,
    )

    return run, test_set


def test_report_contains_headline_counts_and_percentage(tmp_path):
    run, test_set = make_fixture()
    output_path = generate_report(run, test_set, tmp_path / "report.html")

    html = output_path.read_text(encoding="utf-8")
    assert "4 test questions" in html
    assert "1 (25.0%)" in html


def test_report_contains_all_four_verdict_labels(tmp_path):
    run, test_set = make_fixture()
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "True pass" in html
    assert "Lucky pass" in html
    assert "Generation failure" in html
    assert "Retrieval failure" in html


def test_report_contains_verdict_matrix_counts(tmp_path):
    run, test_set = make_fixture()
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    assert 'class="cell-true-pass">1' in html
    assert 'class="cell-lucky-pass">1' in html
    assert 'class="cell-generation-failure">1' in html
    assert 'class="cell-retrieval-failure">1' in html


def test_report_surfaces_ablation_confirmed_count(tmp_path):
    run, test_set = make_fixture(with_ablation=True)
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<strong>1</strong>" in html
    assert "stayed correct even with retrieval turned off" in html


def test_report_states_ablation_disabled_when_candidates_exist_but_not_ablated(tmp_path):
    run, test_set = make_fixture(with_ablation=False)
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Ablation was disabled for this run" in html
    assert "1 lucky-pass case(s) above" in html
    assert "unconfirmed candidates" in html


def test_report_states_ablation_skipped_when_no_lucky_pass_candidates(tmp_path):
    test_set = TestSet(
        testset_id="ts2",
        corpus_hash="hash2",
        created_at=datetime.now(timezone.utc),
        generation_model="gen-model",
        test_cases=[make_test_case("t1", "What is the capital of Freedonia?")],
    )
    scored_cases = [make_scored_case("t1", Verdict.TRUE_PASS, rank=0)]
    summary = RunSummary(
        total_cases=1,
        verdict_counts={
            Verdict.TRUE_PASS: 1,
            Verdict.LUCKY_PASS: 0,
            Verdict.GENERATION_FAILURE: 0,
            Verdict.RETRIEVAL_FAILURE: 0,
        },
        retrieval_hit_rate=1.0,
        answer_correctness_rate=1.0,
        lucky_pass_rate=0.0,
        mean_groundedness=None,
        ablation_confirmed_count=None,
        judge_failure_count=0,
    )
    run = Run(
        config=RunConfig(
            run_id="run2",
            testset_id="ts2",
            adapter_path="my_adapter.py",
            correctness_judge_model="j1",
            groundedness_judge_model="j2",
            run_ablation=True,
            started_at=datetime.now(timezone.utc),
        ),
        finished_at=datetime.now(timezone.utc),
        scored_cases=scored_cases,
        summary=summary,
    )

    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "ablation was skipped" in html
    assert "Ablation was disabled" not in html
    assert "unconfirmed candidates" not in html


def make_run_with_verdict_counts(counts: dict) -> tuple:
    """Build a minimal Run/TestSet pair with exactly the given verdict counts.

    `counts` maps Verdict -> number of cases with that verdict. Retrieval
    rank is set consistently with each verdict's retrieval-hit side of the
    2x2 (TRUE_PASS/GENERATION_FAILURE retrieved, LUCKY_PASS/RETRIEVAL_FAILURE not).
    """
    test_cases = []
    scored_cases = []
    i = 0
    for verdict, count in counts.items():
        for _ in range(count):
            test_id = f"t{i}"
            i += 1
            test_cases.append(make_test_case(test_id, f"question {test_id}"))
            rank = 0 if verdict in (Verdict.TRUE_PASS, Verdict.GENERATION_FAILURE) else None
            scored_cases.append(make_scored_case(test_id, verdict, rank=rank))

    total = sum(counts.values())
    full_counts = {v: counts.get(v, 0) for v in Verdict}
    test_set = TestSet(
        testset_id="ts-dominant",
        corpus_hash="hash-dominant",
        created_at=datetime.now(timezone.utc),
        generation_model="gen-model",
        test_cases=test_cases,
    )
    summary = RunSummary(
        total_cases=total,
        verdict_counts=full_counts,
        retrieval_hit_rate=0.0,
        answer_correctness_rate=0.0,
        lucky_pass_rate=0.0,
        mean_groundedness=None,
        ablation_confirmed_count=None,
        judge_failure_count=0,
    )
    run = Run(
        config=RunConfig(
            run_id="run-dominant",
            testset_id="ts-dominant",
            adapter_path="my_adapter.py",
            correctness_judge_model="j1",
            groundedness_judge_model="j2",
            run_ablation=True,
            started_at=datetime.now(timezone.utc),
        ),
        finished_at=datetime.now(timezone.utc),
        scored_cases=scored_cases,
        summary=summary,
    )
    return run, test_set


def test_headline_highlights_dominant_retrieval_failure_when_no_lucky_pass(tmp_path):
    run, test_set = make_run_with_verdict_counts(
        {Verdict.TRUE_PASS: 12, Verdict.RETRIEVAL_FAILURE: 6}
    )
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "dominant story in this run is retrieval failure" in html
    assert "6 of 18" in html
    assert "None of the correct answers" not in html


def test_headline_highlights_dominant_generation_failure_when_no_lucky_pass(tmp_path):
    run, test_set = make_run_with_verdict_counts(
        {Verdict.TRUE_PASS: 2, Verdict.GENERATION_FAILURE: 5, Verdict.RETRIEVAL_FAILURE: 3}
    )
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "dominant story in this run is generation failure" in html
    assert "5 of 10" in html


def test_headline_all_true_pass_when_no_lucky_pass_and_no_failures(tmp_path):
    run, test_set = make_run_with_verdict_counts({Verdict.TRUE_PASS: 4})
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "no lucky passes and no failures" in html


def test_report_surfaces_judge_failure_count_honestly(tmp_path):
    run, test_set = make_fixture()
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<strong>1</strong> of 4 case(s) had a judge response" in html


def test_report_surfaces_zero_judge_failures_explicitly(tmp_path):
    run, test_set = make_fixture()
    run.summary.judge_failure_count = 0
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "0 parsing failures" in html


def test_report_contains_questions_and_is_self_contained(tmp_path):
    run, test_set = make_fixture()
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "What is the capital of Freedonia?" in html
    assert "Who invented the widget?" in html
    # self-contained: stylesheet inlined, no external link/href to style.css
    assert "<style>" in html
    assert "background: var(--bg)" in html
    assert 'rel="stylesheet"' not in html


def test_report_contains_sort_and_filter_js_no_framework(tmp_path):
    run, test_set = make_fixture()
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "<script>" in html
    assert "data-filter" in html
    assert "data-sort" in html
    assert "cdn" not in html.lower()


def test_report_style_block_contains_real_css_rules(tmp_path):
    run, test_set = make_fixture()
    html = generate_report(run, test_set, tmp_path / "report.html").read_text(encoding="utf-8")

    match = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    assert match is not None, "expected a <style> block in the report"

    style_block = match.group(1)
    # Not just non-empty — actual CSS rule syntax: a selector followed by
    # at least one declaration inside braces, not e.g. a stray comment.
    assert len(style_block.strip()) > 500
    rule_pattern = re.compile(r"[.#a-zA-Z][^{}]*\{[^{}]*[a-zA-Z-]+\s*:\s*[^{}]+;[^{}]*\}")
    assert rule_pattern.search(style_block), "expected at least one real CSS rule (selector { prop: value; })"
    assert "--bg" in style_block
    assert "font-family" in style_block


def test_generate_report_raises_if_stylesheet_cannot_be_loaded(tmp_path):
    run, test_set = make_fixture()

    with mock.patch("ragaudit.report.generator.resources.files", side_effect=FileNotFoundError("missing")):
        try:
            generate_report(run, test_set, tmp_path / "report.html")
            assert False, "expected RuntimeError when the stylesheet can't be loaded"
        except RuntimeError as exc:
            assert "stylesheet" in str(exc).lower()


def test_report_raises_without_summary(tmp_path):
    run, test_set = make_fixture()
    run.summary = None
    try:
        generate_report(run, test_set, tmp_path / "report.html")
        assert False, "expected ValueError"
    except ValueError:
        pass
