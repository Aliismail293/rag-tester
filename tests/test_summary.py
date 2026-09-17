from ragaudit.models.adapter import RAGResponse
from ragaudit.models.scoring import (
    AblationResult,
    CorrectnessResult,
    GroundednessResult,
    RetrievalResult,
    ScoredCase,
    Verdict,
)
from ragaudit.runner.summary import compute_run_summary


def make_scored_case(
    test_id: str,
    verdict: Verdict,
    groundedness_score: float | None = None,
    ablation: AblationResult | None = None,
    correctness_judge_parse_failed: bool = False,
    groundedness_judge_parse_failed: bool = False,
) -> ScoredCase:
    is_correct = verdict in (Verdict.TRUE_PASS, Verdict.LUCKY_PASS)
    hit = verdict in (Verdict.TRUE_PASS, Verdict.GENERATION_FAILURE)
    return ScoredCase(
        test_id=test_id,
        response=RAGResponse(answer="ans", retrieved_chunk_ids=[], contexts=[]),
        correctness=CorrectnessResult(
            is_correct=is_correct, judge_model="j", judge_parse_failed=correctness_judge_parse_failed
        ),
        retrieval=RetrievalResult(hit=hit, expected_chunk_id="c1", retrieved_chunk_ids=[]),
        groundedness=GroundednessResult(
            is_grounded=True,
            score=groundedness_score,
            judge_model="j",
            judge_parse_failed=groundedness_judge_parse_failed,
        ),
        verdict=verdict,
        ablation=ablation,
    )


def test_empty_scored_cases_yields_zeroed_summary():
    summary = compute_run_summary([])
    assert summary.total_cases == 0
    assert summary.retrieval_hit_rate == 0.0
    assert summary.answer_correctness_rate == 0.0
    assert summary.lucky_pass_rate == 0.0
    assert summary.mean_groundedness is None
    assert summary.ablation_confirmed_count is None
    assert summary.judge_failure_count == 0
    assert all(count == 0 for count in summary.verdict_counts.values())


def test_verdict_counts_and_rates():
    cases = [
        make_scored_case("t1", Verdict.TRUE_PASS),
        make_scored_case("t2", Verdict.TRUE_PASS),
        make_scored_case("t3", Verdict.LUCKY_PASS),
        make_scored_case("t4", Verdict.GENERATION_FAILURE),
        make_scored_case("t5", Verdict.RETRIEVAL_FAILURE),
    ]

    summary = compute_run_summary(cases)

    assert summary.total_cases == 5
    assert summary.verdict_counts[Verdict.TRUE_PASS] == 2
    assert summary.verdict_counts[Verdict.LUCKY_PASS] == 1
    assert summary.verdict_counts[Verdict.GENERATION_FAILURE] == 1
    assert summary.verdict_counts[Verdict.RETRIEVAL_FAILURE] == 1

    # retrieval hits: TRUE_PASS (x2) + GENERATION_FAILURE (x1) = 3 of 5
    assert summary.retrieval_hit_rate == 3 / 5
    # correct: TRUE_PASS (x2) + LUCKY_PASS (x1) = 3 of 5
    assert summary.answer_correctness_rate == 3 / 5
    assert summary.lucky_pass_rate == 1 / 5


def test_mean_groundedness_ignores_missing_scores():
    cases = [
        make_scored_case("t1", Verdict.TRUE_PASS, groundedness_score=1.0),
        make_scored_case("t2", Verdict.TRUE_PASS, groundedness_score=0.5),
        make_scored_case("t3", Verdict.TRUE_PASS, groundedness_score=None),
    ]

    summary = compute_run_summary(cases)

    assert summary.mean_groundedness == (1.0 + 0.5) / 2


def test_ablation_confirmed_count_only_counts_lucky_pass_confirmations():
    confirmed = AblationResult(answer="a", is_correct=True, confirms_lucky_pass=True)
    not_confirmed = AblationResult(answer="b", is_correct=False, confirms_lucky_pass=False)

    cases = [
        make_scored_case("t1", Verdict.LUCKY_PASS, ablation=confirmed),
        make_scored_case("t2", Verdict.LUCKY_PASS, ablation=not_confirmed),
        make_scored_case("t3", Verdict.LUCKY_PASS, ablation=None),
        make_scored_case("t4", Verdict.TRUE_PASS, ablation=None),
    ]

    summary = compute_run_summary(cases)

    assert summary.ablation_confirmed_count == 1


def test_ablation_confirmed_count_is_none_when_no_lucky_pass_cases():
    cases = [make_scored_case("t1", Verdict.TRUE_PASS), make_scored_case("t2", Verdict.RETRIEVAL_FAILURE)]
    summary = compute_run_summary(cases)
    assert summary.ablation_confirmed_count is None


def test_judge_failure_count_counts_correctness_failures():
    cases = [
        make_scored_case("t1", Verdict.TRUE_PASS, correctness_judge_parse_failed=True),
        make_scored_case("t2", Verdict.TRUE_PASS),
    ]
    summary = compute_run_summary(cases)
    assert summary.judge_failure_count == 1


def test_judge_failure_count_counts_groundedness_failures():
    cases = [
        make_scored_case("t1", Verdict.TRUE_PASS, groundedness_judge_parse_failed=True),
        make_scored_case("t2", Verdict.TRUE_PASS),
    ]
    summary = compute_run_summary(cases)
    assert summary.judge_failure_count == 1


def test_judge_failure_count_counts_a_case_once_even_if_both_judges_failed():
    cases = [
        make_scored_case(
            "t1", Verdict.TRUE_PASS, correctness_judge_parse_failed=True, groundedness_judge_parse_failed=True
        ),
        make_scored_case("t2", Verdict.TRUE_PASS),
    ]
    summary = compute_run_summary(cases)
    assert summary.judge_failure_count == 1


def test_judge_failure_count_is_zero_when_no_failures():
    cases = [make_scored_case("t1", Verdict.TRUE_PASS), make_scored_case("t2", Verdict.LUCKY_PASS)]
    summary = compute_run_summary(cases)
    assert summary.judge_failure_count == 0
