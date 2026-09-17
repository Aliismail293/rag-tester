from datetime import datetime, timezone

from ragaudit.models.adapter import RAGResponse
from ragaudit.models.scoring import CorrectnessResult, GroundednessResult, RetrievalResult, ScoredCase, Verdict
from ragaudit.models.testset import TestCase
from ragaudit.runner.ablation import run_ablation


def make_test_case(test_id: str) -> TestCase:
    return TestCase(
        test_id=test_id,
        question=f"question {test_id}",
        expected_answer="expected answer",
        source_chunk_id="chunk1",
        source_doc_id="doc1",
        generation_model="gen-model",
        created_at=datetime.now(timezone.utc),
    )


def make_scored_case(test_id: str, verdict: Verdict) -> ScoredCase:
    return ScoredCase(
        test_id=test_id,
        response=RAGResponse(answer="ans", retrieved_chunk_ids=[], contexts=[]),
        correctness=CorrectnessResult(is_correct=verdict in (Verdict.TRUE_PASS, Verdict.LUCKY_PASS), judge_model="j"),
        retrieval=RetrievalResult(hit=verdict in (Verdict.TRUE_PASS, Verdict.GENERATION_FAILURE), expected_chunk_id="chunk1", retrieved_chunk_ids=[]),
        groundedness=GroundednessResult(is_grounded=True, judge_model="j"),
        verdict=verdict,
    )


def test_ablation_confirms_lucky_pass_when_still_correct_without_context():
    test_cases = [make_test_case("t1")]
    scored_cases = [make_scored_case("t1", Verdict.LUCKY_PASS)]

    result = run_ablation(
        scored_cases,
        test_cases,
        no_context_query=lambda q: "still correct answer",
        check_correctness=lambda q, e, a: True,
    )

    assert result[0].ablation is not None
    assert result[0].ablation.confirms_lucky_pass is True
    assert result[0].ablation.answer == "still correct answer"


def test_ablation_does_not_confirm_when_wrong_without_context():
    test_cases = [make_test_case("t1")]
    scored_cases = [make_scored_case("t1", Verdict.LUCKY_PASS)]

    result = run_ablation(
        scored_cases,
        test_cases,
        no_context_query=lambda q: "wrong guess",
        check_correctness=lambda q, e, a: False,
    )

    assert result[0].ablation is not None
    assert result[0].ablation.confirms_lucky_pass is False


def test_ablation_only_touches_lucky_pass_rows():
    test_cases = [make_test_case("t1"), make_test_case("t2"), make_test_case("t3"), make_test_case("t4")]
    scored_cases = [
        make_scored_case("t1", Verdict.TRUE_PASS),
        make_scored_case("t2", Verdict.LUCKY_PASS),
        make_scored_case("t3", Verdict.GENERATION_FAILURE),
        make_scored_case("t4", Verdict.RETRIEVAL_FAILURE),
    ]
    calls: list[str] = []

    def no_context_query(question: str) -> str:
        calls.append(question)
        return "answer"

    result = run_ablation(
        scored_cases,
        test_cases,
        no_context_query=no_context_query,
        check_correctness=lambda q, e, a: True,
    )

    assert calls == ["question t2"]
    assert result[0].ablation is None
    assert result[1].ablation is not None
    assert result[2].ablation is None
    assert result[3].ablation is None


def test_ablation_returns_same_number_of_cases_in_order():
    test_cases = [make_test_case("t1"), make_test_case("t2")]
    scored_cases = [make_scored_case("t1", Verdict.LUCKY_PASS), make_scored_case("t2", Verdict.TRUE_PASS)]

    result = run_ablation(scored_cases, test_cases, lambda q: "a", lambda q, e, a: True)

    assert [c.test_id for c in result] == ["t1", "t2"]
