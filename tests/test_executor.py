from datetime import datetime, timezone

from ragaudit.models.adapter import RAGResponse
from ragaudit.models.scoring import CorrectnessResult, GroundednessResult, Verdict
from ragaudit.models.testset import TestCase, TestSet
from ragaudit.runner.executor import run_test_set


def make_test_case(test_id: str, source_chunk_id: str = "chunk1") -> TestCase:
    return TestCase(
        test_id=test_id,
        question=f"question {test_id}",
        expected_answer="expected",
        source_chunk_id=source_chunk_id,
        source_doc_id="doc1",
        generation_model="gen-model",
        created_at=datetime.now(timezone.utc),
    )


def make_test_set(test_cases: list[TestCase]) -> TestSet:
    return TestSet(
        testset_id="ts1",
        corpus_hash="hash1",
        created_at=datetime.now(timezone.utc),
        generation_model="gen-model",
        test_cases=test_cases,
    )


class ScriptedAdapter:
    """Returns queued responses, or raises if the queued item is an exception."""

    def __init__(self, responses: list):
        self._responses = list(responses)

    def query(self, question: str) -> RAGResponse:
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def always_correct(question: str, expected_answer: str, candidate_answer: str) -> CorrectnessResult:
    return CorrectnessResult(is_correct=True, judge_model="stub-judge")


def always_wrong(question: str, expected_answer: str, candidate_answer: str) -> CorrectnessResult:
    return CorrectnessResult(is_correct=False, judge_model="stub-judge")


def always_grounded(answer: str, contexts: list[str]) -> GroundednessResult:
    return GroundednessResult(is_grounded=True, judge_model="stub-judge")


def test_true_pass_when_correct_and_retrieved():
    test_case = make_test_case("t1", source_chunk_id="chunk1")
    test_set = make_test_set([test_case])
    adapter = ScriptedAdapter([RAGResponse(answer="ans", retrieved_chunk_ids=["chunk1"], contexts=["ctx"])])

    scored = run_test_set(test_set, adapter, always_correct, always_grounded)

    assert len(scored) == 1
    assert scored[0].verdict == Verdict.TRUE_PASS
    assert scored[0].retrieval.hit is True


def test_lucky_pass_when_correct_and_not_retrieved():
    test_case = make_test_case("t1", source_chunk_id="chunk1")
    test_set = make_test_set([test_case])
    adapter = ScriptedAdapter([RAGResponse(answer="ans", retrieved_chunk_ids=["other"], contexts=[])])

    scored = run_test_set(test_set, adapter, always_correct, always_grounded)

    assert scored[0].verdict == Verdict.LUCKY_PASS


def test_generation_failure_when_wrong_and_retrieved():
    test_case = make_test_case("t1", source_chunk_id="chunk1")
    test_set = make_test_set([test_case])
    adapter = ScriptedAdapter([RAGResponse(answer="ans", retrieved_chunk_ids=["chunk1"], contexts=["ctx"])])

    scored = run_test_set(test_set, adapter, always_wrong, always_grounded)

    assert scored[0].verdict == Verdict.GENERATION_FAILURE


def test_retrieval_failure_when_wrong_and_not_retrieved():
    test_case = make_test_case("t1", source_chunk_id="chunk1")
    test_set = make_test_set([test_case])
    adapter = ScriptedAdapter([RAGResponse(answer="ans", retrieved_chunk_ids=[], contexts=[])])

    scored = run_test_set(test_set, adapter, always_wrong, always_grounded)

    assert scored[0].verdict == Verdict.RETRIEVAL_FAILURE


def test_adapter_exception_does_not_kill_the_run():
    test_cases = [make_test_case("t1"), make_test_case("t2"), make_test_case("t3")]
    test_set = make_test_set(test_cases)
    adapter = ScriptedAdapter(
        [
            RAGResponse(answer="ans1", retrieved_chunk_ids=["chunk1"], contexts=["ctx"]),
            RuntimeError("adapter blew up on t2"),
            RAGResponse(answer="ans3", retrieved_chunk_ids=["chunk1"], contexts=["ctx"]),
        ]
    )

    scored = run_test_set(test_set, adapter, always_correct, always_grounded)

    assert len(scored) == 3
    assert scored[0].verdict == Verdict.TRUE_PASS
    failed_case = scored[1]
    assert failed_case.test_id == "t2"
    assert failed_case.correctness.is_correct is False
    assert failed_case.verdict == Verdict.RETRIEVAL_FAILURE
    assert "adapter raised" in failed_case.correctness.rationale
    assert scored[2].verdict == Verdict.TRUE_PASS
