from ragaudit.models.scoring import Verdict
from ragaudit.scoring.verdict import compute_verdict


def test_correct_and_retrieved_is_true_pass():
    assert compute_verdict(is_correct=True, retrieval_hit=True) == Verdict.TRUE_PASS


def test_correct_and_not_retrieved_is_lucky_pass():
    assert compute_verdict(is_correct=True, retrieval_hit=False) == Verdict.LUCKY_PASS


def test_wrong_and_retrieved_is_generation_failure():
    assert compute_verdict(is_correct=False, retrieval_hit=True) == Verdict.GENERATION_FAILURE


def test_wrong_and_not_retrieved_is_retrieval_failure():
    assert compute_verdict(is_correct=False, retrieval_hit=False) == Verdict.RETRIEVAL_FAILURE
