"""Pure function cross-referencing answer correctness against retrieval."""

from ragaudit.models.scoring import Verdict


def compute_verdict(is_correct: bool, retrieval_hit: bool) -> Verdict:
    """Cross-reference correctness against retrieval into a single Verdict."""
    if is_correct and retrieval_hit:
        return Verdict.TRUE_PASS
    if is_correct and not retrieval_hit:
        return Verdict.LUCKY_PASS
    if not is_correct and retrieval_hit:
        return Verdict.GENERATION_FAILURE
    return Verdict.RETRIEVAL_FAILURE
