"""Re-runs LUCKY_PASS candidates with no context to empirically confirm them."""

from typing import Callable

from ragaudit.models.scoring import AblationResult, ScoredCase, Verdict
from ragaudit.models.testset import TestCase

NoContextQuery = Callable[[str], str]
"""question -> answer, produced with retrieval/context disabled entirely."""

CorrectnessCheck = Callable[[str, str, str], bool]
"""question, expected_answer, candidate_answer -> is_correct"""


def run_ablation(
    scored_cases: list[ScoredCase],
    test_cases: list[TestCase],
    no_context_query: NoContextQuery,
    check_correctness: CorrectnessCheck,
) -> list[ScoredCase]:
    """Populate AblationResult for LUCKY_PASS rows only.

    For every ScoredCase whose verdict is LUCKY_PASS, re-asks the question
    with no context at all. If the answer is still correct, retrieval was
    contributing nothing — confirms_lucky_pass is True. All other verdicts
    are returned unchanged (ablation stays None).
    """
    test_case_by_id = {tc.test_id: tc for tc in test_cases}
    updated: list[ScoredCase] = []

    for scored_case in scored_cases:
        if scored_case.verdict != Verdict.LUCKY_PASS:
            updated.append(scored_case)
            continue

        test_case = test_case_by_id[scored_case.test_id]
        no_context_answer = no_context_query(test_case.question)
        is_correct = check_correctness(test_case.question, test_case.expected_answer, no_context_answer)
        ablation = AblationResult(
            answer=no_context_answer,
            is_correct=is_correct,
            confirms_lucky_pass=is_correct,
        )
        updated.append(scored_case.model_copy(update={"ablation": ablation}))

    return updated
