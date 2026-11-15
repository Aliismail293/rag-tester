"""Runs a TestSet through an Adapter and scores every case."""

import logging
from typing import Callable

from ragaudit.models.adapter import Adapter, RAGResponse
from ragaudit.models.scoring import CorrectnessResult, GroundednessResult, RetrievalResult, ScoredCase
from ragaudit.models.testset import TestCase, TestSet
from ragaudit.scoring.retrieval import score_retrieval
from ragaudit.scoring.verdict import compute_verdict

logger = logging.getLogger(__name__)

CorrectnessJudge = Callable[[str, str, str], CorrectnessResult]
"""question, expected_answer, candidate_answer -> CorrectnessResult"""

GroundednessJudge = Callable[[str, list[str]], GroundednessResult]
"""answer, contexts -> GroundednessResult"""

RetrievalScorer = Callable[[TestCase, RAGResponse], RetrievalResult]
"""test case, adapter response -> RetrievalResult. Defaults to matching by chunk id."""

ADAPTER_ERROR_JUDGE_MODEL = "n/a (adapter raised)"


def score_by_id(test_case: TestCase, response: RAGResponse) -> RetrievalResult:
    return score_retrieval(test_case.source_chunk_id, response.retrieved_chunk_ids)


def _failure_scored_case(test_case: TestCase, error: Exception, retrieval_scorer: RetrievalScorer) -> ScoredCase:
    """Build a ScoredCase for a TestCase whose adapter.query() call raised."""
    empty_response = RAGResponse(answer="", retrieved_chunk_ids=[], contexts=[], raw={"error": str(error)})
    retrieval = retrieval_scorer(test_case, empty_response)
    correctness = CorrectnessResult(
        is_correct=False,
        rationale=f"adapter raised: {error}",
        judge_model=ADAPTER_ERROR_JUDGE_MODEL,
    )
    groundedness = GroundednessResult(
        is_grounded=False,
        rationale="adapter raised, no answer to judge",
        judge_model=ADAPTER_ERROR_JUDGE_MODEL,
    )
    verdict = compute_verdict(is_correct=False, retrieval_hit=retrieval.hit)
    return ScoredCase(
        test_id=test_case.test_id,
        response=empty_response,
        correctness=correctness,
        retrieval=retrieval,
        groundedness=groundedness,
        verdict=verdict,
    )


def run_test_set(
    test_set: TestSet,
    adapter: Adapter,
    correctness_judge: CorrectnessJudge,
    groundedness_judge: GroundednessJudge,
    retrieval_scorer: RetrievalScorer = score_by_id,
) -> list[ScoredCase]:
    """Run every TestCase in test_set through adapter and score it.

    retrieval_scorer decides retrieval hits: by chunk id (the default) or,
    for `--match-by text`, by overlap of the returned contexts with the gold
    chunk's text.

    A TestCase whose adapter.query() call raises is recorded as a failed
    ScoredCase (correctness/groundedness both False) instead of aborting
    the run — one broken question must not kill a 500-case run.
    """
    scored_cases: list[ScoredCase] = []

    for test_case in test_set.test_cases:
        try:
            response = adapter.query(test_case.question)
        except Exception as exc:
            logger.warning("Adapter raised for test_id=%s: %s", test_case.test_id, exc)
            scored_cases.append(_failure_scored_case(test_case, exc, retrieval_scorer))
            continue

        retrieval = retrieval_scorer(test_case, response)
        correctness = correctness_judge(test_case.question, test_case.expected_answer, response.answer)
        groundedness = groundedness_judge(response.answer, response.contexts)
        verdict = compute_verdict(is_correct=correctness.is_correct, retrieval_hit=retrieval.hit)

        scored_cases.append(
            ScoredCase(
                test_id=test_case.test_id,
                response=response,
                correctness=correctness,
                retrieval=retrieval,
                groundedness=groundedness,
                verdict=verdict,
            )
        )

    return scored_cases
