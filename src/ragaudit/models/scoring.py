"""Per-case scoring results and the verdict that cross-references them."""

from enum import StrEnum

from pydantic import BaseModel

from .adapter import RAGResponse


class Verdict(StrEnum):
    """The cross-reference of answer correctness against retrieval hit."""

    TRUE_PASS = "true_pass"
    LUCKY_PASS = "lucky_pass"
    GENERATION_FAILURE = "generation_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"


class CorrectnessResult(BaseModel):
    """LLM-judged comparison of RAGResponse.answer against TestCase.expected_answer."""

    is_correct: bool
    score: float | None = None
    """Raw judge score in [0, 1], if the judge emits one."""
    rationale: str | None = None
    judge_model: str


class RetrievalResult(BaseModel):
    """Pure set-membership check — no LLM involved."""

    hit: bool
    expected_chunk_id: str
    retrieved_chunk_ids: list[str]
    rank: int | None = None
    """Index of expected_chunk_id in retrieved_chunk_ids, if present."""


class GroundednessResult(BaseModel):
    """Whether the answer is supported by the contexts the adapter actually returned."""

    is_grounded: bool
    score: float | None = None
    rationale: str | None = None
    judge_model: str


class AblationResult(BaseModel):
    """Re-run with retrieval disabled, to empirically confirm a candidate LUCKY_PASS.

    Only populated for cases that scored correct + not-retrieved.
    """

    answer: str
    is_correct: bool
    confirms_lucky_pass: bool
    """True iff the answer is still correct with no retrieval."""


class ScoredCase(BaseModel):
    """One TestCase run through a pipeline and fully scored."""

    test_id: str
    response: RAGResponse
    correctness: CorrectnessResult
    retrieval: RetrievalResult
    groundedness: GroundednessResult
    verdict: Verdict
    ablation: AblationResult | None = None
