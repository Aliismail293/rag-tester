"""v2 per-question evaluation results and their aggregate."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from .scoring import AblationResult, CorrectnessResult, Verdict


class FailureClass(StrEnum):
    PASS = "pass"
    """A gold chunk was retrieved in the top k and the answer is correct."""
    RETRIEVAL_MISS = "retrieval_miss"
    """Gold chunk(s) not retrieved in the top k, whether or not the answer happened to be right."""
    GENERATION_MISS = "generation_miss"
    """Gold chunk(s) retrieved, answer still wrong."""


class QuestionResult(BaseModel):
    """One question run through a pipeline config and scored. Never contains the gold answer."""

    question_id: str
    split: str
    answer: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    recall_at_k: float
    """Share of the gold chunk_ids found in the top k (0 or 1 for single-chunk questions)."""
    reciprocal_rank: float
    """1 / (1 + rank of the first gold chunk in the top k), or 0 if none was retrieved."""
    retrieval_hit: bool
    """Every gold chunk was retrieved in the top k."""
    correctness: CorrectnessResult
    failure_class: FailureClass
    verdict: Verdict
    """v1's retrieval-contribution signal: a correct answer without retrieval is a LUCKY_PASS."""
    ablation: AblationResult | None = None
    adapter_error: str | None = None
    latency_ms: float | None = None


class EvalSummary(BaseModel):
    """Aggregate of one evaluation: the headline score is the pass rate, in points (0-100)."""

    run_id: str
    created_at: datetime
    split: str
    config_hash: str
    config: dict = Field(default_factory=dict)
    index_version: str
    k: int
    mode: str
    """full (config never evaluated), regression (index changed: new + sampled old), or cached (index unchanged)."""
    n_questions: int
    n_active_in_split: int
    n_new: int = 0
    n_sampled_old: int = 0
    n_cached: int = 0
    n_run: int = 0
    score: float
    """Pass rate x 100: retrieval hit AND correct answer. Lucky passes don't count."""
    answer_accuracy: float
    recall_at_k: float
    mrr: float
    failure_counts: dict[str, int]
    verdict_counts: dict[str, int]
    judge_failures: int = 0
    adapter_errors: int = 0
    lucky_pass_confirmed: int | None = None
    """Of LUCKY_PASS results, how many ablation confirmed (None if ablation didn't run)."""
