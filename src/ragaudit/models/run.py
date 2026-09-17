"""Models for a full audit run and its aggregate summary."""

from datetime import datetime

from pydantic import BaseModel

from .scoring import ScoredCase, Verdict


class RunConfig(BaseModel):
    """Configuration a run was executed with."""

    run_id: str
    testset_id: str
    adapter_path: str
    correctness_judge_model: str
    groundedness_judge_model: str
    run_ablation: bool = True
    started_at: datetime


class RunSummary(BaseModel):
    """Aggregate stats computed after all cases are scored — the report's headline."""

    total_cases: int
    verdict_counts: dict[Verdict, int]
    retrieval_hit_rate: float
    answer_correctness_rate: float
    lucky_pass_rate: float
    """The headline metric."""
    mean_groundedness: float | None = None
    ablation_confirmed_count: int | None = None
    """Of LUCKY_PASS cases, how many ablation confirmed."""
    judge_failure_count: int = 0
    """Cases where the correctness judge, groundedness judge, or both failed to parse."""


class Run(BaseModel):
    """A full audit run: config, every scored case, and the aggregate summary."""

    config: RunConfig
    finished_at: datetime | None = None
    scored_cases: list[ScoredCase]
    summary: RunSummary | None = None
