"""v2 tuning: one trial per evaluated config, and the run's outcome."""

from datetime import datetime

from pydantic import BaseModel, Field

from .evaluation import EvalSummary


class Trial(BaseModel):
    """One pipeline config evaluated on the dev split."""

    index: int
    config: dict
    config_hash: str
    source: str
    """baseline, grid, random, llm, or llm_fallback (the LLM's proposal was invalid; a random one was used)."""
    rationale: str | None = None
    dev: EvalSummary
    created_at: datetime


class TuneResult(BaseModel):
    tune_id: str
    created_at: datetime
    finished_at: datetime | None = None
    strategy: str
    search_space: dict[str, list]
    adapter_path: str
    judge_model: str
    index_version: str
    k: int
    seed: int
    max_trials: int
    trials: list[Trial] = Field(default_factory=list)
    baseline_index: int = 0
    best_index: int = 0
    baseline_test: EvalSummary | None = None
    """Aggregate only: tune never sees test questions or answers."""
    best_test: EvalSummary | None = None
    overfit_threshold: float = 10.0
    overfit_gap: float | None = None
    """Best config's dev score minus its test score, in points."""
    overfit_warning: bool = False
    stopped_reason: str
    """max_trials, search_space_exhausted, or max_llm_calls."""
    llm_calls: int = 0
    best_config_path: str | None = None

    @property
    def baseline(self) -> Trial:
        return self.trials[self.baseline_index]

    @property
    def best(self) -> Trial:
        return self.trials[self.best_index]
