"""Run history: one summary row per `check`, `run`, `evaluate`, `tune` and `improve`.

Rows live in the project's store (`.ragaudit/store.db` next to the docs
folder for `check` and `improve`), so each project keeps its own history.

Fair comparisons: every row keeps the verdict of each question it scored,
keyed by a stable question id. The delta against the previous run of the
same kind is computed only on the questions both runs share, and the
number of shared questions is stored with it, so a change in the question
set can never pass for a change in quality.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from ragaudit.storage.store import Store

KINDS = ("check", "run", "evaluate", "tune", "improve")
SMALL_SHARED_SAMPLE = 10
"""Below this many shared questions the report warns that a delta is noisy."""
TRUE_PASS = "true_pass"


def store_path_for_docs(docs: Path) -> Path:
    """The project store for a docs folder: `.ragaudit/store.db` next to it."""
    return docs.resolve().parent / ".ragaudit" / "store.db"


def question_set_id(question_keys) -> str:
    """Identity of a question set: a hash of its sorted question ids."""
    return hashlib.sha256("\n".join(sorted(question_keys)).encode("utf-8")).hexdigest()


class HistoryRow(BaseModel):
    run_id: str
    kind: str
    created_at: datetime
    adapter_path: str = ""
    config_hash: str | None = None
    corpus_fingerprint: str | None = None
    question_set_id: str | None = None
    n_questions: int = 0
    true_pass_pct: float = 0.0
    retrieval_hit_pct: float = 0.0
    correctness_pct: float = 0.0
    groundedness_pct: float | None = None
    lucky_pass_count: int = 0
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    note: str | None = None
    corpus_changed: bool = False
    outcomes: dict[str, str] = Field(default_factory=dict)
    """question id -> verdict, for shared-question deltas."""
    prev_run_id: str | None = None
    shared_questions: int = 0
    delta_true_pass: float | None = None
    """True-pass points vs the previous run of the same kind, on shared questions only. None if nothing is shared."""
    delta_retrieval_hit: float | None = None
    extra: dict = Field(default_factory=dict)
    """Kind-specific details (tune: baseline/best scores, trials, config changes)."""

    @property
    def small_sample(self) -> bool:
        return self.prev_run_id is not None and self.shared_questions < SMALL_SHARED_SAMPLE


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 1) if total else 0.0


def _shared_delta(current: dict[str, str], previous: dict[str, str], hits_now: dict[str, bool], hits_before: dict[str, bool]):
    shared = sorted(set(current) & set(previous))
    if not shared:
        return 0, None, None
    tp_now = sum(current[k] == TRUE_PASS for k in shared)
    tp_before = sum(previous[k] == TRUE_PASS for k in shared)
    delta_tp = round(_pct(tp_now, len(shared)) - _pct(tp_before, len(shared)), 1)
    delta_hit = None
    if all(k in hits_now and k in hits_before for k in shared):
        delta_hit = round(
            _pct(sum(hits_now[k] for k in shared), len(shared)) - _pct(sum(hits_before[k] for k in shared), len(shared)), 1
        )
    return len(shared), delta_tp, delta_hit


_HIT_VERDICTS = {"true_pass", "generation_failure"}


def _hits(outcomes: dict[str, str]) -> dict[str, bool]:
    return {key: verdict in _HIT_VERDICTS for key, verdict in outcomes.items()}


def load_history(store: Store) -> list[HistoryRow]:
    """Every recorded run of this project, oldest first, with its stored deltas."""
    return [HistoryRow.model_validate_json(raw) for raw in store.load_history_rows()]


def previous_run(rows: list[HistoryRow], kind: str) -> HistoryRow | None:
    same = [row for row in rows if row.kind == kind]
    return same[-1] if same else None


def record_run(
    store: Store,
    *,
    kind: str,
    run_id: str,
    outcomes: dict[str, str],
    adapter_path: str = "",
    config_hash: str | None = None,
    corpus_fingerprint: str | None = None,
    correct_count: int | None = None,
    groundedness_pct: float | None = None,
    models: dict[str, str] | None = None,
    note: str | None = None,
    corpus_changed: bool | None = None,
    extra: dict | None = None,
    created_at: datetime | None = None,
    retrieval_hits: int | None = None,
    true_pass_pct: float | None = None,
    retrieval_hit_pct: float | None = None,
    correctness_pct: float | None = None,
    n_questions: int | None = None,
) -> HistoryRow:
    """Build, save and return one history row. Scores come from `outcomes` unless given explicitly.

    `corpus_changed` defaults to "the corpus fingerprint differs from the
    previous run of the same kind".
    """
    if kind not in KINDS:
        raise ValueError(f"unknown history kind {kind!r}")
    rows = load_history(store)
    previous = previous_run(rows, kind)
    total = len(outcomes) if n_questions is None else n_questions
    counts = {verdict: 0 for verdict in ("true_pass", "lucky_pass", "generation_failure", "retrieval_failure")}
    for verdict in outcomes.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    hits = retrieval_hits if retrieval_hits is not None else counts["true_pass"] + counts["generation_failure"]
    correct = correct_count if correct_count is not None else counts["true_pass"] + counts["lucky_pass"]
    if corpus_changed is None:
        corpus_changed = bool(
            previous is not None
            and corpus_fingerprint
            and previous.corpus_fingerprint
            and previous.corpus_fingerprint != corpus_fingerprint
        )

    shared, delta_tp, delta_hit = 0, None, None
    if previous is not None:
        shared, delta_tp, delta_hit = _shared_delta(outcomes, previous.outcomes, _hits(outcomes), _hits(previous.outcomes))

    row = HistoryRow(
        run_id=run_id,
        kind=kind,
        created_at=created_at or datetime.now(timezone.utc),
        adapter_path=adapter_path,
        config_hash=config_hash,
        corpus_fingerprint=corpus_fingerprint,
        question_set_id=question_set_id(outcomes) if outcomes else None,
        n_questions=total,
        true_pass_pct=true_pass_pct if true_pass_pct is not None else _pct(counts["true_pass"], total),
        retrieval_hit_pct=retrieval_hit_pct if retrieval_hit_pct is not None else _pct(hits, total),
        correctness_pct=correctness_pct if correctness_pct is not None else _pct(correct, total),
        groundedness_pct=groundedness_pct,
        lucky_pass_count=counts["lucky_pass"],
        verdict_counts=counts,
        models=models or {},
        note=note,
        corpus_changed=corpus_changed,
        outcomes=outcomes,
        prev_run_id=previous.run_id if previous else None,
        shared_questions=shared,
        delta_true_pass=delta_tp,
        delta_retrieval_hit=delta_hit,
        extra=extra or {},
    )
    store.save_history_row(row.run_id, row.kind, row.created_at, row.model_dump_json())
    return row


def history_for_report(rows: list[HistoryRow]) -> list[dict]:
    """Plain dicts for the report's chart and run buttons (no per-question outcomes, which can be large)."""
    out = []
    for index, row in enumerate(rows, start=1):
        data = row.model_dump(mode="json", exclude={"outcomes"})
        data["number"] = index
        data["small_sample"] = row.small_sample
        out.append(data)
    return out


def dumps_for_html(data) -> str:
    """JSON safe to embed in a <script> block."""
    return json.dumps(data, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def tune_extra(result) -> dict:
    """The improvement-loop details the report shows, from a finished TuneResult (aggregates only)."""
    baseline, best = result.baseline, result.best
    changes = [
        {"knob": knob, "before": baseline.config.get(knob), "after": best.config.get(knob)}
        for knob in best.config
        if baseline.config.get(knob) != best.config.get(knob)
    ]
    return {
        "tune_id": result.tune_id,
        "strategy": result.strategy,
        "baseline_exam": result.baseline_test.score if result.baseline_test else None,
        "best_exam": result.best_test.score if result.best_test else None,
        "baseline_practice": baseline.dev.score,
        "best_practice": best.dev.score,
        "overfit_gap": result.overfit_gap,
        "overfit_threshold": result.overfit_threshold,
        "overfit_warning": result.overfit_warning,
        "trials": [{"index": t.index, "practice": t.dev.score, "source": t.source, "config": t.config} for t in result.trials],
        "best_index": result.best_index,
        "changes": changes,
        "baseline_config": baseline.config,
        "best_config": best.config,
        "best_config_path": result.best_config_path,
        "stopped_reason": result.stopped_reason,
        "n_practice": baseline.dev.n_questions,
        "n_exam": result.best_test.n_questions if result.best_test else 0,
        "llm_calls": result.llm_calls,
    }


def record_eval_summary(store: Store, summary, *, kind: str, outcomes: dict[str, str], adapter_path: str,
                        note: str | None, models: dict[str, str], extra: dict | None = None, run_id: str | None = None,
                        corpus_changed: bool | None = None) -> HistoryRow:
    """A history row from a v2 EvalSummary (evaluate) or a tune's best test summary (tune, improve)."""
    n = summary.n_questions
    hits = summary.failure_counts.get("pass", 0) + summary.failure_counts.get("generation_miss", 0)
    return record_run(
        store,
        kind=kind,
        run_id=run_id or summary.run_id,
        outcomes=outcomes,
        adapter_path=adapter_path,
        config_hash=summary.config_hash,
        corpus_fingerprint=summary.index_version,
        n_questions=n,
        true_pass_pct=round(summary.score, 1),
        retrieval_hit_pct=_pct(hits, n),
        correctness_pct=round(summary.answer_accuracy * 100, 1),
        models=models,
        note=note,
        extra=extra,
        corpus_changed=corpus_changed,
    )
