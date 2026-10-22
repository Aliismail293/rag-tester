"""v2 evaluation: run questions through a configured pipeline, score them, cache the results.

Per question: recall@k and reciprocal rank against the gold chunk_ids,
answer correctness from v1's LLM judge, a failure class (pass /
retrieval_miss / generation_miss), and v1's verdict (a correct answer
without the gold chunk is a LUCKY_PASS).

Results are cached under (question_id, config_hash, index_version), so a
question is never re-run for the same pipeline config on the same corpus.
Which questions run is decided by `select_questions`:

- the config was never evaluated on this split: run everything (`full`);
- the corpus (index_version) changed but the config didn't: run all new
  questions plus a seeded random sample of N old ones (`regression`),
  because new data mostly breaks old questions by adding distractors, and
  a sample catches that without re-running thousands of questions;
- nothing changed: every result comes from the cache, and any question
  added since is run (`cached`).

This module never reads the test split itself; test-split evaluation goes
through ragaudit.runner.locked, which unlocks it and calls in here.
"""

import hashlib
import json
import logging
import random
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from ragaudit.models.adapter import Adapter, RAGResponse
from ragaudit.models.evaluation import EvalSummary, FailureClass, QuestionResult
from ragaudit.models.question import Question, Split, utc_now
from ragaudit.models.scoring import AblationResult, CorrectnessResult, Verdict
from ragaudit.scoring.verdict import compute_verdict
from ragaudit.storage.store import Store

logger = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_REGRESSION_SAMPLE = 50
SCORING_VERSION = 1
"""Bump when scoring changes meaningfully, so cached results from the old scoring aren't reused."""

CorrectnessJudge = Callable[[str, str, str], CorrectnessResult]
"""question, gold_answer, candidate_answer -> CorrectnessResult (v1's judge_correctness, bound to a client)."""

NoContextQuery = Callable[[str], str]
"""question -> answer with no retrieval at all, for ablation."""

ADAPTER_ERROR_JUDGE_MODEL = "n/a (adapter raised)"


class Selection(BaseModel):
    mode: str
    questions: list[Question]
    n_new: int = 0
    n_sampled_old: int = 0


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_config_hash(config: dict, adapter_sha256: str, judge_model: str, k: int) -> str:
    """Identity of everything that can change a result: pipeline config, adapter code, judge model, k."""
    payload = {
        "config": config,
        "adapter_sha256": adapter_sha256,
        "judge_model": judge_model,
        "k": k,
        "scoring_version": SCORING_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def score_retrieval_at_k(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str], k: int) -> tuple[float, float, bool]:
    """(recall@k, reciprocal rank, every gold chunk retrieved) over the first k retrieved ids."""
    top_k = retrieved_chunk_ids[:k]
    gold = set(gold_chunk_ids)
    found = gold & set(top_k)
    recall = len(found) / len(gold) if gold else 0.0
    first_rank = next((rank for rank, chunk_id in enumerate(top_k) if chunk_id in gold), None)
    reciprocal_rank = 1.0 / (first_rank + 1) if first_rank is not None else 0.0
    return recall, reciprocal_rank, bool(gold) and found == gold


def classify(retrieval_hit: bool, is_correct: bool) -> FailureClass:
    if not retrieval_hit:
        return FailureClass.RETRIEVAL_MISS
    return FailureClass.PASS if is_correct else FailureClass.GENERATION_MISS


def evaluate_question(question: Question, adapter: Adapter, judge: CorrectnessJudge, k: int) -> QuestionResult:
    """Run one question through the adapter and score it. An adapter exception is a scored failure, not a crash."""
    started = time.perf_counter()
    try:
        response = adapter.query(question.question)
        error = None
    except Exception as exc:
        logger.warning("Adapter raised for question_id=%s: %s", question.question_id, exc)
        response, error = RAGResponse(answer=""), str(exc)
    latency_ms = response.latency_ms if response.latency_ms is not None else (time.perf_counter() - started) * 1000

    recall, reciprocal_rank, hit = score_retrieval_at_k(question.chunk_ids, response.retrieved_chunk_ids, k)
    if error is None:
        correctness = judge(question.question, question.gold_answer, response.answer)
    else:
        correctness = CorrectnessResult(
            is_correct=False, rationale=f"adapter raised: {error}", judge_model=ADAPTER_ERROR_JUDGE_MODEL
        )

    return QuestionResult(
        question_id=question.question_id,
        split=question.split.value,
        answer=response.answer,
        retrieved_chunk_ids=response.retrieved_chunk_ids[:k],
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        retrieval_hit=hit,
        correctness=correctness,
        failure_class=classify(hit, correctness.is_correct),
        verdict=compute_verdict(is_correct=correctness.is_correct, retrieval_hit=hit),
        adapter_error=error,
        latency_ms=round(latency_ms, 2),
    )


def select_questions(
    store: Store,
    questions: list[Question],
    split: Split,
    config_hash: str,
    index_version: str,
    regression_sample: int = DEFAULT_REGRESSION_SAMPLE,
    seed: int = 0,
) -> Selection:
    """Pick which active questions this evaluation covers (see the module docstring)."""
    last_run = store.last_eval_run(config_hash, split)
    if last_run is None:
        return Selection(mode="full", questions=questions, n_new=len(questions))

    seen = store.result_question_ids(config_hash, split)
    new = [q for q in questions if q.question_id not in seen]
    old = [q for q in questions if q.question_id in seen]

    if last_run.index_version == index_version:
        return Selection(mode="cached", questions=old + new, n_new=len(new))

    rng = random.Random(f"{seed}:{index_version}")
    sampled_ids = set(rng.sample(sorted(q.question_id for q in old), min(regression_sample, len(old))))
    sampled = [q for q in old if q.question_id in sampled_ids]
    return Selection(mode="regression", questions=new + sampled, n_new=len(new), n_sampled_old=len(sampled))


def summarize(
    results: list[QuestionResult],
    *,
    split: Split,
    config: dict,
    config_hash: str,
    index_version: str,
    k: int,
    selection: Selection,
    n_active_in_split: int,
    n_cached: int,
) -> EvalSummary:
    n = len(results)

    def rate(count: float) -> float:
        return round(count / n, 4) if n else 0.0

    failure_counts = {fc.value: 0 for fc in FailureClass}
    verdict_counts = {v.value: 0 for v in Verdict}
    for result in results:
        failure_counts[result.failure_class.value] += 1
        verdict_counts[result.verdict.value] += 1

    lucky = [r for r in results if r.verdict == Verdict.LUCKY_PASS]
    ablated = [r for r in lucky if r.ablation is not None]
    return EvalSummary(
        run_id=str(uuid.uuid4()),
        created_at=utc_now(),
        split=split.value,
        config_hash=config_hash,
        config=config,
        index_version=index_version,
        k=k,
        mode=selection.mode,
        n_questions=n,
        n_active_in_split=n_active_in_split,
        n_new=selection.n_new,
        n_sampled_old=selection.n_sampled_old,
        n_cached=n_cached,
        n_run=n - n_cached,
        score=round(failure_counts[FailureClass.PASS.value] / n * 100, 2) if n else 0.0,
        answer_accuracy=rate(sum(r.correctness.is_correct for r in results)),
        recall_at_k=rate(sum(r.recall_at_k for r in results)),
        mrr=rate(sum(r.reciprocal_rank for r in results)),
        failure_counts=failure_counts,
        verdict_counts=verdict_counts,
        judge_failures=sum(r.correctness.judge_parse_failed for r in results),
        adapter_errors=sum(r.adapter_error is not None for r in results),
        lucky_pass_confirmed=sum(r.ablation.confirms_lucky_pass for r in ablated) if ablated else None,
    )


def _ablate(result: QuestionResult, question: Question, no_context_query: NoContextQuery, judge: CorrectnessJudge) -> QuestionResult:
    answer = no_context_query(question.question)
    is_correct = judge(question.question, question.gold_answer, answer).is_correct
    return result.model_copy(
        update={"ablation": AblationResult(answer=answer, is_correct=is_correct, confirms_lucky_pass=is_correct)}
    )


def run_evaluation(
    store: Store,
    questions: list[Question],
    split: Split,
    adapter: Adapter,
    judge: CorrectnessJudge,
    *,
    config: dict,
    config_hash: str,
    index_version: str,
    k: int = DEFAULT_K,
    regression_sample: int = DEFAULT_REGRESSION_SAMPLE,
    seed: int = 0,
    full: bool = False,
    no_context_query: NoContextQuery | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[EvalSummary, list[QuestionResult]]:
    """Evaluate the selected questions (cached results reused), save the run, return summary and results.

    `questions` must all be from `split`; callers pass dev questions
    directly and test questions only via ragaudit.runner.locked. Pass
    no_context_query to ablate LUCKY_PASS results (v1's confirmation step).
    """
    if any(q.split != split for q in questions):
        raise ValueError(f"run_evaluation got questions outside the {split.value} split")

    if full:
        selection = Selection(mode="full", questions=questions, n_new=len(questions))
    else:
        selection = select_questions(store, questions, split, config_hash, index_version, regression_sample, seed)

    results: list[QuestionResult] = []
    n_cached = 0
    for index, question in enumerate(selection.questions, start=1):
        result = store.load_result(question.question_id, config_hash, index_version)
        if result is None:
            result = evaluate_question(question, adapter, judge, k)
            store.save_result(result, config_hash, index_version)
        else:
            n_cached += 1
        if no_context_query is not None and result.verdict == Verdict.LUCKY_PASS and result.ablation is None:
            result = _ablate(result, question, no_context_query, judge)
            store.save_result(result, config_hash, index_version)
        results.append(result)
        if on_progress is not None:
            on_progress(index, len(selection.questions))

    summary = summarize(
        results,
        split=split,
        config=config,
        config_hash=config_hash,
        index_version=index_version,
        k=k,
        selection=selection,
        n_active_in_split=len(questions),
        n_cached=n_cached,
    )
    store.save_eval_run(summary)
    return summary, results


def evaluate_dev(store: Store, adapter: Adapter, judge: CorrectnessJudge, **kwargs) -> tuple[EvalSummary, list[QuestionResult]]:
    """Evaluate the active dev split."""
    return run_evaluation(store, store.dev_questions(), Split.DEV, adapter, judge, **kwargs)
