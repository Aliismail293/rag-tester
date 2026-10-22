"""The closed tuning loop: search configs on the dev split, then score baseline and best once on the test split.

1. Trial 0 is the baseline config. Each later trial asks the strategy for
   a config, evaluates it on the **dev split only** (cached results are
   reused) and keeps the best dev score (ties go to the earlier trial).
2. When the search stops (max trials, space exhausted, or the LLM budget),
   the baseline and the best config are evaluated on the **locked test
   split, once each**, through ragaudit.runner.locked. Only aggregate
   scores come back.
3. If the best config's dev score beats its test score by more than the
   threshold, the run is flagged as likely overfit to the dev set.

The loop sees questions only through DevSet; it never touches the test
split except through `evaluate_test_split`.

LLM budget: --max-llm-calls covers the judge and the proposer. Room is
reserved up front for the final test evaluation (two configs x the test
set), and the search stops before a trial that might not fit, so the
honest test score is always produced.
"""

import uuid
from collections.abc import Callable

from ragaudit.adapters.base import AdapterFactory
from ragaudit.llm.factory import BudgetedCompleter, CallBudget, Completer
from ragaudit.models.evaluation import FailureClass, QuestionResult
from ragaudit.models.question import Split, utc_now
from ragaudit.models.tune import Trial, TuneResult
from ragaudit.runner.evaluate import compute_config_hash, run_evaluation
from ragaudit.runner.locked import count_test_questions, evaluate_test_split
from ragaudit.scoring.correctness import judge_correctness
from ragaudit.storage.store import Store
from ragaudit.tune.devset import DevSet
from ragaudit.tune.search_space import SearchSpace, canonical
from ragaudit.tune.strategies import make_strategy

DEFAULT_MAX_TRIALS = 20
DEFAULT_OVERFIT_THRESHOLD = 10.0
FAILED_EXAMPLES_FOR_PROPOSER = 5


class TuneError(Exception):
    """Raised when tuning can't start (no dev questions, budget too small for the final evaluation, ...)."""


def _judge_for(client: Completer):
    def judge(question: str, expected_answer: str, candidate_answer: str):
        return judge_correctness(client, question, expected_answer, candidate_answer)

    return judge


def _feedback(best: Trial, best_results: list[QuestionResult], dev: DevSet) -> dict:
    """Dev-only feedback for the proposer: the best config's failure counts and a few failed dev examples."""
    by_id = {q.question_id: q for q in dev.questions}
    failed = [r for r in best_results if r.failure_class != FailureClass.PASS][:FAILED_EXAMPLES_FOR_PROPOSER]
    return {
        "best_config": best.config,
        "best_dev_score": best.dev.score,
        "failure_counts": best.dev.failure_counts,
        "failed_examples": [
            {
                "question": by_id[r.question_id].question,
                "gold_answer": by_id[r.question_id].gold_answer,
                "pipeline_answer": r.answer,
                "failure_class": r.failure_class.value,
                "gold_chunk_retrieved": r.retrieval_hit,
            }
            for r in failed
            if r.question_id in by_id
        ],
    }


def run_tune(
    store: Store,
    adapter_factory: AdapterFactory,
    adapter_sha256: str,
    adapter_path: str,
    space: SearchSpace,
    judge_client: Completer,
    index_version: str,
    *,
    strategy: str = "grid",
    proposer_client: Completer | None = None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    max_llm_calls: int | None = None,
    seed: int = 0,
    k: int = 5,
    overfit_threshold: float = DEFAULT_OVERFIT_THRESHOLD,
    on_trial: Callable[[Trial, Trial], None] | None = None,
) -> TuneResult:
    dev = DevSet(store)
    if not dev.questions:
        raise TuneError("no active dev questions; run `ragaudit generate --corpus <path>` first.")
    n_test = count_test_questions(store)
    if n_test == 0:
        raise TuneError("no active test questions to score the final configs on; generate more questions first.")
    if max_trials < 1:
        raise TuneError("--max-trials must be at least 1")

    final_reserve = 2 * n_test
    if max_llm_calls is not None and max_llm_calls < final_reserve + len(dev):
        raise TuneError(
            f"--max-llm-calls {max_llm_calls} is too small: the baseline needs up to {len(dev)} judge calls on dev "
            f"and the final test evaluation reserves {final_reserve}."
        )
    search_budget = CallBudget(None if max_llm_calls is None else max_llm_calls - final_reserve)
    judge = _judge_for(BudgetedCompleter(judge_client, search_budget))
    proposer = BudgetedCompleter(proposer_client, search_budget) if proposer_client is not None else None
    chooser = make_strategy(strategy, space, seed, proposer)

    result = TuneResult(
        tune_id=str(uuid.uuid4()),
        created_at=utc_now(),
        strategy=strategy,
        search_space=space.knobs,
        adapter_path=adapter_path,
        judge_model=judge_client.model,
        index_version=index_version,
        k=k,
        seed=seed,
        max_trials=max_trials,
        overfit_threshold=overfit_threshold,
        stopped_reason="max_trials",
    )
    store.save_tune_run(result.tune_id, result.created_at, strategy, space.knobs)

    def config_hash(config: dict) -> str:
        return compute_config_hash(config, adapter_sha256, judge_client.model, k)

    def evaluate_dev(config: dict) -> tuple:
        return run_evaluation(
            store,
            dev.questions,
            Split.DEV,
            adapter_factory(config),
            judge,
            config=config,
            config_hash=config_hash(config),
            index_version=index_version,
            k=k,
            full=True,
        )

    trials: list[Trial] = []
    best_results: list[QuestionResult] = []
    next_proposal = (space.baseline, "baseline", None)
    while next_proposal is not None:
        config, source, rationale = next_proposal
        summary, results = evaluate_dev(config)
        trial = Trial(
            index=len(trials),
            config=config,
            config_hash=config_hash(config),
            source=source,
            rationale=rationale,
            dev=summary,
            created_at=utc_now(),
        )
        trials.append(trial)
        store.save_tune_trial(result.tune_id, trial)
        if trial.dev.score > trials[result.best_index].dev.score or trial.index == 0:
            result.best_index = trial.index
            best_results = results
        if on_trial is not None:
            on_trial(trial, trials[result.best_index])

        if len(trials) >= max_trials:
            result.stopped_reason = "max_trials"
            break
        worst_case_trial = len(dev) + (2 if proposer is not None else 0)
        if search_budget.remaining is not None and search_budget.remaining < worst_case_trial:
            result.stopped_reason = "max_llm_calls"
            break
        proposal = chooser.propose(trials, _feedback(trials[result.best_index], best_results, dev))
        if proposal is None:
            result.stopped_reason = "search_space_exhausted"
            break
        next_proposal = (proposal.config, proposal.source, proposal.rationale)

    result.trials = trials
    final_budget = CallBudget(None)
    final_judge = _judge_for(BudgetedCompleter(judge_client, final_budget))

    def evaluate_test(config: dict):
        return evaluate_test_split(
            store,
            adapter_factory(config),
            final_judge,
            config=config,
            config_hash=config_hash(config),
            index_version=index_version,
            k=k,
            full=True,
        )

    result.baseline_test = evaluate_test(result.baseline.config)
    same_config = canonical(result.best.config) == canonical(result.baseline.config)
    result.best_test = result.baseline_test if same_config else evaluate_test(result.best.config)

    result.overfit_gap = round(result.best.dev.score - result.best_test.score, 2)
    result.overfit_warning = result.overfit_gap > overfit_threshold
    result.llm_calls = search_budget.used + final_budget.used
    result.finished_at = utc_now()
    store.finish_tune_run(result.tune_id, result.finished_at, result.model_dump_json())
    return result
