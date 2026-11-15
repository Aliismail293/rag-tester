"""`ragaudit improve`: the self-improving loop in one command, with no YAML or JSONL to write.

1. Read the docs folder directly (stable chunk ids, v2's directory corpus).
2. `generate` verified, de-worded questions with the per-chunk dev/test
   split, reusing every question already written for unchanged passages.
3. `tune` on dev (practice) only, starting from the baseline settings, then
   score baseline and best once on the locked test split (the exam).
4. Write best_config.yaml, a history row and the report.

Knobs come from the adapter's `KNOBS` dict (`--search-space` overrides).
Without knobs it explains how to add them, evaluates the current settings
and still writes the report.
"""

import math
from pathlib import Path

import typer

from ragaudit import history
from ragaudit.adapters.base import AdapterLoadError, _import_adapter_module, load_adapter_factory
from ragaudit.cli_easy import _adapter_env, _fail
from ragaudit.generate.pipeline import GenerateOptions, generate_questions, source_key
from ragaudit.ingest.corpus import CorpusError, load_corpus
from ragaudit.llm.factory import CallBudget, LLMBudgetExceeded, make_client, resolve_provider
from ragaudit.models.question import Split
from ragaudit.runner.evaluate import DEFAULT_K, compute_config_hash, file_sha256, run_evaluation
from ragaudit.scoring.correctness import judge_correctness
from ragaudit.scoring.text_match import DEFAULT_MATCH_THRESHOLD
from ragaudit.storage.store import Store, StoreSchemaError
from ragaudit.tune.search_space import SearchSpaceError, load_search_space, parse_search_space

DEFAULT_BUDGET_CALLS = 400
DEFAULT_MAX_TRIALS = 8
TARGET_QUESTIONS = 30
"""Active questions improve aims for (dev + test), before the budget trims it."""
MAX_FACTS_PER_CHUNK = 2
GEN_CALLS_PER_CHUNK = 1 + MAX_FACTS_PER_CHUNK * 3.5
"""Fact extraction, then per fact: question, de-wording (1-2), verification."""
ACTIVE_PER_CHUNK = 1.2
"""Questions per chunk that survive the checks, on average."""
DEV_SHARE = 0.7
MIN_GENERATE = 5
"""Passages to write from even when the budget is tight (below this, the run can't tune anything)."""
SECONDS_PER_CALL = {"openai": 3.0, "mock": 0.0}

NO_KNOBS_HELP = (
    "Your adapter declares no settings to tune, so improve will only measure the current settings. "
    "To let it improve them, add a KNOBS dict to {adapter} listing each setting and the values to try, "
    'current value first (for example KNOBS = {{"top_k": [4, 2, 8], "chunk_size": [800, 400]}}), and a '
    "make_adapter(config) function that builds your pipeline from one value of each; `ragaudit init` "
    "templates have both, commented. Then run this command again."
)


def _knobs_space(adapter_path: Path, search_space: Path | None):
    """(SearchSpace or None, has make_adapter)."""
    module = _import_adapter_module(str(adapter_path))
    has_factory = callable(getattr(module, "make_adapter", None))
    if search_space is not None:
        return load_search_space(search_space), has_factory
    knobs = getattr(module, "KNOBS", None)
    if not knobs:
        return None, has_factory
    if not isinstance(knobs, dict):
        raise SearchSpaceError(f"KNOBS in {adapter_path} must be a dict of setting -> list of values")
    baseline = getattr(module, "BASELINE", None) or {}
    return parse_search_space({"knobs": knobs, "baseline": baseline}), has_factory


def _covered_ids(store: Store) -> set[str]:
    keys = store.active_source_keys() | store.attempted_source_keys()
    return {key.rsplit("@", 1)[0] for key in keys if "+" not in key}


def plan_generation(store: Store, chunks: list, target_questions: int = TARGET_QUESTIONS):
    """(chunks in the order to process, how many to process, changed count).

    Edited chunks that had questions come first, then new chunks spread
    evenly over the docs, up to the target.
    """
    from ragaudit.check_questions import spread

    done = store.active_source_keys() | store.attempted_source_keys()
    covered = _covered_ids(store)
    current = [c for c in chunks if source_key([c]) in done]
    pending = [c for c in chunks if source_key([c]) not in done]
    changed = [c for c in pending if c.chunk_id in covered]
    others = [c for c in pending if c.chunk_id not in covered]
    target_chunks = min(len(chunks), math.ceil(target_questions / ACTIVE_PER_CHUNK))
    quota = max(0, min(len(others), target_chunks - len(current)))
    picked = [c for c, _ in spread(others, quota)] if quota else []
    picked_ids = {c.chunk_id for c in picked}
    order = changed + picked + [c for c in others if c.chunk_id not in picked_ids] + current
    return order, len(changed) + len(picked), len(changed)


def estimate_calls(n_chunks: int, n_dev: int, n_test: int, trials: int) -> tuple[int, int, int]:
    """(total, generation, evaluation) LLM calls for generating from n_chunks and running `trials` configs."""
    new_q = n_chunks * ACTIVE_PER_CHUNK
    dev = n_dev + new_q * DEV_SHARE
    test = n_test + new_q * (1 - DEV_SHARE)
    gen = math.ceil(n_chunks * GEN_CALLS_PER_CHUNK)
    evaluation = math.ceil(trials * dev + 2 * test)
    return gen + evaluation, gen, evaluation


def _judge_for(client):
    def judge(question, expected, candidate):
        return judge_correctness(client, question, expected, candidate)

    return judge


def _config_changes(before: dict, after: dict) -> str:
    changes = [f"{k} {before.get(k)} → {after[k]}" for k in after if before.get(k) != after[k]]
    return ", ".join(changes) if changes else "no change beat your current settings"


def run_improve(
    docs: Path,
    adapter_path: Path,
    *,
    budget_calls: int = DEFAULT_BUDGET_CALLS,
    max_trials: int = DEFAULT_MAX_TRIALS,
    note: str | None = None,
    yes: bool = False,
    no_open: bool = False,
    llm: str | None = None,
    generator_model: str | None = None,
    verifier_model: str | None = None,
    judge_model: str | None = None,
    search_space: Path | None = None,
    strategy: str = "random",
    seed: int = 0,
    k: int = DEFAULT_K,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    out: Path = Path("ragaudit-report.html"),
    config_out: Path = Path("best_config.yaml"),
    progress=None,
) -> list[str]:
    with _adapter_env(llm, docs if docs.is_dir() else None):
        return _run_improve(
            docs, adapter_path, budget_calls, max_trials, note, yes, no_open, llm, generator_model, verifier_model,
            judge_model, search_space, strategy, seed, k, match_threshold, out, config_out, progress,
        )


def _run_improve(
    docs, adapter_path, budget_calls, max_trials, note, yes, no_open, llm, generator_model, verifier_model,
    judge_model, search_space, strategy, seed, k, match_threshold, out, config_out, progress,
) -> list[str]:
    import webbrowser

    from ragaudit.cli_easy import adapter_llm_calls
    from ragaudit.cli_v2 import _generation_clients, write_best_config
    from ragaudit.report import v3
    from ragaudit.runner.locked import test_split_outcomes
    from ragaudit.tune.loop import TuneError, run_tune

    def report_progress(phase: str, done: int, total: int, calls: int) -> None:
        if progress is not None:
            progress(phase, done, total, calls)

    if budget_calls < 1:
        _fail("--budget-calls must be at least 1")
    if max_trials < 1:
        _fail("--max-trials must be at least 1")
    if strategy not in ("grid", "random", "llm"):
        _fail(f"--strategy must be grid, random or llm, got {strategy!r}")
    try:
        provider = resolve_provider(llm)
    except ValueError as exc:
        _fail(str(exc))
    if not docs.is_dir():
        _fail(f"docs folder not found: {docs}")
    try:
        chunks = load_corpus(docs)
    except CorpusError as exc:
        _fail(str(exc))
    if not chunks:
        _fail(f"no .txt or .md files with text found under {docs}")
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    try:
        space, has_factory = _knobs_space(adapter_path, search_space)
        factory = load_adapter_factory(str(adapter_path))
    except AdapterLoadError as exc:
        _fail(f"{exc} Run `ragaudit doctor --adapter {adapter_path}` for the exact fix.")
    except SearchSpaceError as exc:
        _fail(str(exc))
    if space is not None and not has_factory and search_space is None:
        _fail(f"{adapter_path} declares KNOBS but no make_adapter(config); add one that builds your pipeline from a config.")
    if space is None:
        typer.echo(NO_KNOBS_HELP.format(adapter=adapter_path))
        typer.echo("")
    try:
        sample = factory(space.baseline if space else {})
    except Exception as exc:
        _fail(f"the adapter could not be built with its baseline settings: {exc}")

    generator, verifier = _generation_clients(llm, generator_model, verifier_model)
    judge_client = make_client(provider, judge_model, mock_model="mock-judge")
    try:
        store = Store(history.store_path_for_docs(docs))
    except StoreSchemaError as exc:
        _fail(str(exc))
    try:
        order, n_generate, n_changed = plan_generation(store, chunks)
        n_dev, n_test = len(store.question_ids(Split.DEV)), len(store.question_ids(Split.TEST))
        trials = max_trials if space else 1
        total, gen_calls, eval_calls = estimate_calls(n_generate, n_dev, n_test, trials)
        floor = max(n_changed, min(n_generate, MIN_GENERATE))
        while n_generate > floor and total > budget_calls:
            n_generate -= 1
            total, gen_calls, eval_calls = estimate_calls(n_generate, n_dev, n_test, trials)
        own = adapter_llm_calls(sample)
        reused = n_dev + n_test
        typer.echo(
            f"Docs: {len(chunks)} passage(s). Questions: {reused} reused for unchanged passages; "
            f"writing more from {n_generate} passage(s)" + (f" ({n_changed} edited)." if n_changed else ".")
        )
        typer.echo(
            f"Estimated LLM calls: about {total} ({gen_calls} to write and verify questions, {eval_calls} to judge "
            f"{trials} setting(s) on practice plus baseline and best on the exam). Budget: {budget_calls} (--budget-calls)."
        )
        minutes = total * SECONDS_PER_CALL.get(provider, 3.0) / 60
        if provider == "mock":
            typer.echo("LLM: mock (offline, free; its scores say nothing about a real pipeline).")
        else:
            typer.echo(f"Estimated time: about {max(1, round(minutes))} minute(s), depending on your model.")
        if own:
            typer.echo(f"Your adapter answers through the same LLM: about {eval_calls // 2 + 1} more call(s), outside the budget.")
        if total > budget_calls:
            typer.echo("The estimate is above the budget; improve stops cleanly when the budget runs out and keeps its progress.")
        if not yes and not typer.confirm("Continue?", default=True):
            typer.echo("Cancelled. Nothing was sent to the LLM.")
            raise typer.Exit(code=0)

        budget = CallBudget(budget_calls)
        # Leave room for at least one practice pass and the two exam passes.
        exp_dev = n_dev + n_generate * ACTIVE_PER_CHUNK * DEV_SHARE
        exp_test = n_test + n_generate * ACTIVE_PER_CHUNK * (1 - DEV_SHARE)
        gen_budget = CallBudget(max(0, budget_calls - math.ceil(exp_dev + 2 * exp_test)))

        def on_chunk(done: int, planned: int) -> None:
            typer.echo(f"  generate {done}/{planned}")
            report_progress("generate", done, planned, gen_budget.used)

        summary = generate_questions(
            store, order, generator, verifier,
            GenerateOptions(max_facts_per_chunk=MAX_FACTS_PER_CHUNK, max_chunks=n_generate),
            gen_budget, on_chunk=on_chunk,
        )
        budget.used = gen_budget.used
        index_version = summary.index_version
        n_dev, n_test = len(store.question_ids(Split.DEV)), len(store.question_ids(Split.TEST))
        typer.echo(
            f"Questions: {summary.generated} new, {summary.rejected} rejected by the checks; "
            f"{n_dev} practice and {n_test} exam question(s) in total."
        )
        if summary.stopped_reason == "max_llm_calls":
            typer.echo("Stopped writing questions at the budget; the rest are written next time.")
        store_note = "Questions written so far are saved and reused next time."
        done_keys = store.active_source_keys() | store.attempted_source_keys()
        untried = sum(1 for c in chunks if source_key([c]) not in done_keys)
        budget_limited = summary.stopped_reason == "max_llm_calls" or budget.remaining == 0 or untried > 0
        if n_dev == 0:
            if budget_limited:
                typer.echo(
                    f"Stopped: no practice question passed the checks yet, and {untried} passage(s) were not tried "
                    f"within the budget of {budget_calls} LLM call(s). {store_note} Raise --budget-calls."
                )
                raise typer.Exit(code=0)
            _fail("no practice questions could be written from these docs; check RAGAUDIT_MODEL with `ragaudit doctor`.")
        if space is not None and n_test == 0 and budget_limited:
            typer.echo(
                f"Stopped: no exam question passed the checks yet, and {untried} passage(s) were not tried within "
                f"the budget of {budget_calls} LLM call(s). Settings can only be compared fairly on exam questions. "
                f"{store_note} Raise --budget-calls."
            )
            raise typer.Exit(code=0)

        remaining = budget.remaining
        adapter_sha = file_sha256(adapter_path)
        result = None
        if space is not None and n_test == 0:
            typer.echo("No exam questions yet (the docs are small), so improve can't score settings fairly; evaluating only.")
        elif space is not None:
            def on_trial(trial, best) -> None:
                typer.echo(f"  trial {trial.index + 1}/{max_trials} practice {trial.dev.score:.0f}  (best {best.dev.score:.0f})")
                report_progress("trial", trial.index + 1, max_trials, budget.used)

            def guarded(config):
                try:
                    return factory(config)
                except Exception as exc:
                    raise TuneError(f"adapter rejected config {config}: {exc}") from exc

            try:
                result = run_tune(
                    store, guarded, adapter_sha, str(adapter_path), space, judge_client, index_version,
                    strategy=strategy, max_trials=max_trials, max_llm_calls=remaining, seed=seed, k=k,
                    on_trial=on_trial, chunk_texts=chunk_texts, match_threshold=match_threshold,
                )
            except TuneError as exc:
                if "too small" in str(exc):
                    typer.echo(
                        f"Stopped: {remaining} LLM call(s) are left in the budget, not enough to score the settings "
                        f"on practice and the exam ({exc}). Questions are saved and reused next time; raise --budget-calls."
                    )
                    raise typer.Exit(code=0)
                _fail(str(exc))
            budget.used += result.llm_calls

        dev_questions = store.dev_questions()
        if result is not None:
            result.best_config_path = str(write_best_config(result, config_out))
            store.finish_tune_run(result.tune_id, result.finished_at, result.model_dump_json())
            typer.echo(f"  final exam: baseline {result.baseline_test.score:.0f} → best {result.best_test.score:.0f}")
            best_hash = result.best.config_hash
            results = [r for r in (store.load_result(q.question_id, best_hash, index_version) for q in dev_questions) if r]
            extra = history.tune_extra(result)
            extra.update(report_path=str(out.resolve()), docs=str(docs), llm_calls=budget.used)
            row = history.record_eval_summary(
                store, result.best_test, kind="improve", run_id=result.tune_id,
                outcomes=test_split_outcomes(store, result.best_test.config_hash, index_version),
                adapter_path=str(adapter_path), note=note, extra=extra,
                models={"judge": judge_client.model, "generator": generator.model, "verifier": verifier.model},
            )
            scope = (
                f"The questions below show the best settings on {len(results)} practice question(s). "
                f"The honest score is the locked exam: {result.best_test.score:.0f} (see Improvement loop)."
            )
            lines = [
                f"Final exam ({result.best_test.n_questions} locked question(s)): baseline {result.baseline_test.score:.0f} "
                f"→ best {result.best_test.score:.0f} ({result.best_test.score - result.baseline_test.score:+.0f} pts).",
                f"Practice: best {result.best.dev.score:.0f} on {result.best.dev.n_questions} question(s); practice − exam = "
                f"{result.overfit_gap:.0f} pts" + (": likely memorising." if result.overfit_warning else ": no sign of memorising."),
                f"Best settings ({len(result.trials)} trial(s)): {_config_changes(result.baseline.config, result.best.config)}; "
                f"saved to {config_out}.",
            ]
        else:
            config = space.baseline if space else {}
            text_hash = compute_config_hash(config, adapter_sha, judge_client.model, k, match_threshold)

            def on_question(done: int, total_q: int) -> None:
                report_progress("answer", done, total_q, budget.used)

            from ragaudit.llm.factory import BudgetedCompleter

            eval_budget = CallBudget(budget.remaining)
            try:
                eval_summary, results = run_evaluation(
                    store, dev_questions, Split.DEV, factory(config), _judge_for(BudgetedCompleter(judge_client, eval_budget)),
                    config=config, config_hash=text_hash, index_version=index_version, k=k, full=False,
                    chunk_texts=chunk_texts, match_threshold=match_threshold, on_progress=on_question,
                )
            except LLMBudgetExceeded:
                typer.echo(f"Stopped: the budget of {budget_calls} LLM call(s) ran out while scoring. Raise --budget-calls.")
                raise typer.Exit(code=0)
            budget.used += eval_budget.used
            row = history.record_eval_summary(
                store, eval_summary, kind="evaluate", outcomes={r.question_id: r.verdict.value for r in results},
                adapter_path=str(adapter_path), note=note, models={"judge": judge_client.model},
                extra={"split": "dev", "report_path": str(out.resolve()), "docs": str(docs), "llm_calls": budget.used,
                       "match_by": "text", "match_threshold": match_threshold},
            )
            why = "no KNOBS" if space is None else "no exam questions"
            scope = f"Current settings on {len(results)} practice question(s); nothing was tuned ({why})."
            lines = [
                f"Practice score with your current settings: {eval_summary.score:.0f} on {eval_summary.n_questions} question(s).",
                "Nothing was tuned: add KNOBS and make_adapter(config) to your adapter (see above)." if space is None
                else "Nothing was tuned: these docs gave no exam questions, so settings can't be compared fairly.",
                f"Questions are saved; the next run reuses them ({n_dev} practice, {n_test} exam).",
            ]
        rows = history.load_history(store)
    finally:
        store.close()

    report_path = v3.render_improve(
        questions=dev_questions, results=results, out=out, run_id=row.run_id, created_at=row.created_at,
        adapter_path=str(adapter_path), history_rows=rows, docs=str(docs), chunk_texts=chunk_texts,
        llm_calls=budget.used, match_threshold=match_threshold, scope_note=scope, kind=row.kind,
    )
    lines.append(f"LLM calls: {budget.used} of {budget_calls}. Report: {report_path}")
    if not no_open:
        webbrowser.open(report_path.resolve().as_uri())
    typer.echo("")
    typer.secho("ragaudit improve: summary", bold=True)
    for line in lines:
        typer.echo(f"  {line}")
    return lines
