"""Command bodies for the v2 CLI (generate --corpus, evaluate, tune).

cli.py owns the Typer app and option declarations; this module holds the
v2 logic behind them so the v1 commands in cli.py stay small and
unchanged. Like cli.py, it contains no scoring or generation logic of its
own: it only wires modules together and prints results.
"""

import os
from pathlib import Path

import typer

from ragaudit import yaml_io
from ragaudit.adapters.base import AdapterLoadError, load_adapter_factory
from ragaudit.generate.pipeline import GenerateOptions, GenerateSummary, generate_questions, source_key
from ragaudit.ingest.corpus import CorpusError, load_corpus
from ragaudit.ingest.versioning import CorpusDiff, sync_corpus
from ragaudit.llm.factory import PROVIDER_ENV, CallBudget, make_client, resolve_provider
from ragaudit.models.evaluation import EvalSummary
from ragaudit.models.question import Split
from ragaudit.models.tune import Trial, TuneResult
from ragaudit.runner.evaluate import compute_config_hash, file_sha256, run_evaluation
from ragaudit.runner.locked import evaluate_test_split
from ragaudit.scoring.correctness import judge_correctness
from ragaudit.storage.store import Store, StoreSchemaError
from ragaudit.tune.loop import TuneError, run_tune
from ragaudit.tune.search_space import SearchSpaceError, load_search_space

VERIFIER_MODEL_ENV = "RAGAUDIT_VERIFIER_MODEL"


def _fail(message: str) -> None:
    from ragaudit.cli import _error_exit

    _error_exit(message)


def _open_store(path: Path) -> Store:
    try:
        return Store(path)
    except StoreSchemaError as exc:
        _fail(str(exc))


def _load_corpus_or_fail(path: Path):
    try:
        chunks = load_corpus(path)
    except CorpusError as exc:
        _fail(str(exc))
    if not chunks:
        _fail(f"no chunks found in corpus {path}")
    return chunks


def _resolve_provider(llm: str | None) -> str:
    """Resolve --llm, and export it so the adapter under test uses the same provider for its own LLM calls."""
    try:
        provider = resolve_provider(llm)
    except ValueError as exc:
        _fail(str(exc))
    if llm is not None:
        os.environ[PROVIDER_ENV] = provider
    return provider


def _generation_clients(llm: str | None, generator_model: str | None, verifier_model: str | None):
    """Generator and verifier clients. The verifier must be a different model from the generator."""
    provider = _resolve_provider(llm)
    if provider == "mock":
        generator = make_client("mock", generator_model, mock_model="mock-generator")
        verifier = make_client("mock", verifier_model, mock_model="mock-verifier")
    else:
        verifier_model = verifier_model or os.environ.get(VERIFIER_MODEL_ENV)
        if not verifier_model:
            _fail(
                f"a verifier model is required: pass --verifier-model or set {VERIFIER_MODEL_ENV}. "
                "It must be a different model from the generator, so it can catch the generator's mistakes."
            )
        generator = make_client("openai", generator_model)
        verifier = make_client("openai", verifier_model)

    if generator.model == verifier.model:
        _fail(
            f"the verifier model must differ from the generator model (both are {generator.model!r}); "
            "a model checking its own output misses its own mistakes."
        )
    return generator, verifier


def format_generate_summary(summary: GenerateSummary, totals: dict) -> list[str]:
    lines = []
    if summary.corpus_diff:
        diff = summary.corpus_diff
        lines.append(
            f"Corpus: {summary.corpus_chunks} chunk(s), index {summary.index_version[:12]}: "
            f"{diff.get('new', 0)} new, {diff.get('changed', 0)} changed, {diff.get('deleted', 0)} deleted, "
            f"{diff.get('unchanged', 0)} unchanged; {diff.get('marked_stale', 0)} question(s) marked stale"
            + (f", {diff['revived']} revived" if diff.get("revived") else "")
            + "."
        )
    lines.append(
        f"Chunks: {summary.chunks_processed} processed, {summary.chunks_skipped} skipped "
        f"(already have questions at this content hash), {summary.chunks_remaining} remaining."
    )
    if summary.chunks_unparseable:
        lines.append(f"  {summary.chunks_unparseable} chunk(s) returned unparseable facts and will be retried next run.")
    if summary.pairs_processed or summary.pairs_remaining:
        lines.append(f"Multi-hop pairs: {summary.pairs_processed} processed, {summary.pairs_remaining} remaining.")

    lines.append(
        f"Questions: {summary.generated} generated (active), {summary.rejected} rejected"
        + (f", {summary.duplicates} already in the store" if summary.duplicates else "")
        + (f"; {summary.multi_hop_generated} of the generated are multi-hop" if summary.multi_hop_generated else "")
        + "."
    )
    if summary.reject_reasons:
        reasons = ", ".join(f"{reason} {count}" for reason, count in sorted(summary.reject_reasons.items()))
        lines.append(f"  Rejected by reason: {reasons}.")
    lines.append(f"  Split of new active questions: dev {summary.split_counts['dev']}, test {summary.split_counts['test']}.")
    lines.append(f"LLM calls: {summary.llm_calls}.")

    by_split = totals["by_split_status"]
    lines.append(
        "Store totals: "
        f"dev {by_split['dev']['active']} active / test {by_split['test']['active']} active, "
        f"{by_split['dev']['stale'] + by_split['test']['stale']} stale, "
        f"{by_split['dev']['rejected'] + by_split['test']['rejected']} rejected."
    )

    if summary.stopped_reason == "max_chunks":
        lines.append(f"Stopped at --max-chunks; {summary.chunks_remaining} chunk(s) still need questions. Re-run to continue.")
    elif summary.stopped_reason == "max_llm_calls":
        remaining = f"{summary.chunks_remaining} chunk(s)"
        if summary.pairs_remaining:
            remaining += f" and {summary.pairs_remaining} multi-hop pair(s)"
        lines.append(f"Stopped at --max-llm-calls; {remaining} still need questions. Progress is saved; re-run to continue.")
    return lines


def run_generate(
    corpus: Path,
    store_path: Path,
    max_facts_per_chunk: int,
    llm: str | None,
    generator_model: str | None,
    verifier_model: str | None,
    max_chunks: int | None,
    max_llm_calls: int | None,
    multi_hop: bool,
    multi_hop_pairs: int,
    lexical_threshold: float,
    retry_processed: bool,
) -> GenerateSummary:
    chunks = _load_corpus_or_fail(corpus)
    generator, verifier = _generation_clients(llm, generator_model, verifier_model)
    options = GenerateOptions(
        max_facts_per_chunk=max_facts_per_chunk,
        lexical_threshold=lexical_threshold,
        multi_hop=multi_hop,
        multi_hop_pairs=multi_hop_pairs,
        max_chunks=max_chunks,
        retry_processed=retry_processed,
    )

    store = _open_store(store_path)
    try:
        typer.echo(
            f"Generating questions for {len(chunks)} chunk(s) with {generator.model}, "
            f"verifying with {verifier.model}..."
        )
        summary = generate_questions(store, chunks, generator, verifier, options, CallBudget(max_llm_calls))
        totals = store.question_counts()
    finally:
        store.close()

    for line in format_generate_summary(summary, totals):
        typer.echo(line)
    typer.echo(f"Saved to {store_path} (generate run {summary.run_id}).")
    return summary


# --- evaluate ----------------------------------------------------------------


def load_config_file(path: Path | None) -> dict:
    """A pipeline config: a YAML mapping of knob -> value (a tune best-config file works as-is)."""
    if path is None:
        return {}
    if not path.is_file():
        _fail(f"config file not found: {path}")
    try:
        data = yaml_io.load(path)
    except yaml_io.YAMLError as exc:
        _fail(f"could not parse config file {path}: {exc}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        _fail(f"config file {path} must contain a mapping of knob: value")
    return data


def _format_diff(diff: CorpusDiff, n_chunks: int) -> str:
    counts = diff.counts()
    text = (
        f"Corpus: {n_chunks} chunk(s), index {diff.index_version[:12]}: {counts['new']} new, "
        f"{counts['changed']} changed, {counts['deleted']} deleted, {counts['unchanged']} unchanged; "
        f"{counts['marked_stale']} question(s) marked stale"
    )
    if counts["revived"]:
        text += f", {counts['revived']} revived"
    return text + "."


def sync_or_load_index(store: Store, corpus: Path | None) -> str:
    """Sync the corpus (printing the diff) and return the current index_version."""
    if corpus is None:
        latest = store.latest_corpus_version()
        if latest is None:
            _fail("no corpus version recorded in the store; pass --corpus (or run `ragaudit generate --corpus` first).")
        typer.echo(f"Corpus: using the last recorded version, index {latest[0][:12]} (pass --corpus to check for changes).")
        return latest[0]

    chunks = _load_corpus_or_fail(corpus)
    diff = sync_corpus(store, chunks)
    typer.echo(_format_diff(diff, len(chunks)))
    done = store.active_source_keys() | store.attempted_source_keys()
    needing = sum(1 for chunk in chunks if source_key([chunk]) not in done)
    if needing:
        typer.echo(f"  {needing} chunk(s) have no questions yet; run `ragaudit generate --corpus {corpus}` to cover them.")
    return diff.index_version


def format_eval_summary(summary: EvalSummary) -> list[str]:
    mode = {
        "full": "full (config not evaluated before)",
        "regression": f"regression ({summary.n_new} new + {summary.n_sampled_old} sampled old question(s))",
        "cached": "cached (corpus and config unchanged)",
    }[summary.mode]
    failures = summary.failure_counts
    verdicts = summary.verdict_counts
    lines = [
        f"Evaluated {summary.n_questions} of {summary.n_active_in_split} active {summary.split} question(s); "
        f"mode {mode}; {summary.n_cached} from cache, {summary.n_run} run.",
        f"Score (pass rate): {summary.score:.1f}  |  answer accuracy {summary.answer_accuracy:.1%}  |  "
        f"recall@{summary.k} {summary.recall_at_k:.1%}  |  MRR {summary.mrr:.3f}",
        f"Failure classes: pass {failures['pass']}, retrieval_miss {failures['retrieval_miss']}, "
        f"generation_miss {failures['generation_miss']}.",
        f"v1 verdicts: true_pass {verdicts['true_pass']}, lucky_pass {verdicts['lucky_pass']}, "
        f"generation_failure {verdicts['generation_failure']}, retrieval_failure {verdicts['retrieval_failure']}.",
    ]
    if summary.lucky_pass_confirmed is not None:
        lines.append(f"Ablation: {summary.lucky_pass_confirmed} lucky pass(es) stayed correct with retrieval off.")
    if summary.judge_failures or summary.adapter_errors:
        lines.append(
            f"Warnings: {summary.judge_failures} judge parse failure(s), {summary.adapter_errors} adapter error(s); "
            "both are scored as failures."
        )
    if summary.mode == "regression":
        lines.append(
            "Note: in regression mode the score covers new questions plus a sample of old ones, not the whole split."
        )
    return lines


def build_adapter(adapter_path: Path, config: dict):
    try:
        return load_adapter_factory(str(adapter_path))(config)
    except AdapterLoadError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(f"adapter rejected the config: {exc}")


def make_judge(provider: str, judge_model: str | None):
    client = make_client(provider, judge_model, mock_model="mock-judge")

    def judge(question: str, expected_answer: str, candidate_answer: str):
        return judge_correctness(client, question, expected_answer, candidate_answer)

    return client, judge


def run_evaluate(
    adapter_path: Path,
    corpus: Path | None,
    config_path: Path | None,
    split: str,
    k: int,
    regression_sample: int,
    seed: int,
    store_path: Path,
    llm: str | None,
    judge_model: str | None,
    ablation: bool,
    full: bool,
) -> EvalSummary:
    from ragaudit.cli import NO_CONTEXT_SYSTEM_PROMPT

    try:
        split_enum = Split(split)
    except ValueError:
        _fail(f"--split must be dev or test, got {split!r}")
    provider = _resolve_provider(llm)
    config = load_config_file(config_path)

    store = _open_store(store_path)
    try:
        index_version = sync_or_load_index(store, corpus)
        if not store.question_ids(split_enum):
            _fail(f"no active {split} questions in {store_path}; run `ragaudit generate --corpus <path>` first.")

        adapter = build_adapter(adapter_path, config)
        judge_client, judge = make_judge(provider, judge_model)
        config_hash = compute_config_hash(config, file_sha256(adapter_path), judge_client.model, k)
        kwargs = dict(
            config=config,
            config_hash=config_hash,
            index_version=index_version,
            k=k,
            regression_sample=regression_sample,
            seed=seed,
            full=full,
            no_context_query=(lambda q: judge_client.complete(NO_CONTEXT_SYSTEM_PROMPT, q)) if ablation else None,
        )
        typer.echo(f"Evaluating the {split} split with adapter {adapter_path}, judge {judge_client.model}...")
        if split_enum == Split.TEST:
            summary = evaluate_test_split(store, adapter, judge, **kwargs)
        else:
            summary, _ = run_evaluation(store, store.dev_questions(), Split.DEV, adapter, judge, **kwargs)
    finally:
        store.close()

    for line in format_eval_summary(summary):
        typer.echo(line)
    typer.echo(f"Saved eval run {summary.run_id} (config {config_hash[:12]}, index {index_version[:12]}) to {store_path}.")
    return summary


# --- tune --------------------------------------------------------------------


def _config_text(config: dict) -> str:
    return " ".join(f"{knob}={value}" for knob, value in config.items())


def format_tune_result(result: TuneResult) -> list[str]:
    baseline, best = result.baseline, result.best
    lines = [
        f"Search stopped: {result.stopped_reason.replace('_', ' ')} after {len(result.trials)} trial(s); "
        f"best is trial {best.index} ({best.source}).",
        "",
        "Final scores (pass rate, points). Test = locked split, evaluated once per config:",
        f"  {'':10}{'dev':>8}{'test':>8}",
        f"  {'baseline':10}{baseline.dev.score:8.1f}{result.baseline_test.score:8.1f}",
        f"  {'best':10}{best.dev.score:8.1f}{result.best_test.score:8.1f}",
        "",
        f"Improvement on test: {result.best_test.score - result.baseline_test.score:+.1f} points "
        f"({result.baseline_test.n_questions} test question(s)).",
    ]
    gap = result.overfit_gap
    if result.overfit_warning:
        lines.append(
            f"WARNING: likely overfitting. The best config scores {gap:.1f} points higher on dev than on test "
            f"(threshold {result.overfit_threshold:g}). Its dev score is inflated by the search; trust the test score, "
            "and consider more dev questions or a smaller search space."
        )
    else:
        lines.append(
            f"Dev-test gap for the best config: {gap:+.1f} points (threshold {result.overfit_threshold:g}): "
            "no sign of overfitting to the dev set."
        )
    lines.append(f"LLM calls (judge + proposer): {result.llm_calls}.")
    return lines


def write_best_config(result: TuneResult, path: Path) -> Path:
    best = result.best
    header = (
        f"Best config from ragaudit tune {result.tune_id}\n"
        f"strategy {result.strategy}, {len(result.trials)} trial(s), trial {best.index}; "
        f"dev {best.dev.score:.1f}, locked test {result.best_test.score:.1f}\n"
        f"Use with: ragaudit evaluate --adapter {result.adapter_path} --config {path}"
    )
    return yaml_io.dump(best.config, path, header=header)


def run_tune_command(
    adapter_path: Path,
    search_space_path: Path,
    corpus: Path | None,
    strategy: str,
    max_trials: int,
    max_llm_calls: int | None,
    seed: int,
    k: int,
    overfit_threshold: float,
    out: Path,
    store_path: Path,
    llm: str | None,
    judge_model: str | None,
    proposer_model: str | None,
) -> TuneResult:
    if strategy not in ("grid", "random", "llm"):
        _fail(f"--strategy must be grid, random or llm, got {strategy!r}")
    provider = _resolve_provider(llm)
    try:
        space = load_search_space(search_space_path)
    except SearchSpaceError as exc:
        _fail(str(exc))

    store = _open_store(store_path)
    try:
        index_version = sync_or_load_index(store, corpus)
        build_adapter(adapter_path, space.baseline)
        base_factory = load_adapter_factory(str(adapter_path))

        def factory(config: dict):
            try:
                return base_factory(config)
            except (ValueError, AdapterLoadError) as exc:
                raise TuneError(f"adapter rejected config {_config_text(config)}: {exc}") from exc

        judge_client = make_client(provider, judge_model, mock_model="mock-judge")
        proposer = make_client(provider, proposer_model, mock_model="mock-proposer") if strategy == "llm" else None
        typer.echo(
            f"Tuning {adapter_path} over {space.size} possible config(s) with strategy {strategy}, "
            f"up to {max_trials} trial(s), scored on the dev split only..."
        )

        def on_trial(trial: Trial, best: Trial) -> None:
            typer.echo(
                f"  trial {trial.index:>2} [{trial.source:<12}] dev {trial.dev.score:5.1f}  "
                f"(best {best.dev.score:5.1f})  {_config_text(trial.config)}"
            )

        try:
            result = run_tune(
                store,
                factory,
                file_sha256(adapter_path),
                str(adapter_path),
                space,
                judge_client,
                index_version,
                strategy=strategy,
                proposer_client=proposer,
                max_trials=max_trials,
                max_llm_calls=max_llm_calls,
                seed=seed,
                k=k,
                overfit_threshold=overfit_threshold,
                on_trial=on_trial,
            )
        except TuneError as exc:
            _fail(str(exc))
        result.best_config_path = str(write_best_config(result, out))
        store.finish_tune_run(result.tune_id, result.finished_at, result.model_dump_json())
    finally:
        store.close()

    typer.echo("")
    for line in format_tune_result(result):
        typer.echo(line)
    typer.echo(f"Best config written to {out}. Trial log saved to {store_path} (tune {result.tune_id}).")
    return result
