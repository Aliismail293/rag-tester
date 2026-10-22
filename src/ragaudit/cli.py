"""The ragaudit CLI: ingest -> generate -> run -> report.

This module only wires together already-implemented modules (loading,
chunking, generation, running, ablation, scoring aggregation, storage,
and reporting) behind a Typer app. It contains no scoring or business
logic of its own.

v2 adds `generate --corpus`, `evaluate` and `tune`; their bodies live in
cli_v2.py. Without --corpus, `generate` runs exactly the v1 code path.
"""

import functools
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from ragaudit import cli_v2
from ragaudit.adapters.base import AdapterLoadError, load_adapter
from ragaudit.generate.pipeline import (
    DEFAULT_LEXICAL_THRESHOLD,
    DEFAULT_MAX_FACTS_PER_CHUNK,
    DEFAULT_MULTI_HOP_PAIRS,
)
from ragaudit.generate.testset_generator import generate_test_set
from ragaudit.ingest import chunker
from ragaudit.ingest.loader import load_documents as load_documents_from_disk
from ragaudit.llm.client import LLMClient, LLMClientError
from ragaudit.models.run import Run, RunConfig
from ragaudit.models.scoring import Verdict
from ragaudit.report.generator import generate_report
from ragaudit.runner.ablation import run_ablation
from ragaudit.runner.evaluate import DEFAULT_K, DEFAULT_REGRESSION_SAMPLE
from ragaudit.runner.executor import run_test_set
from ragaudit.runner.summary import compute_run_summary
from ragaudit.scoring.correctness import judge_correctness
from ragaudit.scoring.groundedness import judge_groundedness
from ragaudit.storage import db as db_module
from ragaudit.storage import repository
from ragaudit.storage.store import DEFAULT_STORE_PATH
from ragaudit.tune.loop import DEFAULT_MAX_TRIALS, DEFAULT_OVERFIT_THRESHOLD

app = typer.Typer(help="Audits RAG pipelines by checking whether retrieval actually did the work.")

DB_OPTION = typer.Option(Path("ragaudit.db"), "--db", help="SQLite database path.")
STORE_OPTION = typer.Option(DEFAULT_STORE_PATH, "--store", help="v2 store path.")
LLM_OPTION = typer.Option(
    None, "--llm", help="v2 LLM provider: openai (default, RAGAUDIT_API_*) or mock (offline). Env: RAGAUDIT_LLM."
)

NO_CONTEXT_SYSTEM_PROMPT = (
    "Answer the question using only your own internal knowledge. You have not "
    "been given any retrieved context or documents to work from."
)


def _error_exit(message: str) -> None:
    """Print a clean, colored error to stderr and exit with status 1."""
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _cli_safe(func):
    """Convert any exception that escapes a command into a clean error, never a stack trace."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except AdapterLoadError as exc:
            _error_exit(str(exc))
        except LLMClientError as exc:
            _error_exit(f"LLM request failed: {exc}")
        except KeyError as exc:
            _error_exit(f"missing required environment variable: {exc}")
        except Exception as exc:  # last-resort safety net
            _error_exit(f"unexpected error: {exc}")

    return wrapper


@app.command()
@_cli_safe
def ingest(
    path: Path = typer.Argument(..., help="Directory containing .txt/.md documents to ingest."),
    db: Path = DB_OPTION,
    chunk_size: int = typer.Option(chunker.DEFAULT_CHUNK_SIZE, "--chunk-size", help="Characters per chunk."),
    overlap: int = typer.Option(chunker.DEFAULT_OVERLAP, "--overlap", help="Character overlap between chunks."),
) -> None:
    """Load documents from PATH, chunk them, and persist to the database."""
    if not path.exists():
        _error_exit(f"path not found: {path}")

    documents = load_documents_from_disk(path)
    if not documents:
        _error_exit(f"no .txt/.md documents found under {path}")

    all_chunks = []
    with typer.progressbar(documents, label="Chunking documents") as progress:
        for document in progress:
            all_chunks.extend(chunker.chunk_document(document, chunk_size=chunk_size, overlap=overlap))

    conn = db_module.get_connection(db)
    try:
        repository.save_documents(conn, documents)
        repository.save_chunks(conn, all_chunks)
    finally:
        conn.close()

    typer.echo(f"Ingested {len(documents)} document(s) into {len(all_chunks)} chunk(s). Saved to {db}")


@app.command()
@_cli_safe
def generate(
    n: Optional[int] = typer.Option(
        None,
        "--n",
        help="Number of questions to generate per chunk (default 1). With --corpus: max facts per chunk (default 3).",
    ),
    db: Path = DB_OPTION,
    corpus: Optional[Path] = typer.Option(
        None,
        "--corpus",
        help="v2: build a verified, de-worded, dev/test-split question set from this .jsonl file or directory.",
    ),
    store: Path = STORE_OPTION,
    llm: Optional[str] = LLM_OPTION,
    generator_model: Optional[str] = typer.Option(None, "--generator-model", help="v2: generator model."),
    verifier_model: Optional[str] = typer.Option(
        None, "--verifier-model", help="v2: verifier model; must differ from the generator."
    ),
    max_chunks: Optional[int] = typer.Option(None, "--max-chunks", help="v2: process at most this many chunks."),
    max_llm_calls: Optional[int] = typer.Option(None, "--max-llm-calls", help="v2: stop after this many LLM calls."),
    multi_hop: bool = typer.Option(False, "--multi-hop", help="v2: also write questions that need two chunks."),
    multi_hop_pairs: int = typer.Option(
        DEFAULT_MULTI_HOP_PAIRS, "--multi-hop-pairs", help="v2: max chunk pairs for multi-hop per run."
    ),
    lexical_threshold: float = typer.Option(
        DEFAULT_LEXICAL_THRESHOLD,
        "--lexical-threshold",
        help="v2: max share of question content words found in the chunk before a rewrite is rejected.",
    ),
    retry_processed: bool = typer.Option(
        False, "--retry-processed", help="v2: re-process chunks that previously yielded no active question."
    ),
) -> None:
    """Generate a test set from ingested chunks and persist it."""
    if corpus is not None:
        cli_v2.run_generate(
            corpus=corpus,
            store_path=store,
            max_facts_per_chunk=n if n is not None else DEFAULT_MAX_FACTS_PER_CHUNK,
            llm=llm,
            generator_model=generator_model,
            verifier_model=verifier_model,
            max_chunks=max_chunks,
            max_llm_calls=max_llm_calls,
            multi_hop=multi_hop,
            multi_hop_pairs=multi_hop_pairs,
            lexical_threshold=lexical_threshold,
            retry_processed=retry_processed,
        )
        return

    n = 1 if n is None else n
    if not db.exists():
        _error_exit(f"database not found at {db} — run `ragaudit ingest <path>` first.")

    conn = db_module.get_connection(db)
    try:
        documents = repository.load_documents(conn)
        chunks = repository.load_chunks(conn)
        if not documents or not chunks:
            _error_exit("no ingested documents found — run `ragaudit ingest <path>` first.")

        client = LLMClient()
        typer.echo(
            f"Generating {n} question(s) per chunk for {len(chunks)} chunk(s) using {client.model} "
            "(this may take a while)..."
        )
        test_set = generate_test_set(client, documents, chunks, n_per_chunk=n)
        repository.save_test_set(conn, test_set)
    finally:
        conn.close()

    typer.echo(
        f"Generated {len(test_set.test_cases)} test case(s) from {len(chunks)} chunk(s). "
        f"Saved test set '{test_set.testset_id}' to {db}"
    )


@app.command()
@_cli_safe
def evaluate(
    adapter: Path = typer.Option(..., "--adapter", help="Adapter file: `make_adapter(config)`, `adapter`, or `query`."),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="Current corpus, to detect new/changed/deleted chunks (default: last recorded)."
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML pipeline config passed to the adapter."),
    split: str = typer.Option("dev", "--split", help="dev (default) or test. Test goes through the locked path."),
    k: int = typer.Option(DEFAULT_K, "--k", help="Score retrieval over the top k retrieved chunk ids."),
    regression_sample: int = typer.Option(
        DEFAULT_REGRESSION_SAMPLE,
        "--regression-sample",
        help="When only the corpus changed: old questions to re-run alongside all new ones.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the regression sample."),
    full: bool = typer.Option(False, "--full", help="Ignore regression sampling and cover every active question."),
    ablation: bool = typer.Option(False, "--ablation", help="Re-ask lucky passes with no context (v1 ablation)."),
    store: Path = STORE_OPTION,
    llm: Optional[str] = LLM_OPTION,
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Correctness judge model."),
) -> None:
    """v2: evaluate a pipeline on generated questions, with caching and regression sampling."""
    cli_v2.run_evaluate(
        adapter_path=adapter,
        corpus=corpus,
        config_path=config,
        split=split,
        k=k,
        regression_sample=regression_sample,
        seed=seed,
        store_path=store,
        llm=llm,
        judge_model=judge_model,
        ablation=ablation,
        full=full,
    )


@app.command()
@_cli_safe
def tune(
    adapter: Path = typer.Option(..., "--adapter", help="Adapter file exposing `make_adapter(config)`."),
    search_space: Path = typer.Option(..., "--search-space", help="YAML file of knobs and their allowed values."),
    corpus: Optional[Path] = typer.Option(None, "--corpus", help="Current corpus (default: last recorded)."),
    strategy: str = typer.Option("grid", "--strategy", help="grid, random (seeded) or llm."),
    max_trials: int = typer.Option(DEFAULT_MAX_TRIALS, "--max-trials", help="Max configs to evaluate, baseline included."),
    max_llm_calls: Optional[int] = typer.Option(
        None, "--max-llm-calls", help="Cap on judge + proposer calls; room for the final test evaluation is reserved."
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the random strategy and fallbacks."),
    k: int = typer.Option(DEFAULT_K, "--k", help="Score retrieval over the top k retrieved chunk ids."),
    overfit_threshold: float = typer.Option(
        DEFAULT_OVERFIT_THRESHOLD, "--overfit-threshold", help="Warn when best dev minus best test exceeds this (points)."
    ),
    out: Path = typer.Option(Path("best_config.yaml"), "--out", help="Where to write the best config (YAML)."),
    store: Path = STORE_OPTION,
    llm: Optional[str] = LLM_OPTION,
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Correctness judge model."),
    proposer_model: Optional[str] = typer.Option(None, "--proposer-model", help="Model for --strategy llm."),
) -> None:
    """v2: search pipeline configs on the dev split; report baseline vs best on the locked test split."""
    cli_v2.run_tune_command(
        adapter_path=adapter,
        search_space_path=search_space,
        corpus=corpus,
        strategy=strategy,
        max_trials=max_trials,
        max_llm_calls=max_llm_calls,
        seed=seed,
        k=k,
        overfit_threshold=overfit_threshold,
        out=out,
        store_path=store,
        llm=llm,
        judge_model=judge_model,
        proposer_model=proposer_model,
    )


@app.command(name="run")
@_cli_safe
def run_cmd(
    adapter: Path = typer.Option(..., "--adapter", help="Path to a Python file exposing `query` or `adapter`."),
    db: Path = DB_OPTION,
    testset_id: Optional[str] = typer.Option(
        None, "--testset-id", help="Test set id to run (defaults to the most recently generated)."
    ),
    no_ablation: bool = typer.Option(False, "--no-ablation", help="Skip the retrieval-disabled ablation step."),
) -> None:
    """Run the adapter over a test set, score it, ablate lucky passes, and persist the Run."""
    if not db.exists():
        _error_exit(f"database not found at {db} — run `ragaudit ingest` and `ragaudit generate` first.")

    conn = db_module.get_connection(db)
    try:
        resolved_testset_id = testset_id or repository.get_latest_testset_id(conn)
        if resolved_testset_id is None:
            _error_exit("no test set found — run `ragaudit generate` first.")

        test_set = repository.load_test_set(conn, resolved_testset_id)
        if test_set is None:
            _error_exit(f"test set '{resolved_testset_id}' not found.")

        try:
            loaded_adapter = load_adapter(str(adapter))
        except AdapterLoadError as exc:
            _error_exit(str(exc))

        client = LLMClient()

        def correctness_judge(question: str, expected_answer: str, candidate_answer: str):
            return judge_correctness(client, question, expected_answer, candidate_answer)

        def groundedness_judge(answer: str, contexts: list[str]):
            return judge_groundedness(client, answer, contexts)

        typer.echo(f"Running {len(test_set.test_cases)} test case(s) through adapter {adapter}...")
        scored_cases = run_test_set(test_set, loaded_adapter, correctness_judge, groundedness_judge)

        run_ablation_flag = not no_ablation
        if run_ablation_flag:
            lucky_pass_count = sum(1 for case in scored_cases if case.verdict == Verdict.LUCKY_PASS)
            if lucky_pass_count:
                typer.echo(f"Ablation: enabled — running against {lucky_pass_count} lucky-pass candidate(s)...")

                def no_context_query(question: str) -> str:
                    return client.complete(NO_CONTEXT_SYSTEM_PROMPT, question)

                def check_correctness(question: str, expected_answer: str, candidate_answer: str) -> bool:
                    return judge_correctness(client, question, expected_answer, candidate_answer).is_correct

                scored_cases = run_ablation(scored_cases, test_set.test_cases, no_context_query, check_correctness)
            else:
                typer.echo("Ablation: enabled, but no lucky-pass candidates were found — nothing to ablate.")
        else:
            typer.echo("Ablation: disabled (--no-ablation).")

        summary = compute_run_summary(scored_cases)
        run_id = str(uuid.uuid4())
        run = Run(
            config=RunConfig(
                run_id=run_id,
                testset_id=test_set.testset_id,
                adapter_path=str(adapter),
                correctness_judge_model=client.model,
                groundedness_judge_model=client.model,
                run_ablation=run_ablation_flag,
                started_at=datetime.now(timezone.utc),
            ),
            finished_at=datetime.now(timezone.utc),
            scored_cases=scored_cases,
            summary=summary,
        )
        repository.save_run(conn, run)
    finally:
        conn.close()

    typer.echo(
        f"Run '{run_id}' complete: {summary.total_cases} case(s), "
        f"lucky-pass rate {summary.lucky_pass_rate:.1%}. Saved to {db}"
    )


@app.command()
@_cli_safe
def report(
    out: Optional[Path] = typer.Option(
        None, "--out", help="Output HTML file path (default report.html; generate-report.html / tune-report.html for v2 kinds)."
    ),
    db: Path = DB_OPTION,
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Run id to report on (defaults to the most recently run). For v2 kinds, the generate run or tune id."
    ),
    kind: str = typer.Option(
        "run", "--kind", help="run (v1 run report, the default), generate (v2 generate run) or tune (v2 tune run)."
    ),
    store: Path = STORE_OPTION,
) -> None:
    """Render the latest (or specified) Run into a static HTML report. v2: --kind generate|tune."""
    if kind in ("generate", "tune"):
        output_path = cli_v2.run_report(kind, store, out or Path(f"{kind}-report.html"), run_id)
        typer.echo(f"Report written to {output_path}")
        return
    if kind != "run":
        _error_exit(f"--kind must be run, generate or tune, got {kind!r}")
    out = out or Path("report.html")
    if not db.exists():
        _error_exit(f"database not found at {db} — run `ragaudit ingest`, `generate`, and `run` first.")

    conn = db_module.get_connection(db)
    try:
        resolved_run_id = run_id or repository.get_latest_run_id(conn)
        if resolved_run_id is None:
            _error_exit("no run found — run `ragaudit run --adapter <path>` first.")

        run = repository.load_run(conn, resolved_run_id)
        if run is None:
            _error_exit(f"run '{resolved_run_id}' not found.")

        test_set = repository.load_test_set(conn, run.config.testset_id)
        if test_set is None:
            _error_exit(f"test set '{run.config.testset_id}' referenced by run '{resolved_run_id}' not found.")
    finally:
        conn.close()

    output_path = generate_report(run, test_set, out)
    typer.echo(f"Report written to {output_path}")


if __name__ == "__main__":
    app()
