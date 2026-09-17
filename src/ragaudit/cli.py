"""The ragaudit CLI: ingest -> generate -> run -> report.

This module only wires together already-implemented modules (loading,
chunking, generation, running, ablation, scoring aggregation, storage,
and reporting) behind a Typer app. It contains no scoring or business
logic of its own.
"""

import functools
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from ragaudit.adapters.base import AdapterLoadError, load_adapter
from ragaudit.generate.testset_generator import generate_test_set
from ragaudit.ingest import chunker
from ragaudit.ingest.loader import load_documents as load_documents_from_disk
from ragaudit.llm.client import LLMClient, LLMClientError
from ragaudit.models.run import Run, RunConfig
from ragaudit.models.scoring import Verdict
from ragaudit.report.generator import generate_report
from ragaudit.runner.ablation import run_ablation
from ragaudit.runner.executor import run_test_set
from ragaudit.runner.summary import compute_run_summary
from ragaudit.scoring.correctness import judge_correctness
from ragaudit.scoring.groundedness import judge_groundedness
from ragaudit.storage import db as db_module
from ragaudit.storage import repository

app = typer.Typer(help="Audits RAG pipelines by checking whether retrieval actually did the work.")

DB_OPTION = typer.Option(Path("ragaudit.db"), "--db", help="SQLite database path.")

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
    n: int = typer.Option(1, "--n", help="Number of questions to generate per chunk."),
    db: Path = DB_OPTION,
) -> None:
    """Generate a test set from ingested chunks and persist it."""
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
    out: Path = typer.Option(Path("report.html"), "--out", help="Output HTML file path."),
    db: Path = DB_OPTION,
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Run id to report on (defaults to the most recently run)."
    ),
) -> None:
    """Render the latest (or specified) Run into a static HTML report."""
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
