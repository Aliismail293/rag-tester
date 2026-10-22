"""Command bodies for the v2 CLI (generate --corpus, evaluate, tune).

cli.py owns the Typer app and option declarations; this module holds the
v2 logic behind them so the v1 commands in cli.py stay small and
unchanged. Like cli.py, it contains no scoring or generation logic of its
own: it only wires modules together and prints results.
"""

import os
from pathlib import Path

import typer

from ragaudit.generate.pipeline import GenerateOptions, GenerateSummary, generate_questions
from ragaudit.ingest.corpus import CorpusError, load_corpus
from ragaudit.llm.factory import CallBudget, make_client, resolve_provider
from ragaudit.storage.store import Store, StoreSchemaError

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


def _generation_clients(llm: str | None, generator_model: str | None, verifier_model: str | None):
    """Generator and verifier clients. The verifier must be a different model from the generator."""
    try:
        provider = resolve_provider(llm)
    except ValueError as exc:
        _fail(str(exc))

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
