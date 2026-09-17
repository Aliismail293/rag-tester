# ragaudit

## Try it in 60 seconds

Requires Docker and an OpenAI-compatible API key.

```bash
export RAGAUDIT_API_KEY=sk-...
export RAGAUDIT_API_BASE=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export RAGAUDIT_MODEL=gpt-4o-mini

docker compose up --build
```

This ingests a small invented corpus (`demo/sample_corpus/`), generates a
test set from it, runs the test set through a deliberately mediocre
keyword/BM25 adapter (`demo/example_adapter.py`, no embeddings, no vector
DB), and writes the finished report to `demo/output/report.html`. Open
that file in a browser once the container exits.

The demo corpus describes an invented company, an invented product, and
invented internal processes — facts that don't exist anywhere for an LLM
to have learned during pretraining. That's deliberate: it means any
answer the adapter gets right had to actually come from the retrieved
context, so a `LUCKY_PASS` in this demo's report is a real, meaningful
finding rather than an artifact of the model already knowing the answer.

Audits RAG pipelines by checking whether retrieval actually did the work —
not just whether the final answer was correct.

A RAG system can return a correct answer while retrieving the wrong
documents, because the model answered from pretraining rather than from
context. Standard evals score that as a pass. ragaudit catches it and
labels it `LUCKY_PASS`.

## Status

Functional end to end: ingestion, test-set generation, running, scoring
(including ablation), storage, reporting, and the CLI are all implemented.
See `demo/` for a working example.

## How it works

1. Ingest a corpus of documents; ragaudit assigns stable chunk IDs.
2. Generate a test set: an LLM writes questions whose answers live in each
   chunk, storing `(question, expected_answer, source_chunk_id)`.
3. Run your RAG pipeline through an adapter.
4. Score answer correctness, retrieval hit rate, and groundedness.
5. Cross-reference correctness against retrieval:
   - correct + retrieved → `TRUE_PASS`
   - correct + NOT retrieved → `LUCKY_PASS` (the headline output)
   - wrong + retrieved → `GENERATION_FAILURE`
   - wrong + NOT retrieved → `RETRIEVAL_FAILURE`
6. For candidate `LUCKY_PASS` cases, re-run the question with retrieval
   disabled. If the answer is still correct, retrieval contributed
   nothing — this empirically confirms the verdict.
7. Emit a static HTML report.

## Important: chunk IDs must match

ragaudit's retrieval scoring is set membership: it checks whether the
`source_chunk_id` it assigned during ingestion appears in the
`retrieved_chunk_ids` your adapter returns. This only works if **your RAG
pipeline is indexed against the chunks ragaudit produced**, not chunks it
made independently with its own boundaries and IDs.

The supported workflow is:

1. Run `ragaudit ingest ./docs` to chunk your corpus and assign IDs.
2. Index *those chunks* into your own pipeline (vector store, etc.), so
   your pipeline's retrieved documents carry ragaudit's chunk IDs as
   metadata.
3. Your adapter's `query()` returns those same IDs in
   `retrieved_chunk_ids`.

If your pipeline chunks the corpus itself and can't surface ragaudit's
IDs, retrieval hit rate will read as 0% and every correct answer will be
misreported as `LUCKY_PASS` — not because retrieval failed, but because
the tool can't recognize IDs it didn't assign. There is no ID-agnostic
fallback (e.g. text-overlap matching) yet; it may be added later as an
opt-in `--match-by text` mode for pipelines that can't be re-indexed.

## Adapter interface

Implement one function:

```python
def query(question: str) -> RAGResponse:
    ...

# RAGResponse = { answer: str, retrieved_chunk_ids: list[str], contexts: list[str] }
```

Adapters ship for: a raw callable, LangChain, LlamaIndex.

## CLI

```
ragaudit ingest ./docs                  # load, chunk, and persist a corpus
ragaudit generate --n 100               # generate a test set (n questions/chunk)
ragaudit run --adapter my_adapter.py    # run + score + ablate against the latest test set
ragaudit report --out report.html       # render the latest run as a static HTML report
```

Every command takes `--db` (default `./ragaudit.db`) to point at a
specific database file.

## License

MIT
