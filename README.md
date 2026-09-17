# ragaudit

## Try it in 60 seconds

Requires Docker and an OpenAI-compatible API key.

```bash
export RAGAUDIT_API_KEY=sk-...
export RAGAUDIT_API_BASE=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export RAGAUDIT_MODEL=gpt-4o-mini

docker compose up --build
```

This runs three scenarios and writes three reports to `demo/output/`:

| Report | Corpus | Adapter | What it shows |
|---|---|---|---|
| `report-healthy.html` | `demo/sample_corpus/` (invented facts) | `example_adapter.py` | Retrieval working, answers grounded. |
| `report-degraded.html` | `demo/sample_corpus/` (invented facts) | `weak_adapter.py` | Retrieval quietly broken (30% random chunks), failures visible. |
| `report-lucky.html` | `demo/lucky_corpus/` (well-known facts) | `weak_adapter.py` | Answers correct anyway, retrieval contributing nothing. |

`demo/sample_corpus/` (18 chunks, deliberately overlapping topics —
multiple chunks touch the same product or the same SLA terms) describes
an invented company, product, and internal processes — facts that don't
exist anywhere for an LLM to have learned during pretraining. That's
deliberate: any answer an adapter gets right in the first two reports
had to actually come from the retrieved context, so a `LUCKY_PASS` there
would be a real, meaningful finding, not an artifact of the model
already knowing the answer. Comparing `report-healthy.html` against
`report-degraded.html` — same test set, same generation model, only
retrieval quality changed — shows how much retrieval quality alone moves
the verdict breakdown.

`demo/lucky_corpus/` (12 chunks: capital cities, basic chemistry, famous
historical dates) flips the premise: these are facts any LLM already
knows cold. Running the *degraded* adapter against it means retrieval
frequently returns the wrong chunk, or a random one — yet the model
answers correctly anyway, from pretraining, and ablation (re-asking with
no context at all) confirms it stayed correct. `report-lucky.html` is
what a corpus full of `LUCKY_PASS` actually looks like: the case ragaudit
exists to catch, that a correctness-only eval would score as a clean
pass.

- `demo/example_adapter.py` — keyword/BM25 retrieval, no embeddings, no
  vector DB, top-1 only — "healthy but mediocre."
- `demo/weak_adapter.py` — identical retrieval, except a seeded 30% of
  queries get a random chunk instead of the true top match, simulating a
  retrieval index that has quietly gone stale — "degraded."

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
