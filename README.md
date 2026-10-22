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
| `report-lucky.html` | `demo/lucky_corpus/` (well-known facts) | `ungrounded_adapter.py` | Answers correct anyway, retrieval contributing nothing. |

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

`demo/lucky_corpus/` (13 chunks: world capitals, the periodic table and
other basic chemistry, famous historical dates, basic physics constants)
flips the premise: these are facts any LLM already knows cold. It's
ingested into its own database (`ragaudit-lucky.db`) rather than sharing
the sample corpus's, so ingesting it can't overwrite chunks the other two
reports depend on. Running `ungrounded_adapter.py` against it, with
ablation enabled, means retrieval frequently returns the wrong chunk, or
a random one — yet the model answers correctly anyway, from pretraining,
and ablation (re-asking with no context at all) confirms it stayed
correct. `report-lucky.html` is what a corpus full of `LUCKY_PASS`
actually looks like: the case ragaudit exists to catch, that a
correctness-only eval would score as a clean pass.

- `demo/example_adapter.py` — keyword/BM25 retrieval, no embeddings, no
  vector DB, top-1 only — "healthy but mediocre."
- `demo/weak_adapter.py` — identical retrieval, except a seeded 30% of
  queries get a random chunk instead of the true top match, simulating a
  retrieval index that has quietly gone stale — "degraded." Its
  generation prompt strictly says "only use the provided context," so a
  bad retrieval mostly shows up as a visible failure, not a lucky guess.
- `demo/ungrounded_adapter.py` — same retrieval logic, degraded further
  (70% random), but with a permissive generation prompt: the context is
  offered as background, and the model is told to answer from its own
  knowledge if the context doesn't help. That prompt difference is the
  whole point — it's what turns bad retrieval into a *hidden* failure
  instead of a visible one, which is exactly what `report-lucky.html`
  is built to catch.

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

v2 (0.2.0) adds question generation with verification, cached and
regression evaluation, and config tuning with a locked test split; see
[v2: generate, evaluate, tune](#v2-generate-evaluate-tune).

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

## v2: generate, evaluate, tune

v1 audits one run. v2 adds a loop for improving a pipeline without fooling
yourself: build a question set that can't be gamed, evaluate cheaply as the
corpus changes, and let ragaudit search pipeline configs while keeping a
test split it never tunes on.

### What it does

- **`generate --corpus`** extracts facts from each chunk, writes a question
  per fact, rewrites it away from the passage's wording, and has a second
  model verify it is answerable, unambiguous and supported. Questions that
  still copy the passage, use relative time ("recently", "last quarter"),
  or fail verification are rejected with a reason. `--multi-hop` adds
  questions that need two chunks.
- **`evaluate`** runs your pipeline on the dev split. Results are cached per
  (question, config, corpus version), so an unchanged config re-scores
  instantly. When the corpus changes, questions on edited or deleted chunks
  go stale, and only new questions plus a seeded sample of old ones are
  re-run.
- **`tune`** searches a declared space of pipeline knobs (grid, seeded
  random, or an LLM that reads dev failures and proposes the next config),
  then scores the baseline and the best config once each on the locked test
  split.
- **`report --kind generate|tune`** renders both as self-contained HTML.

### Why the scores are honest

- **Questions don't share wording with their passage.** If over half of a
  question's content words appear in its source chunk (configurable with
  `--lexical-threshold`), it gets one more rewrite and is then rejected.
  Keyword search alone can't ace the set.
- **A second model verifies every question.** The verifier must be a
  different model from the generator.
- **The split is per chunk, fixed by hash.** Each chunk goes to dev (70% of
  chunks) or test (30%) from a salted hash of its chunk id, and every
  question from that chunk inherits its split, so no chunk feeds both
  sides. Tuning on dev can't teach the pipeline a test chunk. Hashing the
  id, not the content, means editing a chunk never moves it. A multi-hop
  question is only kept when all its chunks share a split; otherwise it's
  rejected as `cross_split` (and `generate` only pairs same-split chunks).
- **Tuning can't see the test split.** Reading test questions raises an
  error unless one function, used only for the final score, unlocks it. A
  test also parses the tuning code and fails if it imports the lock or the
  database directly. The LLM proposer only ever sees dev data.
- **A pass needs retrieval and a correct answer.** The score is the share of
  questions where the gold chunk was in the top k *and* the answer was
  judged correct. A correct answer without the gold chunk counts as a
  retrieval miss, so tuning can't reward a model answering from memory.
- **Overfitting is flagged.** If the best config's dev score beats its test
  score by more than `--overfit-threshold` points (default 10), the run says
  so.

### Quickstart

Offline, with the bundled demo and the deterministic mock LLM:

```bash
pip install -e .
sh demo/v2/run_demo.sh      # reports in demo/v2/output/
```

Against your own corpus and pipeline, with a real model:

```bash
export RAGAUDIT_API_KEY=sk-... RAGAUDIT_MODEL=gpt-4o-mini
ragaudit generate --corpus corpus.jsonl --verifier-model gpt-4o
ragaudit evaluate --adapter my_pipeline.py --corpus corpus.jsonl
ragaudit tune --adapter my_pipeline.py --search-space space.yaml --strategy llm
ragaudit report --kind tune --out tune-report.html
```

A corpus is a `.jsonl` file, one chunk per line:
`{"chunk_id": "...", "text": "...", "timestamp": "2026-01-12", "doc_id": "...", "position": 0}`
(only `chunk_id` and `text` are required), or a directory of `.txt`/`.md`
files, which get ids like `guide.md#3`. Chunk ids must stay stable across
edits so ragaudit can tell an edited chunk from a new one.

For `tune`, the adapter exposes `make_adapter(config) -> adapter` (see
`demo/v2/toy_adapter.py`); plain v1 adapters still work with `evaluate`.
The search space is YAML:

```yaml
knobs:
  chunk_size: [200, 400, 800]
  top_k: [3, 5, 10]
  hybrid_weight: {min: 0.0, max: 1.0, step: 0.25}
  reranker: [none, cross_encoder]
baseline:            # optional; defaults to each knob's first value
  chunk_size: 400
```

`tune` writes the best config to `best_config.yaml`, which `evaluate
--config` accepts directly. Reading YAML uses PyYAML if installed
(`pip install "ragaudit[yaml]"`) and a built-in parser for simple files
otherwise.

### CLI reference

| Command | Key options |
|---|---|
| `ragaudit generate --corpus PATH` | `--n` (max facts per chunk, default 3), `--multi-hop`, `--multi-hop-pairs 10`, `--lexical-threshold 0.5`, `--max-chunks`, `--max-llm-calls`, `--generator-model`, `--verifier-model`, `--retry-processed` |
| `ragaudit evaluate --adapter FILE` | `--config YAML`, `--corpus PATH`, `--split dev\|test`, `--k 5`, `--regression-sample 50`, `--seed`, `--full`, `--ablation`, `--judge-model` |
| `ragaudit tune --adapter FILE --search-space YAML` | `--strategy grid\|random\|llm`, `--max-trials 20`, `--max-llm-calls`, `--seed`, `--k 5`, `--overfit-threshold 10`, `--out best_config.yaml`, `--judge-model`, `--proposer-model` |
| `ragaudit report --kind generate\|tune` | `--run-id` (generate run or tune id; default latest), `--out` |

Every v2 command takes `--store` (default `.ragaudit/store.db`, separate
from v1's `ragaudit.db`) and `--llm openai|mock` (or `RAGAUDIT_LLM`).
`--llm mock` runs everything offline with a deterministic mock that has
deliberate flaws, for demos and tests; its numbers say nothing about a
real pipeline. Without `--corpus`, `generate` is the unchanged v1 command,
and `report` defaults to the v1 run report.

## License

MIT
