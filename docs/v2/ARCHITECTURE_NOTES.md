# ragaudit v2: architecture notes

Written during Phase 0 (recon), before any v2 code. Describes how v1 is put
together, what the existing `generate` module does, and where each v2 piece
plugs in.

## How v1 is structured

v1 is a small, layered Python package (`src/ragaudit`, ~1,300 lines) with a
Typer CLI on top. Every layer is a plain module with pure functions where
possible, and every LLM-dependent function takes a `Completer` protocol
(`model: str`, `complete(system, user) -> str`) so tests can stub it.

| Layer | Module | What it does |
|---|---|---|
| Models | `models/document.py`, `testset.py`, `scoring.py`, `run.py`, `adapter.py` | Pydantic models: `Document`, `Chunk`, `TestCase`/`TestSet`, `ScoredCase` + `Verdict`, `Run`/`RunSummary`, `RAGResponse` + `Adapter` protocol. |
| Ingest | `ingest/loader.py`, `ingest/chunker.py` | Walk a directory for `.txt`/`.md`; fixed-size character chunking with overlap. `chunk_id = sha256(doc_id:position:text)`, and `doc_id` includes the content hash, so **an edited document gets entirely new chunk ids**. |
| LLM | `llm/client.py` | `LLMClient`, a thin OpenAI-compatible chat client over httpx, configured from `RAGAUDIT_API_BASE`/`_API_KEY`/`_MODEL`; retries 429/5xx. |
| Generate | `generate/testset_generator.py` | See below. |
| Scoring | `scoring/correctness.py`, `groundedness.py`, `retrieval.py`, `verdict.py` | Two LLM judges (strict-JSON, conservative fallback on parse failure), a pure set-membership retrieval check, and the 2x2 verdict (`TRUE_PASS`, `LUCKY_PASS`, `GENERATION_FAILURE`, `RETRIEVAL_FAILURE`). |
| Runner | `runner/executor.py`, `ablation.py`, `summary.py` | Run a `TestSet` through an adapter (adapter exceptions become failed cases, never abort), ablate lucky passes with a no-context query, aggregate into a `RunSummary`. |
| Storage | `storage/db.py`, `repository.py` | One SQLite file (`ragaudit.db` default), `CREATE TABLE IF NOT EXISTS` schema, no versioning. `save_documents`/`save_chunks` replace the whole table on every ingest. |
| Report | `report/generator.py`, `templates/report.html.jinja`, `templates/style.css` | Jinja template, stylesheet **inlined** into a single self-contained HTML file. `docs/style.css` is a generated copy (`scripts/sync_landing_css.py`). |
| Adapters | `adapters/base.py`, `callable_adapter.py` | `load_adapter(path)` imports a user file exposing either `adapter` (object with `query`) or `query(question) -> RAGResponse`. |
| CLI | `cli.py` | `ingest`, `generate --n`, `run --adapter [--no-ablation]`, `report --out`; all take `--db`. `_cli_safe` turns any exception into a clean one-line error. |

Tests: 109 pytest tests (`tests/`), no network; LLM is stubbed with tiny
`StubClient` classes and `monkeypatch` on `ragaudit.cli.LLMClient`. CI runs
`pytest -v` on Python 3.11. No linter or formatter is configured.

Demo: `demo/run_demo.sh` (Docker) runs three scenarios against
`demo/sample_corpus` (invented facts) and `demo/lucky_corpus` (well-known
facts) with three BM25 adapters, producing the three reports published in
`docs/`.

## What the existing `generate` module does

`generate/testset_generator.py` makes **one LLM call per chunk** with a
system prompt asking for `n` questions answerable only from the passage,
phrased like a real user, never referring to "the passage". It parses
`{"questions": [{"question", "answer"}]}`, skips unparseable chunks with a
warning, and builds `TestCase`s (`test_id = sha256(chunk_id:index)`) into a
`TestSet` keyed by a corpus hash plus timestamp.

What it does **not** do, and v2 adds:

- no separation of fact extraction from question writing, so there's no
  explicit "fact" or gold answer provenance beyond the chunk id;
- no check that the question doesn't just copy the chunk's wording;
- no second-model verification of the answer key;
- no guard against relative time ("last week");
- no stable question identity: `test_id` depends on the response index, and
  every `generate` makes a brand-new test set;
- no dev/test split, no incremental behaviour, no budget caps.

## Where v2 plugs in

Guiding constraint: **v1 stays byte-for-byte identical**. v2 uses a
separate SQLite store (`.ragaudit/store.db`), separate templates and a
separate supplementary stylesheet (because v1 inlines `style.css` into
every report, editing it would change v1 output), and only *adds* options
to existing CLI commands.

### Files to create

| File | Phase | Purpose |
|---|---|---|
| `src/ragaudit/models/question.py` | 1 | `Question` model, `QuestionType`/`Split`/`QuestionStatus` enums, `compute_question_id`, `assign_split` / `split_for_chunks` (70/30 per chunk, by hash of `chunk_id`; questions inherit their chunks' split). |
| `src/ragaudit/models/evaluation.py` | 4 | `QuestionResult`, `FailureClass`, `EvalSummary`. |
| `src/ragaudit/models/tune.py` | 5 | `Trial`, `TuneResult`. |
| `src/ragaudit/storage/store.py` | 1, 3, 4, 5 | v2 store: schema-version table, questions, corpus versions, eval-result cache, eval runs, generate runs, tune runs/trials. Test-split reads are refused unless the locked evaluation path has unlocked them. |
| `src/ragaudit/runner/locked.py` | 1, 4 | The **only** code path that reads test-split gold answers. Returns aggregate scores only. |
| `src/ragaudit/ingest/corpus.py` | 2, 3 | Load a v2 corpus (JSONL with `chunk_id`, `text`, optional `timestamp`; or a directory chunked with stable `relpath#position` ids), `chunk_hash`, `index_version`, corpus diffing. |
| `src/ragaudit/llm/mock.py` | 2 | Deterministic offline `MockLLM` that answers every ragaudit prompt (generation, de-wording, verification, judging, tuning proposals, demo answering). |
| `src/ragaudit/llm/factory.py` | 2 | `make_client(provider, model)`: `mock` or the existing `LLMClient`; `BudgetedCompleter` for `--max-llm-calls`. |
| `src/ragaudit/generate/pipeline.py` | 2 | Fact extraction, question writing, de-wording + lexical overlap, verification, relative-time validator, multi-hop, incremental/budgeted orchestration. Extends the generate package; `testset_generator.py` is untouched. |
| `src/ragaudit/generate/prompts.py` | 2 | v2 prompts as constants (shared with the mock). |
| `src/ragaudit/generate/text.py` | 2 | Tokenizer, stopwords, lexical overlap, relative-time regex. |
| `src/ragaudit/runner/evaluate.py` | 4 | Score questions (recall@k, MRR, correctness via v1 `judge_correctness`, failure class, v1 verdict), result cache, regression sampling. |
| `src/ragaudit/tune/search_space.py`, `strategies.py`, `loop.py` | 5 | Search-space YAML, `grid`/`random`/`llm` strategies, closed loop on dev, final locked test evaluation, overfitting warning. |
| `src/ragaudit/yaml_io.py` | 5 | Read/write the small YAML subset used for configs; uses PyYAML if installed, otherwise a built-in subset parser (no new required dependency). |
| `src/ragaudit/report/v2.py`, `templates/generate_report.html.jinja`, `templates/tune_report.html.jinja`, `templates/v2.css` | 6 | Generate and tune reports with inline SVG charts. Inline `style.css` (unchanged) + `v2.css`. |
| `src/ragaudit/cli_v2.py` | 2, 4, 5, 6 | v2 command bodies (`generate --corpus`, `evaluate`, `tune`, `report --kind`). |
| `demo/v2/` | 7 | Sample corpus (~45 timestamped chunks + an update), toy adapter with real knobs, search space, demo script. |
| `tests/test_v2_*.py` | all | New tests; all LLM calls go through `MockLLM` or stubs. |
| `CHANGELOG.md` | 7 | |

### Files to modify (additively)

| File | Change |
|---|---|
| `src/ragaudit/cli.py` | `generate` gains v2 options; without `--corpus` it runs the exact v1 code path. Register `evaluate` and `tune`. `report` gains `--kind` (default `run` = v1 path). |
| `src/ragaudit/adapters/base.py` | Add `load_adapter_factory(path)` so a pipeline can receive a config dict (`make_adapter(config)` or `adapter.configure(config)`). `load_adapter` keeps its behaviour and error messages. |
| `src/ragaudit/models/adapter.py` | Add a `ConfigurableAdapter` protocol (documentation + typing only). |
| `README.md`, `docs/index.html` | v2 sections. |
| `pyproject.toml` | Version 0.2.0, optional `yaml` extra. |
| `.gitignore` | Ignore `.ragaudit/`. |

### Key design decisions (details in the progress log)

- **Chunk identity.** v2 requires stable chunk ids so a *changed* chunk
  (same id, new hash) can be told apart from a new one. JSONL corpora
  supply ids; directory corpora use `relative/path.md#position`.
- **Pipelines that re-chunk.** `chunk_size` is a tuning knob, but gold
  labels refer to canonical corpus chunks. A pipeline that re-chunks must
  map retrieved spans back to the canonical `chunk_id`s they came from
  (the toy adapter does this), otherwise retrieval can't be scored.
- **Score.** The headline score is the **pass rate** (gold chunk retrieved
  *and* answer correct), in points 0–100. A correct answer without the
  gold chunk is a `retrieval_miss` (and a v1 `LUCKY_PASS`), so lucky
  passes can't inflate the score that `tune` optimises.
- **Locked test set.** Enforced in three layers: the store refuses to
  return test-split questions or results unless the locked path in
  `runner/locked.py` has unlocked it; `tune` only receives a dev-only
  `DevSet` and aggregate test scores; and tests check both the runtime
  guard and that the `tune` package never imports the unlock machinery.
