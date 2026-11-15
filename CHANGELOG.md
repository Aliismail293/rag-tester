# Changelog

## 0.3.0

Easy mode: a first-time user can test their own pipeline in a few
minutes, without re-indexing it. Existing commands, flags and defaults
are unchanged.

### Added

- `--match-by id|text` (and `--match-threshold`, default 0.5) on `run`
  and `evaluate`. Text mode counts a retrieval hit when one of the
  contexts the adapter returns overlaps the gold chunk's text (overlap of
  word pairs, measured against the shorter side), so pipelines that chunk
  the corpus themselves work as they are. `id` stays the default. In
  `evaluate`, text mode needs `--corpus` and is cached as its own config;
  id-mode cache keys are unchanged.
- `ragaudit init --framework plain|langchain|llamaindex`: a commented
  adapter with one TODO where the pipeline plugs in, plus a `.env.example`
  for OpenAI-compatible APIs and local Ollama.
- `ragaudit init --framework demo`: a ready-to-run toy pipeline (keyword
  search over the docs folder, top 3 paragraphs, answered by the configured
  `RAGAUDIT_*` LLM) for seeing a full report before wiring up your own.
  `check` points it at its `--docs` folder and counts its answer calls in
  the estimate. `--llm` is now passed on to adapters that use ragaudit's
  LLM, in `check` and `doctor`.
- `ragaudit doctor`: PASS/FAIL with an exact fix for the Python version,
  the `RAGAUDIT_*` settings (whether set, never their values), one test
  call to the API, the adapter import, and a valid `RAGResponse` on a
  sample question.
- `ragaudit check --docs PATH --adapter FILE`: ingest, generate (20
  questions by default, `--questions`), run with ablation and report in
  one command, matching by text unless `--match-by id`. It tries the
  adapter once first, prints the estimated LLM calls and asks before
  spending (`--yes` to skip), stops at a budget cap (`--max-llm-calls`,
  default the estimate's upper bound), opens the report (`--no-open` to
  skip), and prints a summary of at most six lines.
- The same plain-English summary (verdict, retrieval hit rate, answer
  correctness, lucky passes, most useful fix) at the top of the HTML run
  report. The docs sample reports are re-rendered with it.
- All commands read `RAGAUDIT_*` settings from `./.env`; exported
  variables take priority.
- Published on PyPI as `ragaudit-cli` (`pip install ragaudit-cli`;
  the YAML extra is `pip install "ragaudit-cli[yaml]"`). The command and
  the import package are still `ragaudit`.
- PyPI packaging metadata and a release workflow
  (`.github/workflows/publish.yml`) that publishes on a `vX.Y.Z` tag via
  trusted publishing.
- "Test your RAG in 5 minutes" at the top of the README and the landing
  page.

### Fixed

- The source distribution contained only the report templates, with no
  Python code.
- Two report tests depended on gitignored demo databases, so CI failed.
- Long ids in the run report header, and the landing page's install
  section, no longer widen the page on phones.

## 0.2.0

v2: generate, evaluate and tune with honest scores. Everything in 0.1.0
works as before; v1 commands, flags, outputs and reports are unchanged.

### Added

- `ragaudit generate --corpus PATH`: fact extraction per chunk, de-worded
  questions with a lexical-overlap check, a relative-time check, and
  verification by a second model. Rejected questions keep a reason.
  Optional multi-hop questions (`--multi-hop`). Budgets: `--max-chunks`,
  `--max-llm-calls`. Re-runs skip chunks that already have questions.
- A v2 store (`.ragaudit/store.db`) with a per-chunk dev (70% of chunks) /
  test (30%) split: each chunk's split comes from a hash of its id, and
  all its questions inherit it, so no chunk feeds both splits. Multi-hop
  questions spanning both splits are rejected as `cross_split`. The test
  split is locked: reading it raises unless the single final-evaluation
  path unlocks it.
- Corpus versioning: each corpus is fingerprinted, and edited or deleted
  chunks mark their questions stale (reverting an edit revives them).
- `ragaudit evaluate`: scores a pipeline on the dev split (or, explicitly,
  the test split) with results cached per question, config and corpus
  version; regression mode re-runs new questions plus a seeded sample of
  old ones when only the corpus changed.
- `ragaudit tune`: grid, seeded random or LLM-proposed configs over a YAML
  search space, evaluated on dev; baseline and best are scored once on the
  locked test split, with an overfitting warning. Writes the best config
  as YAML.
- `ragaudit report --kind generate|tune`: HTML reports for generate and
  tune runs.
- `--llm mock`: an offline, deterministic mock LLM for demos and tests.
- `demo/v2/`: a 48-chunk invented corpus, a corpus update, a configurable
  toy pipeline, a search space and `run_demo.sh`.
- Optional extra `ragaudit[yaml]` (PyYAML); a built-in parser handles
  simple YAML without it.

## 0.1.0

- Initial release: ingest, generate, run (with ablation), report, and the
  LUCKY_PASS verdict.
