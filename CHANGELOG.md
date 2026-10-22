# Changelog

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
