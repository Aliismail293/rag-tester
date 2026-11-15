# v3 architecture notes

Where run history, `ragaudit improve`, `ragaudit ui` and the new report plug
into the existing code, and which files change.

## Run history (`.ragaudit/store.db`)

- **Store.** The v2 store (`storage/store.py`, already schema-versioned)
  gets schema version 3: a `history` table (one summary row per run) and a
  `check_questions` table (the reusable question pool for `check`). Opening a
  version-1 or version-2 store migrates it in place; v2 tables are untouched.
- **Per project.** `check` and `improve` keep their store in
  `<docs folder's parent>/.ragaudit/store.db`, so two projects never share
  history. `evaluate` and `tune` write to their `--store`; `run` writes to the
  default store in the working folder.
- **Fair comparisons.** `check` now chunks the docs with stable ids
  (`relative/path.md#position`, from `ingest/corpus.py`) and keeps its
  questions in `check_questions`, keyed by chunk id and chunk hash:
  - same corpus fingerprint: the same questions are reused;
  - edited or deleted chunks: their questions are retired, and only new or
    changed chunks get new questions; the row is marked `corpus_changed`;
  - `--fresh-questions`, or a different `--questions` count: a new set.
- **Deltas.** Each history row stores per-question verdicts. The delta vs the
  previous run of the same kind is computed on the questions both runs share,
  and the shared count is stored with it.
- **API.** `ragaudit.history.load_history(store)` returns every run, oldest
  first, with deltas. The report reads only this.

New: `src/ragaudit/history.py`, `src/ragaudit/check_questions.py`.
Changed: `storage/store.py`, `cli_easy.py` (check), `cli.py` (`--note`,
`--fresh-questions`, history rows for `run`), `cli_v2.py` (history rows for
`evaluate` and `tune`).

## The new report

- One self-contained HTML file (`report/templates/v3/`), rendered by
  `report/v3.py` from a neutral view model, so v1 runs (`check`, `run`) and
  v2 results (`improve`) produce the same report.
- It is the default output of `check`, `run`, `report` and `improve`. The
  old template stays available as `ragaudit report --legacy`.
- Tabs are rendered server-side; JavaScript only switches tabs, draws the
  chart selection, copies commands and runs the popup and slides, so the
  page reads without JavaScript.

## `ragaudit improve`

- `cli_improve.py`: loads the docs folder with `load_corpus` (no JSONL),
  runs v2 `generate_questions` on a budget-sized, evenly spread set of
  chunks, then `run_tune` (baseline on dev, search on dev, baseline and best
  once on the locked test split), writes `best_config.yaml`, a history row
  and the report.
- Knobs come from the adapter's `KNOBS` dict (`--search-space` overrides).
  `run_tune` gains text matching (`chunk_texts`), because most adapters,
  including the demo one, return passages rather than ragaudit chunk ids.
- `scaffold.py`: every template gets a commented `KNOBS` and
  `make_adapter(config)`; the demo adapter has real knobs (chunk size,
  top_k, answer prompt).

## `ragaudit ui`

- `ui/server.py`: stdlib `ThreadingHTTPServer` on 127.0.0.1, a random token
  required on every request, Host and Origin checks, and paths confined to
  the project folder. It calls ragaudit functions directly (never a shell)
  and runs `check` or `improve` as a background job with polling.
- `ui/setup.html`: the 4-step wizard.
- `ragaudit` with no arguments prints a short welcome.

## Files touched

`pyproject.toml`, `README.md`, `CHANGELOG.md`, `docs/index.html`,
`docs/sample-report.html`, `src/ragaudit/{cli,cli_easy,cli_v2,scaffold}.py`,
new `src/ragaudit/{history,check_questions,cli_improve}.py`,
`src/ragaudit/storage/store.py`, `src/ragaudit/tune/loop.py`,
`src/ragaudit/models/evaluation.py`, new `src/ragaudit/report/v3.py` and
templates, new `src/ragaudit/ui/`, and tests for each.
