"""The v2 store: questions, corpus versions, cached eval results, and tuning logs.

A separate SQLite file from v1's `ragaudit.db` (default `.ragaudit/store.db`)
so v1 tables and behaviour are never touched. Uses the stdlib `sqlite3`
module only, with a one-row `schema_version` table so a future schema
change can tell old stores apart from new ones.

Every read that could return test-split questions or results goes through
a guard: unless the final evaluation path in `ragaudit.runner.locked` has
unlocked the test split, it raises LockedTestSetError. Rejected questions
are exempt, since they are never scored and so can't leak anything.
"""

import json
import sqlite3
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from ragaudit.models.evaluation import EvalSummary, QuestionResult
from ragaudit.models.question import Question, QuestionStatus, QuestionType, Split
from ragaudit.models.tune import Trial, TuneResult
from ragaudit.storage.lock import LockedTestSetError, is_test_split_unlocked

DEFAULT_STORE_PATH = Path(".ragaudit") / "store.db"
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    question_id TEXT PRIMARY KEY,
    chunk_ids_json TEXT NOT NULL,
    chunk_hashes_json TEXT NOT NULL,
    fact TEXT NOT NULL,
    question TEXT NOT NULL,
    gold_answer TEXT NOT NULL,
    question_type TEXT NOT NULL,
    split TEXT NOT NULL,
    status TEXT NOT NULL,
    reject_reason TEXT,
    created_at TEXT NOT NULL,
    source_timestamp TEXT,
    generator_model TEXT NOT NULL,
    verifier_model TEXT,
    lexical_overlap REAL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_split_status ON questions (split, status);

CREATE TABLE IF NOT EXISTS corpus_versions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    index_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    chunk_map_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_attempts (
    source_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    outcome TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generate_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    question_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    index_version TEXT NOT NULL,
    split TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (question_id, config_hash, index_version)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    index_version TEXT NOT NULL,
    split TEXT NOT NULL,
    mode TEXT NOT NULL,
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tune_runs (
    tune_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    strategy TEXT NOT NULL,
    search_space_json TEXT NOT NULL,
    result_json TEXT
);

CREATE TABLE IF NOT EXISTS tune_trials (
    tune_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (tune_id, trial_index)
);
"""

_QUESTION_COLUMNS = (
    "question_id, chunk_ids_json, chunk_hashes_json, fact, question, gold_answer, question_type, split, "
    "status, reject_reason, created_at, source_timestamp, generator_model, verifier_model, lexical_overlap, "
    "metadata_json"
)


class StoreSchemaError(Exception):
    """Raised when the store was written by a newer, incompatible ragaudit."""


def _require_unlocked_for(splits: set[Split], statuses: set[QuestionStatus]) -> None:
    if Split.TEST not in splits or statuses == {QuestionStatus.REJECTED}:
        return
    if not is_test_split_unlocked():
        raise LockedTestSetError(
            "test-split questions and results are locked: they are only readable through the final "
            "evaluation path (ragaudit.runner.locked). Use the dev split for tuning."
        )


def _question_to_row(q: Question) -> tuple:
    return (
        q.question_id,
        json.dumps(q.chunk_ids),
        json.dumps(q.chunk_hashes),
        q.fact,
        q.question,
        q.gold_answer,
        q.question_type.value,
        q.split.value,
        q.status.value,
        q.reject_reason,
        q.created_at.isoformat(),
        q.source_timestamp.isoformat() if q.source_timestamp else None,
        q.generator_model,
        q.verifier_model,
        q.lexical_overlap,
        json.dumps(q.metadata),
    )


def _row_to_question(r: tuple) -> Question:
    return Question(
        question_id=r[0],
        chunk_ids=json.loads(r[1]),
        chunk_hashes=json.loads(r[2]),
        fact=r[3],
        question=r[4],
        gold_answer=r[5],
        question_type=QuestionType(r[6]),
        split=Split(r[7]),
        status=QuestionStatus(r[8]),
        reject_reason=r[9],
        created_at=datetime.fromisoformat(r[10]),
        source_timestamp=datetime.fromisoformat(r[11]) if r[11] else None,
        generator_model=r[12],
        verifier_model=r[13],
        lexical_overlap=r[14],
        metadata=json.loads(r[15]),
    )


class Store:
    """A v2 store backed by one SQLite file. Use as a context manager to close it."""

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.path = path
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(SCHEMA)
        self._check_schema_version()
        self._conn.commit()

    def _check_schema_version(self) -> None:
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        elif row[0] > SCHEMA_VERSION:
            raise StoreSchemaError(
                f"store {self.path} has schema version {row[0]}, but this ragaudit only understands "
                f"up to {SCHEMA_VERSION}. Upgrade ragaudit."
            )

    @property
    def schema_version(self) -> int:
        return self._conn.execute("SELECT version FROM schema_version").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- questions ---------------------------------------------------------

    def save_questions(self, questions: Iterable[Question]) -> int:
        """Insert questions, keeping any existing row with the same id. Returns how many were new.

        The id is a hash of the source content and fact, so an existing row
        with the same id is the same question; the first write wins.
        """
        inserted = 0
        for q in questions:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO questions ({_QUESTION_COLUMNS}) VALUES ({', '.join('?' * 16)})",
                _question_to_row(q),
            )
            inserted += cursor.rowcount
        self._conn.commit()
        return inserted

    def load_questions(
        self,
        split: Split | None = None,
        statuses: Iterable[QuestionStatus] = (QuestionStatus.ACTIVE,),
    ) -> list[Question]:
        """Load questions by split and status, oldest first.

        split=None means both splits. Any request that could include
        non-rejected test-split questions raises LockedTestSetError unless
        the final evaluation path has unlocked the test split.
        """
        status_set = set(statuses)
        split_set = {split} if split is not None else set(Split)
        _require_unlocked_for(split_set, status_set)

        split_marks = ", ".join("?" * len(split_set))
        status_marks = ", ".join("?" * len(status_set))
        rows = self._conn.execute(
            f"SELECT {_QUESTION_COLUMNS} FROM questions "
            f"WHERE split IN ({split_marks}) AND status IN ({status_marks}) "
            "ORDER BY created_at, question_id",
            (*[s.value for s in split_set], *[s.value for s in status_set]),
        ).fetchall()
        return [_row_to_question(r) for r in rows]

    def dev_questions(self) -> list[Question]:
        """Active dev-split questions, with answers. The only question view `tune` is given."""
        return self.load_questions(split=Split.DEV)

    def question_ids(self, split: Split, status: QuestionStatus = QuestionStatus.ACTIVE) -> list[str]:
        """Ids only (no text, no answers), so safe for either split."""
        rows = self._conn.execute(
            "SELECT question_id FROM questions WHERE split = ? AND status = ? ORDER BY created_at, question_id",
            (split.value, status.value),
        ).fetchall()
        return [r[0] for r in rows]

    def question_counts(self) -> dict:
        """Counts by split/status and by reject reason. No question text, so safe for either split."""
        by_split_status: dict[str, dict[str, int]] = {s.value: {st.value: 0 for st in QuestionStatus} for s in Split}
        for split, status, count in self._conn.execute(
            "SELECT split, status, COUNT(*) FROM questions GROUP BY split, status"
        ):
            by_split_status[split][status] = count
        reasons = Counter(
            {
                reason: count
                for reason, count in self._conn.execute(
                    "SELECT reject_reason, COUNT(*) FROM questions WHERE status = ? GROUP BY reject_reason",
                    (QuestionStatus.REJECTED.value,),
                )
            }
        )
        return {"by_split_status": by_split_status, "reject_reasons": dict(reasons)}

    def set_question_status(
        self, question_ids: Iterable[str], status: QuestionStatus, reject_reason: str | None = None
    ) -> int:
        """Set status (and reject_reason) on the given questions. Returns how many rows changed."""
        changed = 0
        for question_id in question_ids:
            cursor = self._conn.execute(
                "UPDATE questions SET status = ?, reject_reason = ? WHERE question_id = ?",
                (status.value, reject_reason, question_id),
            )
            changed += cursor.rowcount
        self._conn.commit()
        return changed

    def active_source_keys(self) -> set[str]:
        """Generation source keys ("chunk_id@hash" joined by "+") that already have active questions."""
        keys = set()
        for ids_json, hashes_json in self._conn.execute(
            "SELECT chunk_ids_json, chunk_hashes_json FROM questions WHERE status = ?",
            (QuestionStatus.ACTIVE.value,),
        ):
            pairs = zip(json.loads(ids_json), json.loads(hashes_json))
            keys.add("+".join(sorted(f"{chunk_id}@{chunk_hash}" for chunk_id, chunk_hash in pairs)))
        return keys

    # --- generation bookkeeping ---------------------------------------------

    def attempted_source_keys(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT source_key FROM generation_attempts")}

    def record_generation_attempt(self, source_key: str, outcome: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO generation_attempts (source_key, created_at, outcome) VALUES (?, ?, ?)",
            (source_key, datetime.now(timezone.utc).isoformat(), outcome),
        )
        self._conn.commit()

    def save_generate_run(self, run_id: str, created_at: datetime, summary_json: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO generate_runs (run_id, created_at, summary_json) VALUES (?, ?, ?)",
            (run_id, created_at.isoformat(), summary_json),
        )
        self._conn.commit()

    def load_generate_run(self, run_id: str | None = None) -> str | None:
        """Summary JSON of a generate run (the latest if run_id is None), or None."""
        if run_id is None:
            row = self._conn.execute("SELECT summary_json FROM generate_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        else:
            row = self._conn.execute("SELECT summary_json FROM generate_runs WHERE run_id = ?", (run_id,)).fetchone()
        return row[0] if row else None

    # --- corpus versions and staleness --------------------------------------

    def latest_corpus_version(self) -> tuple[str, dict[str, str]] | None:
        """(index_version, chunk_id -> chunk_hash) of the last recorded corpus, or None."""
        row = self._conn.execute(
            "SELECT index_version, chunk_map_json FROM corpus_versions ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def record_corpus_version(self, index_version: str, chunk_map: dict[str, str]) -> None:
        self._conn.execute(
            "INSERT INTO corpus_versions (index_version, created_at, chunk_map_json) VALUES (?, ?, ?)",
            (index_version, datetime.now(timezone.utc).isoformat(), json.dumps(chunk_map, sort_keys=True)),
        )
        self._conn.commit()

    def corpus_version_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM corpus_versions").fetchone()[0]

    def refresh_staleness(self, current_chunks: dict[str, str]) -> tuple[int, int]:
        """Mark active questions stale if any source chunk changed or vanished; revive stale ones whose sources are back.

        Reads only ids and hashes, never question text or answers, so it
        runs over both splits. Returns (marked_stale, revived).
        """
        to_stale, to_revive = [], []
        for question_id, ids_json, hashes_json, status in self._conn.execute(
            "SELECT question_id, chunk_ids_json, chunk_hashes_json, status FROM questions WHERE status IN (?, ?)",
            (QuestionStatus.ACTIVE.value, QuestionStatus.STALE.value),
        ):
            sources_present = all(
                current_chunks.get(chunk_id) == chunk_hash
                for chunk_id, chunk_hash in zip(json.loads(ids_json), json.loads(hashes_json))
            )
            if status == QuestionStatus.ACTIVE.value and not sources_present:
                to_stale.append(question_id)
            elif status == QuestionStatus.STALE.value and sources_present:
                to_revive.append(question_id)
        self.set_question_status(to_stale, QuestionStatus.STALE)
        self.set_question_status(to_revive, QuestionStatus.ACTIVE)
        return len(to_stale), len(to_revive)

    # --- evaluation results (the cache) ------------------------------------

    def save_result(self, result: QuestionResult, config_hash: str, index_version: str) -> None:
        """Cache one result under (question_id, config_hash, index_version)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO eval_results "
            "(question_id, config_hash, index_version, split, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.question_id,
                config_hash,
                index_version,
                result.split,
                datetime.now(timezone.utc).isoformat(),
                result.model_dump_json(),
            ),
        )
        self._conn.commit()

    def load_result(self, question_id: str, config_hash: str, index_version: str) -> QuestionResult | None:
        """A cached result, or None. Test-split results are locked like test questions (judge rationales can echo answers)."""
        row = self._conn.execute(
            "SELECT split, data_json FROM eval_results WHERE question_id = ? AND config_hash = ? AND index_version = ?",
            (question_id, config_hash, index_version),
        ).fetchone()
        if row is None:
            return None
        _require_unlocked_for({Split(row[0])}, {QuestionStatus.ACTIVE})
        return QuestionResult.model_validate_json(row[1])

    def result_question_ids(self, config_hash: str, split: Split) -> set[str]:
        """Ids of questions with a cached result for this config at any index version. Ids only, so safe."""
        rows = self._conn.execute(
            "SELECT DISTINCT question_id FROM eval_results WHERE config_hash = ? AND split = ?",
            (config_hash, split.value),
        )
        return {r[0] for r in rows}

    def save_eval_run(self, summary: EvalSummary) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO eval_runs "
            "(run_id, created_at, config_hash, config_json, index_version, split, mode, summary_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                summary.run_id,
                summary.created_at.isoformat(),
                summary.config_hash,
                json.dumps(summary.config, sort_keys=True),
                summary.index_version,
                summary.split,
                summary.mode,
                summary.model_dump_json(),
            ),
        )
        self._conn.commit()

    def last_eval_run(self, config_hash: str, split: Split) -> EvalSummary | None:
        """The latest evaluation summary for this config and split (aggregates only, so safe for test)."""
        row = self._conn.execute(
            "SELECT summary_json FROM eval_runs WHERE config_hash = ? AND split = ? ORDER BY created_at DESC LIMIT 1",
            (config_hash, split.value),
        ).fetchone()
        return EvalSummary.model_validate_json(row[0]) if row else None

    # --- tuning logs -------------------------------------------------------

    def save_tune_run(self, tune_id: str, created_at: datetime, strategy: str, search_space: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tune_runs (tune_id, created_at, strategy, search_space_json) VALUES (?, ?, ?, ?)",
            (tune_id, created_at.isoformat(), strategy, json.dumps(search_space)),
        )
        self._conn.commit()

    def save_tune_trial(self, tune_id: str, trial: Trial) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tune_trials (tune_id, trial_index, created_at, config_hash, data_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (tune_id, trial.index, trial.created_at.isoformat(), trial.config_hash, trial.model_dump_json()),
        )
        self._conn.commit()

    def finish_tune_run(self, tune_id: str, finished_at: datetime, result_json: str) -> None:
        self._conn.execute(
            "UPDATE tune_runs SET finished_at = ?, result_json = ? WHERE tune_id = ?",
            (finished_at.isoformat(), result_json, tune_id),
        )
        self._conn.commit()

    def load_tune_result(self, tune_id: str | None = None) -> TuneResult | None:
        """A finished tune run (the latest if tune_id is None), or None. Test scores in it are aggregates only."""
        query = "SELECT result_json FROM tune_runs WHERE result_json IS NOT NULL"
        if tune_id is None:
            row = self._conn.execute(f"{query} ORDER BY created_at DESC LIMIT 1").fetchone()
        else:
            row = self._conn.execute(f"{query} AND tune_id = ?", (tune_id,)).fetchone()
        return TuneResult.model_validate_json(row[0]) if row else None

    def load_tune_trials(self, tune_id: str) -> list[Trial]:
        rows = self._conn.execute(
            "SELECT data_json FROM tune_trials WHERE tune_id = ? ORDER BY trial_index", (tune_id,)
        ).fetchall()
        return [Trial.model_validate_json(r[0]) for r in rows]
