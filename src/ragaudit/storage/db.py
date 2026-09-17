"""SQLite connection management and schema creation.

Migrations are intentionally trivial: every table is created with
`CREATE TABLE IF NOT EXISTS`, so re-opening an existing database is a
no-op. No migration framework, no versioning — if the schema needs to
change later, that's a deliberate future decision, not something this
module tries to anticipate.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS testsets (
    testset_id TEXT PRIMARY KEY,
    corpus_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    generation_model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS test_cases (
    test_id TEXT PRIMARY KEY,
    testset_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    source_chunk_id TEXT NOT NULL,
    source_doc_id TEXT NOT NULL,
    generation_model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    testset_id TEXT NOT NULL,
    adapter_path TEXT NOT NULL,
    correctness_judge_model TEXT NOT NULL,
    groundedness_judge_model TEXT NOT NULL,
    run_ablation INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS scored_cases (
    run_id TEXT NOT NULL,
    test_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (run_id, test_id)
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection at db_path and ensure the schema exists.

    db_path may be ":memory:" for an ephemeral in-process database.
    """
    connection = sqlite3.connect(str(db_path))
    connection.executescript(SCHEMA)
    connection.commit()
    return connection
