"""Lossless save/load functions for TestSet and Run.

Simple flat fields get their own columns; nested scoring structures
(ScoredCase, RunSummary) are stored as JSON via Pydantic's own
model_dump_json/model_validate_json, so round-tripping is exactly as
lossless as Pydantic's own serialization — no hand-rolled re-encoding
of nested objects.
"""

import json
import sqlite3
from datetime import datetime

from ragaudit.models.document import Chunk, Document
from ragaudit.models.run import Run, RunConfig, RunSummary
from ragaudit.models.scoring import ScoredCase
from ragaudit.models.testset import TestCase, TestSet


def save_documents(conn: sqlite3.Connection, documents: list[Document]) -> None:
    """Replace the full set of ingested documents with `documents`."""
    conn.execute("DELETE FROM documents")
    for document in documents:
        conn.execute(
            "INSERT INTO documents (doc_id, source_path, content, content_hash, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (document.doc_id, document.source_path, document.content, document.content_hash, json.dumps(document.metadata)),
        )
    conn.commit()


def load_documents(conn: sqlite3.Connection) -> list[Document]:
    """Load every ingested Document."""
    rows = conn.execute("SELECT doc_id, source_path, content, content_hash, metadata_json FROM documents").fetchall()
    return [
        Document(doc_id=r[0], source_path=r[1], content=r[2], content_hash=r[3], metadata=json.loads(r[4]))
        for r in rows
    ]


def save_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Replace the full set of ingested chunks with `chunks`."""
    conn.execute("DELETE FROM chunks")
    for chunk in chunks:
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, position, text, token_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk.chunk_id, chunk.doc_id, chunk.position, chunk.text, chunk.token_count, json.dumps(chunk.metadata)),
        )
    conn.commit()


def load_chunks(conn: sqlite3.Connection) -> list[Chunk]:
    """Load every ingested Chunk, ordered by document then position."""
    rows = conn.execute(
        "SELECT chunk_id, doc_id, position, text, token_count, metadata_json FROM chunks ORDER BY doc_id, position"
    ).fetchall()
    return [
        Chunk(chunk_id=r[0], doc_id=r[1], position=r[2], text=r[3], token_count=r[4], metadata=json.loads(r[5]))
        for r in rows
    ]


def get_latest_testset_id(conn: sqlite3.Connection) -> str | None:
    """Return the testset_id of the most recently created TestSet, or None if there is none."""
    row = conn.execute("SELECT testset_id FROM testsets ORDER BY created_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def get_latest_run_id(conn: sqlite3.Connection) -> str | None:
    """Return the run_id of the most recently started Run, or None if there is none."""
    row = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def save_test_set(conn: sqlite3.Connection, test_set: TestSet) -> None:
    """Persist a TestSet and its TestCases, replacing any existing rows with the same testset_id."""
    conn.execute("DELETE FROM test_cases WHERE testset_id = ?", (test_set.testset_id,))
    conn.execute("DELETE FROM testsets WHERE testset_id = ?", (test_set.testset_id,))

    conn.execute(
        "INSERT INTO testsets (testset_id, corpus_hash, created_at, generation_model) VALUES (?, ?, ?, ?)",
        (test_set.testset_id, test_set.corpus_hash, test_set.created_at.isoformat(), test_set.generation_model),
    )

    for position, test_case in enumerate(test_set.test_cases):
        conn.execute(
            """
            INSERT INTO test_cases (
                test_id, testset_id, position, question, expected_answer,
                source_chunk_id, source_doc_id, generation_model, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                test_case.test_id,
                test_set.testset_id,
                position,
                test_case.question,
                test_case.expected_answer,
                test_case.source_chunk_id,
                test_case.source_doc_id,
                test_case.generation_model,
                test_case.created_at.isoformat(),
                json.dumps(test_case.metadata),
            ),
        )

    conn.commit()


def load_test_set(conn: sqlite3.Connection, testset_id: str) -> TestSet | None:
    """Load a TestSet by id, with its TestCases in their original order. None if not found."""
    row = conn.execute(
        "SELECT testset_id, corpus_hash, created_at, generation_model FROM testsets WHERE testset_id = ?",
        (testset_id,),
    ).fetchone()
    if row is None:
        return None

    case_rows = conn.execute(
        """
        SELECT test_id, question, expected_answer, source_chunk_id, source_doc_id,
               generation_model, created_at, metadata_json
        FROM test_cases WHERE testset_id = ? ORDER BY position ASC
        """,
        (testset_id,),
    ).fetchall()

    test_cases = [
        TestCase(
            test_id=r[0],
            question=r[1],
            expected_answer=r[2],
            source_chunk_id=r[3],
            source_doc_id=r[4],
            generation_model=r[5],
            created_at=datetime.fromisoformat(r[6]),
            metadata=json.loads(r[7]),
        )
        for r in case_rows
    ]

    return TestSet(
        testset_id=row[0],
        corpus_hash=row[1],
        created_at=datetime.fromisoformat(row[2]),
        generation_model=row[3],
        test_cases=test_cases,
    )


def save_run(conn: sqlite3.Connection, run: Run) -> None:
    """Persist a Run: config, scored cases, and summary. Replaces any existing rows with the same run_id."""
    run_id = run.config.run_id
    conn.execute("DELETE FROM scored_cases WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    conn.execute(
        """
        INSERT INTO runs (
            run_id, testset_id, adapter_path, correctness_judge_model,
            groundedness_judge_model, run_ablation, started_at, finished_at, summary_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            run.config.testset_id,
            run.config.adapter_path,
            run.config.correctness_judge_model,
            run.config.groundedness_judge_model,
            int(run.config.run_ablation),
            run.config.started_at.isoformat(),
            run.finished_at.isoformat() if run.finished_at else None,
            run.summary.model_dump_json() if run.summary is not None else None,
        ),
    )

    for position, scored_case in enumerate(run.scored_cases):
        conn.execute(
            "INSERT INTO scored_cases (run_id, test_id, position, data_json) VALUES (?, ?, ?, ?)",
            (run_id, scored_case.test_id, position, scored_case.model_dump_json()),
        )

    conn.commit()


def load_run(conn: sqlite3.Connection, run_id: str) -> Run | None:
    """Load a Run by id, with its ScoredCases in their original order. None if not found."""
    row = conn.execute(
        """
        SELECT run_id, testset_id, adapter_path, correctness_judge_model,
               groundedness_judge_model, run_ablation, started_at, finished_at, summary_json
        FROM runs WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None

    config = RunConfig(
        run_id=row[0],
        testset_id=row[1],
        adapter_path=row[2],
        correctness_judge_model=row[3],
        groundedness_judge_model=row[4],
        run_ablation=bool(row[5]),
        started_at=datetime.fromisoformat(row[6]),
    )
    finished_at = datetime.fromisoformat(row[7]) if row[7] else None
    summary = RunSummary.model_validate_json(row[8]) if row[8] is not None else None

    case_rows = conn.execute(
        "SELECT data_json FROM scored_cases WHERE run_id = ? ORDER BY position ASC",
        (run_id,),
    ).fetchall()
    scored_cases = [ScoredCase.model_validate_json(r[0]) for r in case_rows]

    return Run(config=config, finished_at=finished_at, scored_cases=scored_cases, summary=summary)
