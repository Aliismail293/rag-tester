import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ragaudit.models.adapter import RAGResponse
from ragaudit.models.run import Run, RunConfig, RunSummary
from ragaudit.models.scoring import (
    AblationResult,
    CorrectnessResult,
    GroundednessResult,
    RetrievalResult,
    ScoredCase,
    Verdict,
)
from ragaudit.models.testset import TestCase, TestSet
from ragaudit.storage.db import get_connection
from ragaudit.storage.repository import load_run, load_test_set, save_run, save_test_set


@pytest.fixture(params=[":memory:", "file"])
def conn(request, tmp_path):
    if request.param == ":memory:":
        connection = get_connection(":memory:")
    else:
        connection = get_connection(tmp_path / "ragaudit.db")
    yield connection
    connection.close()


def make_test_case(test_id: str) -> TestCase:
    return TestCase(
        test_id=test_id,
        question=f"question for {test_id}",
        expected_answer="an answer",
        source_chunk_id="chunk1",
        source_doc_id="doc1",
        generation_model="gen-model",
        created_at=datetime.now(timezone.utc),
        metadata={"difficulty": "easy"},
    )


def make_test_set() -> TestSet:
    return TestSet(
        testset_id="ts1",
        corpus_hash="corpus-hash-1",
        created_at=datetime.now(timezone.utc),
        generation_model="gen-model",
        test_cases=[make_test_case("t1"), make_test_case("t2"), make_test_case("t3")],
    )


def make_scored_case(test_id: str, verdict: Verdict, with_ablation: bool = False) -> ScoredCase:
    return ScoredCase(
        test_id=test_id,
        response=RAGResponse(
            answer="the answer",
            retrieved_chunk_ids=["chunk1", "chunk2"],
            contexts=["ctx a", "ctx b"],
            latency_ms=123.4,
            raw={"debug": "info"},
        ),
        correctness=CorrectnessResult(is_correct=True, score=0.9, rationale="good", judge_model="j1"),
        retrieval=RetrievalResult(hit=True, expected_chunk_id="chunk1", retrieved_chunk_ids=["chunk1", "chunk2"], rank=0),
        groundedness=GroundednessResult(is_grounded=True, score=0.8, rationale="supported", judge_model="j2"),
        verdict=verdict,
        ablation=(
            AblationResult(answer="no-context answer", is_correct=True, confirms_lucky_pass=True)
            if with_ablation
            else None
        ),
    )


def make_run(with_ablation: bool) -> Run:
    return Run(
        config=RunConfig(
            run_id="run1",
            testset_id="ts1",
            adapter_path="my_adapter.py",
            correctness_judge_model="j1",
            groundedness_judge_model="j2",
            run_ablation=with_ablation,
            started_at=datetime.now(timezone.utc),
        ),
        finished_at=datetime.now(timezone.utc),
        scored_cases=[
            make_scored_case("t1", Verdict.TRUE_PASS),
            make_scored_case("t2", Verdict.LUCKY_PASS, with_ablation=with_ablation),
            make_scored_case("t3", Verdict.RETRIEVAL_FAILURE),
        ],
        summary=RunSummary(
            total_cases=3,
            verdict_counts={
                Verdict.TRUE_PASS: 1,
                Verdict.LUCKY_PASS: 1,
                Verdict.GENERATION_FAILURE: 0,
                Verdict.RETRIEVAL_FAILURE: 1,
            },
            retrieval_hit_rate=2 / 3,
            answer_correctness_rate=2 / 3,
            lucky_pass_rate=1 / 3,
            mean_groundedness=0.8,
            ablation_confirmed_count=1 if with_ablation else None,
            judge_failure_count=0,
        ),
    )


def test_test_set_round_trip_is_lossless(conn):
    original = make_test_set()
    save_test_set(conn, original)
    loaded = load_test_set(conn, "ts1")
    assert loaded == original


def test_test_set_round_trip_preserves_order(conn):
    original = make_test_set()
    save_test_set(conn, original)
    loaded = load_test_set(conn, "ts1")
    assert [c.test_id for c in loaded.test_cases] == ["t1", "t2", "t3"]


def test_load_missing_test_set_returns_none(conn):
    assert load_test_set(conn, "does-not-exist") is None


def test_run_round_trip_is_lossless_with_ablation(conn):
    original = make_run(with_ablation=True)
    save_run(conn, original)
    loaded = load_run(conn, "run1")
    assert loaded == original
    assert loaded.scored_cases[1].ablation is not None
    assert loaded.scored_cases[1].ablation.confirms_lucky_pass is True


def test_run_round_trip_is_lossless_without_ablation(conn):
    original = make_run(with_ablation=False)
    save_run(conn, original)
    loaded = load_run(conn, "run1")
    assert loaded == original
    assert all(case.ablation is None for case in loaded.scored_cases)


def test_run_round_trip_preserves_scored_case_order(conn):
    original = make_run(with_ablation=True)
    save_run(conn, original)
    loaded = load_run(conn, "run1")
    assert [c.test_id for c in loaded.scored_cases] == ["t1", "t2", "t3"]


def test_load_missing_run_returns_none(conn):
    assert load_run(conn, "does-not-exist") is None


def test_saving_run_twice_overwrites_rather_than_duplicates(conn):
    run = make_run(with_ablation=False)
    save_run(conn, run)
    save_run(conn, run)
    loaded = load_run(conn, "run1")
    assert len(loaded.scored_cases) == 3
