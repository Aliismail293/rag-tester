import sqlite3
from datetime import datetime, timezone

import pytest

from ragaudit.models.question import (
    Question,
    QuestionStatus,
    QuestionType,
    Split,
    assign_split,
    compute_question_id,
)
from ragaudit.storage.lock import LockedTestSetError, _unlock_test_split
from ragaudit.storage.store import SCHEMA_VERSION, Store, StoreSchemaError


def make_question(i: int, **overrides) -> Question:
    fields = dict(
        chunk_ids=[f"c{i}"],
        chunk_hashes=[f"h{i}"],
        fact=f"fact number {i}",
        question=f"question {i}?",
        gold_answer=f"answer {i}",
        generator_model="gen",
        verifier_model="ver",
    )
    fields.update(overrides)
    return Question.create(**fields)


def first_question_in(split: Split) -> Question:
    return next(make_question(i) for i in range(1000) if make_question(i).split == split)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / ".ragaudit" / "store.db") as s:
        yield s


# --- question identity and split ---


def test_question_id_is_stable_and_depends_on_chunk_hash_and_fact():
    a = compute_question_id(["h1"], "the sky is green")
    assert a == compute_question_id(["h1"], "the sky is green")
    assert a != compute_question_id(["h2"], "the sky is green")
    assert a != compute_question_id(["h1"], "the sky is blue")


def test_question_type_follows_number_of_chunks():
    assert make_question(1).question_type == QuestionType.SINGLE
    multi = make_question(1, chunk_ids=["a", "b"], chunk_hashes=["ha", "hb"])
    assert multi.question_type == QuestionType.MULTI_HOP


def test_split_is_deterministic_and_roughly_70_30_by_chunk():
    chunk_ids = [f"doc-{i}#0" for i in range(10_000)]
    splits = [assign_split(chunk_id) for chunk_id in chunk_ids]
    assert splits == [assign_split(chunk_id) for chunk_id in chunk_ids]
    dev_share = splits.count(Split.DEV) / len(splits)
    assert 0.68 < dev_share < 0.72


def test_created_at_is_absolute_utc():
    naive = make_question(1, created_at=datetime(2026, 1, 2, 3, 4, 5))
    assert naive.created_at.tzinfo == timezone.utc
    assert naive.created_at.isoformat() == "2026-01-02T03:04:05+00:00"


def test_adding_questions_never_moves_old_ones_between_splits(store):
    first_batch = [make_question(i) for i in range(100)]
    store.save_questions(first_batch)
    with _unlock_test_split():
        before = {q.question_id: q.split for q in store.load_questions()}

    store.save_questions([make_question(i) for i in range(100, 300)])
    with _unlock_test_split():
        after = {q.question_id: q.split for q in store.load_questions()}

    assert all(after[qid] == split for qid, split in before.items())


# --- store basics ---


def test_store_creates_parent_dir_and_records_schema_version(tmp_path):
    path = tmp_path / "nested" / "dir" / "store.db"
    with Store(path) as s:
        assert s.schema_version == SCHEMA_VERSION
    assert path.exists()


def test_store_refuses_newer_schema(tmp_path):
    path = tmp_path / "store.db"
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(StoreSchemaError):
        Store(path)


def test_question_round_trip_is_lossless(store):
    q = first_question_in(Split.DEV).model_copy(
        update={
            "source_timestamp": datetime(2025, 5, 1, tzinfo=timezone.utc),
            "lexical_overlap": 0.25,
            "metadata": {"k": "v"},
        }
    )
    store.save_questions([q])
    assert store.load_questions(split=Split.DEV) == [q]


def test_save_questions_is_idempotent(store):
    q = make_question(1)
    assert store.save_questions([q]) == 1
    assert store.save_questions([q]) == 0


def test_set_question_status_and_counts(store):
    dev_q = first_question_in(Split.DEV)
    test_q = first_question_in(Split.TEST)
    store.save_questions([dev_q, test_q])
    store.set_question_status([test_q.question_id], QuestionStatus.REJECTED, "too_lexical")

    counts = store.question_counts()
    assert counts["by_split_status"]["dev"]["active"] == 1
    assert counts["by_split_status"]["test"]["rejected"] == 1
    assert counts["reject_reasons"] == {"too_lexical": 1}


# --- the locked test set ---


def test_dev_questions_never_include_test_items(store):
    store.save_questions([make_question(i) for i in range(50)])
    dev = store.dev_questions()
    assert dev and all(q.split == Split.DEV for q in dev)


def test_reading_test_split_without_unlock_raises(store):
    store.save_questions([make_question(i) for i in range(50)])
    with pytest.raises(LockedTestSetError):
        store.load_questions(split=Split.TEST)
    with pytest.raises(LockedTestSetError):
        store.load_questions(split=None)


def test_rejected_test_questions_are_readable_since_they_are_never_scored(store):
    test_q = first_question_in(Split.TEST)
    store.save_questions([test_q])
    store.set_question_status([test_q.question_id], QuestionStatus.REJECTED, "ambiguous")
    rejected = store.load_questions(split=Split.TEST, statuses=[QuestionStatus.REJECTED])
    assert [q.question_id for q in rejected] == [test_q.question_id]


def test_unlock_is_scoped_to_the_block(store):
    store.save_questions([first_question_in(Split.TEST)])
    with _unlock_test_split():
        assert len(store.load_questions(split=Split.TEST)) == 1
    with pytest.raises(LockedTestSetError):
        store.load_questions(split=Split.TEST)


def test_question_ids_are_readable_for_either_split(store):
    test_q = first_question_in(Split.TEST)
    store.save_questions([test_q])
    assert store.question_ids(Split.TEST) == [test_q.question_id]
