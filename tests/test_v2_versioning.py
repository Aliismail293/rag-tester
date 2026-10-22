from pathlib import Path

import pytest

from ragaudit.generate.pipeline import generate_questions
from ragaudit.ingest.corpus import load_corpus
from ragaudit.ingest.versioning import compute_index_version, diff_chunk_maps, sync_corpus
from ragaudit.llm.mock import MockLLM
from ragaudit.models.document import Chunk
from ragaudit.models.question import Question, QuestionStatus
from ragaudit.storage.lock import _unlock_test_split
from ragaudit.storage.store import Store

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "v2"


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", position=0, text=text)


def question_on(chunks: list[Chunk], fact: str) -> Question:
    from ragaudit.ingest.corpus import chunk_hash

    return Question.create(
        chunk_ids=[c.chunk_id for c in chunks],
        chunk_hashes=[chunk_hash(c) for c in chunks],
        fact=fact,
        question=f"{fact}?",
        gold_answer="x",
        generator_model="g",
    )


def statuses(store: Store) -> dict[str, QuestionStatus]:
    with _unlock_test_split():
        return {q.fact: q.status for q in store.load_questions(split=None, statuses=list(QuestionStatus))}


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "store.db") as s:
        yield s


def test_index_version_is_order_and_id_independent():
    assert compute_index_version(["b", "a"]) == compute_index_version(["a", "b"])
    assert compute_index_version({"x": "a", "y": "b"}) == compute_index_version({"p": "b", "q": "a"})
    assert compute_index_version(["a"]) != compute_index_version(["a", "b"])


def test_diff_classifies_new_changed_deleted():
    new, changed, deleted, unchanged = diff_chunk_maps({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9", "d": "4"})
    assert (new, changed, deleted, unchanged) == (["d"], ["b"], ["c"], 1)


def test_first_sync_records_a_version_and_everything_is_new(store):
    diff = sync_corpus(store, [chunk("a", "alpha"), chunk("b", "beta")])
    assert diff.new == ["a", "b"] and diff.previous_index_version is None
    assert store.corpus_version_count() == 1


def test_unchanged_corpus_does_not_record_a_new_version(store):
    chunks = [chunk("a", "alpha")]
    sync_corpus(store, chunks)
    diff = sync_corpus(store, chunks)
    assert not diff.index_changed and diff.unchanged == 1
    assert store.corpus_version_count() == 1


def test_changed_and_deleted_chunks_make_their_questions_stale(store):
    a, b, c = chunk("a", "alpha"), chunk("b", "beta"), chunk("c", "gamma")
    store.save_questions([question_on([a], "on a"), question_on([b], "on b"), question_on([c], "on c")])
    sync_corpus(store, [a, b, c])

    diff = sync_corpus(store, [a, chunk("b", "beta, edited")])
    assert diff.changed == ["b"] and diff.deleted == ["c"]
    assert diff.marked_stale == 2
    assert statuses(store) == {"on a": QuestionStatus.ACTIVE, "on b": QuestionStatus.STALE, "on c": QuestionStatus.STALE}


def test_multi_hop_question_goes_stale_when_either_chunk_changes(store):
    a, b = chunk("a", "alpha"), chunk("b", "beta")
    store.save_questions([question_on([a, b], "spans a and b")])
    sync_corpus(store, [a, chunk("b", "beta v2")])
    assert statuses(store)["spans a and b"] == QuestionStatus.STALE


def test_reverting_an_edit_revives_its_questions(store):
    a = chunk("a", "alpha")
    store.save_questions([question_on([a], "on a")])
    sync_corpus(store, [chunk("a", "alpha v2")])
    diff = sync_corpus(store, [a])
    assert diff.revived == 1
    assert statuses(store)["on a"] == QuestionStatus.ACTIVE


def test_rejected_questions_are_never_revived_or_staled(store):
    a = chunk("a", "alpha")
    rejected = question_on([a], "rejected").model_copy(update={"status": QuestionStatus.REJECTED, "reject_reason": "ambiguous"})
    store.save_questions([rejected])
    sync_corpus(store, [chunk("a", "alpha v2")])
    sync_corpus(store, [a])
    assert statuses(store)["rejected"] == QuestionStatus.REJECTED


def test_corpus_update_regenerates_only_new_and_changed_chunks(store):
    original = load_corpus(DEMO_DIR / "corpus.jsonl")
    updated = load_corpus(DEMO_DIR / "corpus_update.jsonl")
    generate_questions(store, original, MockLLM("mock-generator"), MockLLM("mock-verifier"))

    summary = generate_questions(store, updated, MockLLM("mock-generator"), MockLLM("mock-verifier"))
    assert summary.corpus_diff["new"] == 4
    assert summary.corpus_diff["changed"] == 3
    assert summary.corpus_diff["deleted"] == 2
    assert summary.corpus_diff["marked_stale"] > 0
    # Only the 4 new and 3 changed chunks are processed; the rest are skipped.
    assert summary.chunks_processed == 7
    assert summary.chunks_skipped == len(updated) - 7
