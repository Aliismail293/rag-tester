"""The dev/test split is per chunk: no chunk ever feeds questions to both splits."""

import json
import sqlite3
from pathlib import Path

import pytest

from ragaudit.generate import prompts
from ragaudit.generate.pipeline import GenerateOptions, build_checked_question, candidate_pairs, generate_questions
from ragaudit.ingest.corpus import load_corpus
from ragaudit.llm.mock import MockLLM
from ragaudit.models.document import Chunk
from ragaudit.models.question import (
    Question,
    QuestionStatus,
    QuestionType,
    RejectReason,
    Split,
    assign_split,
    split_for_chunks,
)
from ragaudit.storage.lock import _unlock_test_split
from ragaudit.storage.store import SCHEMA_VERSION, Store

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "v2"


def chunk_ids_in(split: Split, n: int, prefix: str = "chunk") -> list[str]:
    ids = (f"{prefix}-{i}" for i in range(10_000))
    return [chunk_id for chunk_id in ids if assign_split(chunk_id) == split][:n]


def make(chunk_ids: list[str], fact: str = "a fact", hashes: list[str] | None = None, **kwargs) -> Question:
    return Question.create(
        chunk_ids=chunk_ids,
        chunk_hashes=hashes or [f"hash-of-{c}" for c in chunk_ids],
        fact=fact,
        question=f"question about {fact}?",
        gold_answer="answer",
        generator_model="gen",
        **kwargs,
    )


def all_questions(store: Store) -> list[Question]:
    with _unlock_test_split():
        return store.load_questions(split=None, statuses=list(QuestionStatus))


def chunks_by_split(questions: list[Question]) -> dict[Split, set[str]]:
    seen: dict[Split, set[str]] = {Split.DEV: set(), Split.TEST: set()}
    for q in questions:
        if q.split is not None:
            seen[q.split].update(q.chunk_ids)
    return seen


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "store.db") as s:
        yield s


# --- assignment ---


def test_split_is_a_function_of_the_chunk_id_and_roughly_70_30():
    ids = [f"guide.md#{i}" for i in range(10_000)]
    splits = [assign_split(chunk_id) for chunk_id in ids]
    assert splits == [assign_split(chunk_id) for chunk_id in ids]
    assert 0.68 < splits.count(Split.DEV) / len(splits) < 0.72


def test_every_question_from_a_chunk_inherits_its_split():
    for split in Split:
        for chunk_id in chunk_ids_in(split, 5):
            questions = [make([chunk_id], fact=f"fact {i}") for i in range(20)]
            assert {q.split for q in questions} == {split}


def test_editing_a_chunk_never_moves_it_between_splits():
    for chunk_id in chunk_ids_in(Split.DEV, 10) + chunk_ids_in(Split.TEST, 10):
        before = make([chunk_id], hashes=["original content hash"])
        after = make([chunk_id], hashes=["edited content hash"])
        assert before.question_id != after.question_id
        assert before.split == after.split == assign_split(chunk_id)


def test_multi_hop_takes_the_split_only_when_every_chunk_agrees():
    dev = chunk_ids_in(Split.DEV, 2)
    test = chunk_ids_in(Split.TEST, 2)
    assert make(dev).split == Split.DEV and make(dev).status == QuestionStatus.ACTIVE
    assert make(test).split == Split.TEST and make(test).status == QuestionStatus.ACTIVE

    cross = make([dev[0], test[0]], status=QuestionStatus.ACTIVE)
    assert cross.question_type == QuestionType.MULTI_HOP
    assert cross.split is None
    assert cross.status == QuestionStatus.REJECTED
    assert cross.reject_reason == RejectReason.CROSS_SPLIT.value
    assert split_for_chunks([dev[0], test[0]]) is None


def test_only_rejected_questions_may_have_no_split():
    with pytest.raises(ValueError, match="no split"):
        Question(**{**make(chunk_ids_in(Split.DEV, 1)).model_dump(), "split": None})


# --- generation ---


def chunk(chunk_id: str, text: str, doc_id: str = "d", position: int = 0) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, position=position, text=text)


def test_candidate_pairs_never_cross_splits():
    ids = chunk_ids_in(Split.DEV, 6, "p") + chunk_ids_in(Split.TEST, 6, "p")
    chunks = [chunk(chunk_id, f"Shared depot tram words {i}", position=i) for i, chunk_id in enumerate(sorted(ids))]
    pairs = candidate_pairs(chunks)
    assert pairs
    assert all(split_for_chunks([c.chunk_id for c in pair]) is not None for pair in pairs)


def test_a_cross_split_question_is_rejected_without_paying_for_checks():
    class CountingLLM:
        model = "counting"
        calls = 0

        def complete(self, system, user):
            CountingLLM.calls += 1
            raise AssertionError("no LLM call expected for a cross-split question")

    a = chunk(chunk_ids_in(Split.DEV, 1)[0], "The north depot holds 12 trams.")
    b = chunk(chunk_ids_in(Split.TEST, 1)[0], "The south depot holds 9 buses.", position=1)
    question, _ = build_checked_question(CountingLLM(), CountingLLM(), [a, b], "fact", "21", "How many vehicles?", 0.5)
    assert CountingLLM.calls == 0
    assert question.status == QuestionStatus.REJECTED and question.reject_reason == "cross_split"
    assert question.split is None


def test_a_cross_split_reject_round_trips_through_the_store(store):
    cross = make([chunk_ids_in(Split.DEV, 1)[0], chunk_ids_in(Split.TEST, 1)[0]])
    store.save_questions([cross])
    [loaded] = all_questions(store) or [None]
    assert loaded is None  # no split, so neither split's query returns it
    counts = store.question_counts()
    assert counts["by_split_status"]["none"]["rejected"] == 1
    assert counts["reject_reasons"] == {"cross_split": 1}


def test_no_chunk_appears_in_both_splits_after_generating_the_demo(store):
    chunks = load_corpus(DEMO_DIR / "corpus.jsonl")
    generate_questions(store, chunks, MockLLM("mock-generator"), MockLLM("mock-verifier"), GenerateOptions(multi_hop=True))
    questions = all_questions(store)
    assert any(q.question_type == QuestionType.MULTI_HOP and q.status == QuestionStatus.ACTIVE for q in questions)

    seen = chunks_by_split(questions)
    assert seen[Split.DEV] and seen[Split.TEST]
    assert not seen[Split.DEV] & seen[Split.TEST]
    for q in questions:
        if q.split is not None:
            assert all(assign_split(chunk_id) == q.split for chunk_id in q.chunk_ids)

    # The 70/30 target is at the chunk level.
    chunk_splits = [assign_split(c.chunk_id) for c in chunks]
    assert {c.chunk_id for c in chunks if assign_split(c.chunk_id) == Split.DEV} >= seen[Split.DEV]
    assert 0.5 < chunk_splits.count(Split.DEV) / len(chunk_splits) < 0.9


def test_split_is_stable_across_reruns_and_chunk_edits(tmp_path):
    original = load_corpus(DEMO_DIR / "corpus.jsonl")
    updated = load_corpus(DEMO_DIR / "corpus_update.jsonl")
    edited = {c.chunk_id for c in updated} & {c.chunk_id for c in original}
    edited = {cid for cid in edited if next(c.text for c in original if c.chunk_id == cid) != next(c.text for c in updated if c.chunk_id == cid)}
    assert edited  # the demo update edits some chunks in place

    def run(path: Path, corpora: list[list[Chunk]]) -> list[Question]:
        with Store(path) as s:
            for corpus in corpora:
                generate_questions(s, corpus, MockLLM("mock-generator"), MockLLM("mock-verifier"), GenerateOptions(multi_hop=True))
            return all_questions(s)

    once = run(tmp_path / "a.db", [original])
    twice = run(tmp_path / "b.db", [original, original])
    # A rerun adds questions (the multi-hop pair cap is per run) but never moves an existing one.
    assert {(q.question_id, q.split) for q in once} <= {(q.question_id, q.split) for q in twice}
    assert not chunks_by_split(twice)[Split.DEV] & chunks_by_split(twice)[Split.TEST]

    after_edit = run(tmp_path / "c.db", [original, updated])
    split_of = {}
    for q in after_edit:
        for chunk_id in q.chunk_ids:
            if q.split is not None:
                assert split_of.setdefault(chunk_id, q.split) == q.split
    for chunk_id in edited:
        old = {q.split for q in after_edit if chunk_id in q.chunk_ids and q.status == QuestionStatus.STALE}
        new = {q.split for q in after_edit if chunk_id in q.chunk_ids and q.status == QuestionStatus.ACTIVE}
        assert old | new == {assign_split(chunk_id)}
    assert not chunks_by_split(after_edit)[Split.DEV] & chunks_by_split(after_edit)[Split.TEST]


# --- migrating a schema-1 store (per-question splits) ---

V1_QUESTIONS = """
CREATE TABLE questions (
    question_id TEXT PRIMARY KEY, chunk_ids_json TEXT NOT NULL, chunk_hashes_json TEXT NOT NULL,
    fact TEXT NOT NULL, question TEXT NOT NULL, gold_answer TEXT NOT NULL, question_type TEXT NOT NULL,
    split TEXT NOT NULL, status TEXT NOT NULL, reject_reason TEXT, created_at TEXT NOT NULL,
    source_timestamp TEXT, generator_model TEXT NOT NULL, verifier_model TEXT, lexical_overlap REAL,
    metadata_json TEXT NOT NULL
);
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version (version) VALUES (1);
CREATE TABLE eval_results (
    question_id TEXT NOT NULL, config_hash TEXT NOT NULL, index_version TEXT NOT NULL, split TEXT NOT NULL,
    created_at TEXT NOT NULL, data_json TEXT NOT NULL, PRIMARY KEY (question_id, config_hash, index_version)
);
"""


def test_a_schema_1_store_is_re_split_per_chunk_on_open(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(V1_QUESTIONS)
    dev_chunk, test_chunk = chunk_ids_in(Split.DEV, 1)[0], chunk_ids_in(Split.TEST, 1)[0]
    # Schema 1 split per question, so the same chunk could sit on both sides.
    rows = [
        ("q1", [dev_chunk], "test", "active"),
        ("q2", [dev_chunk], "dev", "active"),
        ("q3", [test_chunk], "dev", "stale"),
        ("q4", [dev_chunk, test_chunk], "dev", "active"),
    ]
    for qid, chunk_ids, split, status in rows:
        conn.execute(
            "INSERT INTO questions VALUES (?, ?, ?, 'f', 'q?', 'a', ?, ?, ?, NULL, '2026-01-01T00:00:00+00:00', NULL, 'g', NULL, NULL, '{}')",
            (qid, json.dumps(chunk_ids), json.dumps(["h"] * len(chunk_ids)), "multi_hop" if len(chunk_ids) > 1 else "single", split, status),
        )
        conn.execute("INSERT INTO eval_results VALUES (?, 'cfg', 'idx', ?, 'now', '{}')", (qid, split))
    conn.commit()
    conn.close()

    with Store(path) as store:
        assert store.schema_version == SCHEMA_VERSION
        by_id = {q.question_id: q for q in all_questions(store)}
        assert by_id["q1"].split == by_id["q2"].split == Split.DEV
        assert by_id["q3"].split == Split.TEST and by_id["q3"].status == QuestionStatus.STALE
        assert "q4" not in by_id  # cross-split: rejected with no split
        assert store.question_counts()["by_split_status"]["none"]["rejected"] == 1
        # Cached results of questions whose split changed are dropped; q2 didn't move.
        assert store.result_question_ids("cfg", Split.DEV) == {"q2"}
        assert store.result_question_ids("cfg", Split.TEST) == set()
        assert not chunks_by_split(list(by_id.values()))[Split.DEV] & chunks_by_split(list(by_id.values()))[Split.TEST]

    with Store(path) as store:  # reopening is a no-op
        assert store.schema_version == SCHEMA_VERSION
        assert len(all_questions(store)) == 3
