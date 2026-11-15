"""The reusable question set of `ragaudit check`, so two checks are scored on the same questions.

Questions are keyed by the stable chunk id they came from
(`relative/path.md#position`, as v2's directory corpora use) and that
chunk's content hash, and kept in the project's store. Each check plans
its question set against the current docs:

- nothing changed: every question is reused, so scores stay comparable;
- a chunk was edited: its questions are retired and the same number is
  written from its new text;
- a chunk was deleted: its questions are retired;
- chunks were added: questions are written for them, at the same density
  as the existing set (at least one);
- `--fresh-questions`, a different `--questions` count, or no earlier
  set: a brand-new set, spread evenly over the docs.

Only new or changed chunks ever cost generation calls.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ragaudit.ingest.corpus import chunk_hash
from ragaudit.ingest.versioning import compute_index_version
from ragaudit.models.document import Chunk
from ragaudit.models.testset import TestCase
from ragaudit.storage.store import Store

META_TARGET = "check.target"
META_CHUNK_MAP = "check.chunk_map"
STABLE_ID_KEY = "stable_chunk_id"


def question_key(chunk_id: str, content_hash: str, question: str) -> str:
    return hashlib.sha256(f"{chunk_id}@{content_hash}\n{question.strip()}".encode("utf-8")).hexdigest()


@dataclass
class PooledQuestion:
    key: str
    chunk_id: str
    chunk_hash: str
    question: str
    expected_answer: str
    generation_model: str
    created_at: datetime


@dataclass
class QuestionPlan:
    reused: list[PooledQuestion]
    to_generate: list[tuple[Chunk, int]]
    """(stable-id chunk, questions to write from it)."""
    retire: dict[str, str] = field(default_factory=dict)
    """question key -> reason (changed, deleted, fresh)."""
    fresh: bool = False
    fresh_reason: str | None = None
    corpus_changed: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    """new / changed / deleted chunk counts vs the last check."""

    @property
    def generate_calls(self) -> int:
        return len(self.to_generate)

    @property
    def new_questions(self) -> int:
        return sum(n for _, n in self.to_generate)

    @property
    def questions(self) -> int:
        return len(self.reused) + self.new_questions


def spread(chunks: list, questions: int) -> list[tuple]:
    """Spread `questions` evenly over `chunks`: one each from evenly spaced chunks, or more per chunk when there are few."""
    if not chunks or questions <= 0:
        return []
    if questions <= len(chunks):
        return [(chunks[(i * len(chunks)) // questions], 1) for i in range(questions)]
    base, extra = divmod(questions, len(chunks))
    return [(chunk, base + 1 if i < extra else base) for i, chunk in enumerate(chunks)]


def load_pool(store: Store) -> list[PooledQuestion]:
    return [
        PooledQuestion(r[0], r[1], r[2], r[3], r[4], r[5], datetime.fromisoformat(r[6]))
        for r in store.load_check_questions()
    ]


def plan_questions(store: Store, chunks: list[Chunk], target: int, fresh: bool = False) -> QuestionPlan:
    """Decide which pooled questions to reuse and which chunks need new ones. Writes nothing."""
    current = {chunk.chunk_id: chunk_hash(chunk) for chunk in chunks}
    raw_map = store.get_meta(META_CHUNK_MAP)
    previous_map: dict[str, str] = json.loads(raw_map) if raw_map else {}
    previous_target = store.get_meta(META_TARGET)
    pool = load_pool(store)

    new_ids = [cid for cid in current if previous_map and cid not in previous_map]
    changed_ids = [cid for cid in current if cid in previous_map and previous_map[cid] != current[cid]]
    deleted_ids = [cid for cid in previous_map if cid not in current]
    counts = {"new": len(new_ids), "changed": len(changed_ids), "deleted": len(deleted_ids)}
    corpus_changed = bool(previous_map) and compute_index_version(previous_map) != compute_index_version(current)

    reason = None
    if fresh:
        reason = "--fresh-questions"
    elif not pool:
        reason = "first check" if not previous_map else "no saved questions"
    elif previous_target is not None and int(previous_target) != target:
        reason = f"--questions changed from {previous_target} to {target}"
    if reason is not None:
        return QuestionPlan(
            reused=[],
            to_generate=spread(chunks, target),
            retire={q.key: "fresh" for q in pool},
            fresh=True,
            fresh_reason=reason,
            corpus_changed=corpus_changed,
            counts=counts,
        )

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    reused, retire = [], {}
    retired_per_changed: dict[str, int] = {}
    for q in pool:
        if current.get(q.chunk_id) == q.chunk_hash:
            reused.append(q)
        elif q.chunk_id in current:
            retire[q.key] = "changed"
            retired_per_changed[q.chunk_id] = retired_per_changed.get(q.chunk_id, 0) + 1
        else:
            retire[q.key] = "deleted"

    to_generate = [(by_id[cid], n) for cid, n in retired_per_changed.items()]
    if new_ids:
        density = max(1, round(target * len(new_ids) / len(chunks)))
        to_generate += spread([by_id[cid] for cid in new_ids], density)
    if not reused and not to_generate:
        return QuestionPlan(
            reused=[], to_generate=spread(chunks, target), retire=retire, fresh=True,
            fresh_reason="no saved question matches the docs", corpus_changed=corpus_changed, counts=counts,
        )
    return QuestionPlan(reused=reused, to_generate=to_generate, retire=retire, corpus_changed=corpus_changed, counts=counts)


def commit_plan(store: Store, plan: QuestionPlan, chunks: list[Chunk], target: int) -> None:
    """Retire what the plan retires and remember the docs it was planned against."""
    reasons: dict[str, list[str]] = {}
    for key, reason in plan.retire.items():
        reasons.setdefault(reason, []).append(key)
    for reason, keys in reasons.items():
        store.retire_check_questions(keys, reason)
    store.set_meta(META_CHUNK_MAP, json.dumps({c.chunk_id: chunk_hash(c) for c in chunks}, sort_keys=True))
    store.set_meta(META_TARGET, str(target))


def add_generated(store: Store, chunk: Chunk, cases: list[TestCase]) -> list[PooledQuestion]:
    """Save freshly written questions for a stable-id chunk to the pool."""
    content_hash = chunk_hash(chunk)
    now = datetime.now(timezone.utc)
    pooled = []
    for case in cases:
        pooled.append(
            PooledQuestion(
                question_key(chunk.chunk_id, content_hash, case.question),
                chunk.chunk_id,
                content_hash,
                case.question,
                case.expected_answer,
                case.generation_model,
                now,
            )
        )
    store.save_check_questions(
        (q.key, q.chunk_id, q.chunk_hash, q.question, q.expected_answer, q.generation_model, q.created_at.isoformat())
        for q in pooled
    )
    return pooled


def to_test_case(q: PooledQuestion, source_chunk: Chunk) -> TestCase:
    """A pooled question as a v1 TestCase. Its test_id is the stable question key, so runs can be compared."""
    return TestCase(
        test_id=q.key,
        question=q.question,
        expected_answer=q.expected_answer,
        source_chunk_id=source_chunk.chunk_id,
        source_doc_id=source_chunk.doc_id,
        generation_model=q.generation_model,
        created_at=q.created_at,
        metadata={STABLE_ID_KEY: q.chunk_id},
    )
