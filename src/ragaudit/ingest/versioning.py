"""Corpus versions and staleness: what changed since the last generate/evaluate run.

`index_version` is a hash of the sorted set of current chunk hashes. Each
run diffs the current corpus (chunk_id -> chunk_hash) against the last
recorded one:

- new chunk: no questions yet, so `generate` will write some;
- changed chunk (same id, new hash): its old questions are marked stale,
  and `generate` writes new ones for the new content;
- deleted chunk: its questions are marked stale.

Staleness is decided per question from every (chunk_id, chunk_hash) it
depends on, so a multi-hop question goes stale if either chunk changes.
If a corpus edit is reverted, the questions it made stale become active
again, because their source content is back exactly as it was.
"""

import hashlib

from pydantic import BaseModel, Field

from ragaudit.ingest.corpus import chunk_hash
from ragaudit.models.document import Chunk
from ragaudit.storage.store import Store


class CorpusDiff(BaseModel):
    index_version: str
    previous_index_version: str | None = None
    new: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    unchanged: int = 0
    marked_stale: int = 0
    revived: int = 0

    @property
    def index_changed(self) -> bool:
        return self.index_version != self.previous_index_version

    def counts(self) -> dict[str, int]:
        return {
            "new": len(self.new),
            "changed": len(self.changed),
            "deleted": len(self.deleted),
            "unchanged": self.unchanged,
            "marked_stale": self.marked_stale,
            "revived": self.revived,
        }


def chunk_map(chunks: list[Chunk]) -> dict[str, str]:
    return {chunk.chunk_id: chunk_hash(chunk) for chunk in chunks}


def compute_index_version(chunk_hashes: list[str] | dict[str, str]) -> str:
    """Hash of the sorted set of chunk hashes: same content, same version, whatever the order or ids."""
    hashes = chunk_hashes.values() if isinstance(chunk_hashes, dict) else chunk_hashes
    return hashlib.sha256("\n".join(sorted(set(hashes))).encode("utf-8")).hexdigest()


def diff_chunk_maps(previous: dict[str, str], current: dict[str, str]) -> tuple[list[str], list[str], list[str], int]:
    """(new ids, changed ids, deleted ids, unchanged count), ids sorted."""
    new = sorted(cid for cid in current if cid not in previous)
    changed = sorted(cid for cid in current if cid in previous and previous[cid] != current[cid])
    deleted = sorted(cid for cid in previous if cid not in current)
    unchanged = sum(1 for cid in current if previous.get(cid) == current[cid])
    return new, changed, deleted, unchanged


def sync_corpus(store: Store, chunks: list[Chunk]) -> CorpusDiff:
    """Diff the corpus against the last recorded version, update question staleness, record the new version."""
    current = chunk_map(chunks)
    index_version = compute_index_version(current)
    last = store.latest_corpus_version()
    previous_version, previous_map = last if last is not None else (None, {})

    new, changed, deleted, unchanged = diff_chunk_maps(previous_map, current)
    marked_stale, revived = store.refresh_staleness(current)
    if last is None or previous_version != index_version or previous_map != current:
        store.record_corpus_version(index_version, current)

    return CorpusDiff(
        index_version=index_version,
        previous_index_version=previous_version,
        new=new,
        changed=changed,
        deleted=deleted,
        unchanged=unchanged,
        marked_stale=marked_stale,
        revived=revived,
    )
