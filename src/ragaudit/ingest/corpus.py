"""Loads a v2 corpus: chunks with stable ids, content hashes and optional capture timestamps.

Two input formats:

- a `.jsonl` file, one chunk per line:
  `{"chunk_id": str, "text": str, "timestamp"?: ISO-8601, "doc_id"?: str, "position"?: int}`
- a directory of `.txt`/`.md` files, loaded and chunked with the v1 loader
  and chunker, but with stable ids `relative/path.md#position`.

v2 needs chunk ids that survive edits: a *changed* chunk (same id, new
hash) is handled differently from a new one. v1's chunk ids hash the
document content, so an edit gives every chunk a fresh id; that's why
directory corpora get path-based ids here instead.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ragaudit.ingest import chunker
from ragaudit.ingest.loader import load_documents
from ragaudit.models.document import Chunk

TIMESTAMP_KEY = "timestamp"


class CorpusError(Exception):
    """Raised when a corpus file is malformed."""


def chunk_hash(chunk: Chunk) -> str:
    """Content hash of a chunk's text."""
    return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 date or datetime; naive values are taken as UTC."""
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def chunk_timestamp(chunk: Chunk) -> datetime | None:
    value = chunk.metadata.get(TIMESTAMP_KEY)
    return parse_timestamp(value) if value else None


def _load_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise CorpusError(f"{path}:{line_no}: expected a JSON object")

        chunk_id = record.get("chunk_id")
        text = record.get("text")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise CorpusError(f"{path}:{line_no}: missing string field 'chunk_id'")
        if not isinstance(text, str) or not text.strip():
            raise CorpusError(f"{path}:{line_no}: missing non-empty string field 'text'")

        metadata: dict[str, str] = {}
        timestamp = record.get(TIMESTAMP_KEY)
        if timestamp:
            try:
                metadata[TIMESTAMP_KEY] = parse_timestamp(str(timestamp)).isoformat()
            except ValueError as exc:
                raise CorpusError(f"{path}:{line_no}: timestamp is not ISO-8601: {timestamp!r}") from exc
        for key, value in record.items():
            if key not in {"chunk_id", "text", "doc_id", "position", TIMESTAMP_KEY} and isinstance(value, str):
                metadata[key] = value

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=str(record.get("doc_id") or chunk_id),
                position=int(record.get("position", len(chunks))),
                text=text,
                metadata=metadata,
            )
        )
    return chunks


def _load_directory(root: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents(root):
        relative = Path(document.source_path).relative_to(root).as_posix()
        for chunk in chunker.chunk_document(document, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                chunk.model_copy(update={"chunk_id": f"{relative}#{chunk.position}", "doc_id": relative})
            )
    return chunks


def load_corpus(
    path: str | Path,
    chunk_size: int = chunker.DEFAULT_CHUNK_SIZE,
    overlap: int = chunker.DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Load a corpus from a .jsonl file or a directory of .txt/.md files. Chunk ids must be unique."""
    path = Path(path)
    if not path.exists():
        raise CorpusError(f"corpus not found: {path}")

    chunks = _load_directory(path, chunk_size, overlap) if path.is_dir() else _load_jsonl(path)

    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise CorpusError(f"duplicate chunk_id in corpus: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
    return chunks
