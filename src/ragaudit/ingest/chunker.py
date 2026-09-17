"""Splits Document content into fixed-size, overlapping Chunk objects."""

import hashlib

from ragaudit.models.document import Chunk, Document

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100


def _chunk_id(doc_id: str, position: int, text: str) -> str:
    payload = f"{doc_id}:{position}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a Document's content into fixed-size chunks with overlap.

    Chunking is deterministic: chunk_id is a hash of (doc_id, position,
    text), so re-chunking the same document with the same parameters always
    produces the same chunk_ids.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = document.content
    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    position = 0
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size]
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(document.doc_id, position, piece),
                doc_id=document.doc_id,
                position=position,
                text=piece,
            )
        )
        position += 1
        start += step

    return chunks
