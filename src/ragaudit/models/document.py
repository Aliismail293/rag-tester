"""Models for ingested source documents and their chunks."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single ingested source file."""

    doc_id: str
    """Stable hash of source_path + content_hash."""
    source_path: str
    content: str
    content_hash: str
    """Sha256 of raw content, used to detect re-ingestion changes."""
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A stable, addressable slice of a Document."""

    chunk_id: str
    """Deterministic: hash(doc_id, position, text) — same corpus, same ids."""
    doc_id: str
    position: int
    """0-indexed order within the document."""
    text: str
    token_count: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
