"""The contract users implement to plug their RAG pipeline into ragaudit."""

from typing import Protocol

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """The contract every adapter must return for a single query.

    retrieved_chunk_ids must be the chunk IDs ragaudit assigned during
    `ragaudit ingest` — the pipeline under test has to be indexed against
    ragaudit's chunks for retrieval scoring to mean anything. See README.
    """

    answer: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    raw: dict | None = None
    """Optional adapter-specific debug payload."""


class Adapter(Protocol):
    """The interface users implement to plug their RAG pipeline in."""

    def query(self, question: str) -> RAGResponse: ...


class ConfigurableAdapter(Protocol):
    """Optional v2 extension: an adapter that can be reconfigured, e.g. by `ragaudit tune`.

    Alternatively an adapter file can define `make_adapter(config: dict) -> Adapter`.
    Plain v1 adapters keep working everywhere a config isn't needed.
    """

    def query(self, question: str) -> RAGResponse: ...

    def configure(self, config: dict) -> None: ...
