"""Models for generated test sets: questions with known ground-truth sources."""

from datetime import datetime

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """One generated question with its ground-truth source chunk."""

    test_id: str
    question: str
    expected_answer: str
    source_chunk_id: str
    source_doc_id: str
    """Denormalized, avoids a join for report rendering."""
    generation_model: str
    created_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)


class TestSet(BaseModel):
    """A generated batch of TestCases tied to a specific corpus state."""

    testset_id: str
    corpus_hash: str
    """Hash over all Document.content_hash, for reproducibility/staleness checks."""
    created_at: datetime
    generation_model: str
    test_cases: list[TestCase]
