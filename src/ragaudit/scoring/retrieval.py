"""Pure set-membership retrieval scoring — no LLM, no network."""

from ragaudit.models.scoring import RetrievalResult


def score_retrieval(expected_chunk_id: str, retrieved_chunk_ids: list[str]) -> RetrievalResult:
    """Check whether the expected chunk appears among the retrieved chunk IDs.

    rank is the first matching index in retrieved_chunk_ids, or None if the
    expected chunk was not retrieved.
    """
    rank = retrieved_chunk_ids.index(expected_chunk_id) if expected_chunk_id in retrieved_chunk_ids else None
    return RetrievalResult(
        hit=rank is not None,
        expected_chunk_id=expected_chunk_id,
        retrieved_chunk_ids=retrieved_chunk_ids,
        rank=rank,
    )
