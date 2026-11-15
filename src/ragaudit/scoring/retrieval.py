"""Pure retrieval scoring — no LLM, no network. By chunk id (set membership) or by text overlap."""

from ragaudit.models.scoring import RetrievalResult
from ragaudit.scoring.text_match import first_matching_context


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


def score_retrieval_by_text(
    expected_chunk_id: str,
    gold_text: str,
    retrieved_chunk_ids: list[str],
    contexts: list[str],
    threshold: float,
) -> RetrievalResult:
    """A hit if any returned context overlaps the gold chunk's text at or above threshold.

    rank is the index of the first matching context. retrieved_chunk_ids
    are kept as returned (they may be the pipeline's own ids).
    """
    rank, best = first_matching_context(gold_text, contexts, threshold)
    return RetrievalResult(
        hit=rank is not None,
        expected_chunk_id=expected_chunk_id,
        retrieved_chunk_ids=retrieved_chunk_ids,
        rank=rank,
        match_by="text",
        match_score=round(best, 4),
    )
